import os
import sys
import warnings

from flask import Flask, Response, stream_with_context

from middle_layer.security import PublicBindWithoutAuthError as _PublicBindWithoutAuthError
from middle_layer.security import apply_security_headers as _apply_security_headers
from middle_layer.security import check_api_key as _check_api_key
from middle_layer.security import enforce_safe_bind as _enforce_safe_bind
from middle_layer.security import resolve_max_request_bytes as _resolve_max_request_bytes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = _resolve_max_request_bytes()

try:
    from litellm import completion as litellm_completion
    _litellm_import_error = None
except Exception as _e:  # noqa: BLE001
    litellm_completion = None
    _litellm_import_error = str(_e)

# LM Studio configuration - use the correct API endpoint
LM_STUDIO_URL = os.environ.get('LM_STUDIO_URL', 'http://127.0.0.1:1234')
LM_STUDIO_MODELS_ENDPOINT = f"{LM_STUDIO_URL}/v1/models"

# Anthropic configuration (optional). If ANTHROPIC_API_KEY is set, we can route "big" tasks to Opus.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-4-opus-20250522")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2025-04-14")
USE_LITELLM_FOR_ANTHROPIC = os.environ.get("USE_LITELLM_FOR_ANTHROPIC", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
ENABLE_LITELLM_PREFIX_ROUTING = os.environ.get("ENABLE_LITELLM_PREFIX_ROUTING", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
LITELLM_TIMEOUT_SECONDS = int(os.environ.get("LITELLM_TIMEOUT_SECONDS", "120"))

# Routing knobs: default local-first; only escalate when clearly big.
BIG_TASK_MIN_WORDS = int(os.environ.get("BIG_TASK_MIN_WORDS", "80"))
BIG_TASK_MIN_CHARS = int(os.environ.get("BIG_TASK_MIN_CHARS", "500"))
BIG_TASK_MIN_BULLETS = int(os.environ.get("BIG_TASK_MIN_BULLETS", "4"))
BIG_TASK_MIN_STEP_MARKERS = int(os.environ.get("BIG_TASK_MIN_STEP_MARKERS", "3"))

# Optional incoming request authentication (recommended even on localhost).
# If MIDDLE_LAYER_API_KEY is set, requests must include header X-API-Key matching it.
MIDDLE_LAYER_API_KEY = os.environ.get("MIDDLE_LAYER_API_KEY")

# Cache for discovered model ID (with timestamp to avoid stale cache)
_cached_model_id = None
_cache_timestamp = 0
CACHE_TTL_SECONDS = 60

# ---------------------------------------------------------------------------
# Multi-model & swarm configuration
# ---------------------------------------------------------------------------

# The HTTP client (probes + chat call + per-instance caches) lives in
# ``middle_layer/lmstudio_client.py`` (Pass 3); the gateway keeps one
# instance plus thin module-level wrappers so historical monkey-patching
# of ``mod.get_lmstudio_model_ids`` et al. keeps working.
MODEL_LIST_TTL = int(os.environ.get("MODEL_LIST_TTL", "30"))

from middle_layer.lmstudio_client import (
    LMStudioClient,  # noqa: E402
    is_chat_capable_model_id,  # noqa: E402, F401
)

_LMSTUDIO = LMStudioClient(LM_STUDIO_URL, model_list_ttl=MODEL_LIST_TTL)

# Prefer LM Studio model ids that are already loaded over not-loaded ones.
# Three modes:
#   0 / false / no / off       — legacy: treat every installed id as equally
#                                available; first-match-wins ordering.
#   1 / true / yes / on        — prefer loaded ids; fall back to the installed
#                                set if nothing loaded matches a request.
#   strict / only / 2 (def.)   — *only* use loaded ids when LM Studio reports
#                                at least one loaded model. Role/DEFAULT_MODEL
#                                preferences and explicit-id requests never
#                                fall through to the installed set, so MiddleLayer
#                                won't silently ask LM Studio to JIT-load a
#                                different model than the one(s) you have resident.
#                                A request for a specific not-loaded id returns 503
#                                (or whatever ``ON_MODEL_MISS`` says).
#
# Default changed in 0.1.x from "1" -> "strict" so the in-process default
# matches the launcher (``scripts/start.sh --profile lmstudio``). Unset
# environments emit a one-shot DeprecationWarning explaining how to pin the
# legacy "prefer-loaded with installed fallthrough" behavior.
_PREFER_LOADED_DEFAULT = "strict"
_PREFER_LOADED_LEGACY_DEFAULT = "1"
_PREFER_LOADED_ENV = os.environ.get("PREFER_LOADED_MODELS")
if _PREFER_LOADED_ENV is None:
    # Dynamic-by-default: stick to whatever LM Studio currently has loaded
    # and never silently JIT-load installed-but-not-loaded ids. The old
    # default ("1" = prefer-loaded with fall-through to the installed set)
    # caused chat requests to JIT giant models behind the operator's back
    # when a role/DEFAULT_MODEL preference happened to substring-match an
    # installed id before any loaded one. AGENTS.md non-negotiable #1
    # (deprecation path): keep the legacy default available via env var
    # for one minor version and warn so operators can flip back.
    warnings.warn(
        "PREFER_LOADED_MODELS is unset: defaulting to 'strict' (was "
        f"{_PREFER_LOADED_LEGACY_DEFAULT!r}) so MiddleLayer never JIT-loads "
        "installed-but-not-loaded LM Studio models. Set "
        "PREFER_LOADED_MODELS=1 explicitly to keep the legacy prefer-loaded "
        "behavior; will be removed in 0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    _PREFER_LOADED_RAW = _PREFER_LOADED_DEFAULT
else:
    _PREFER_LOADED_RAW = _PREFER_LOADED_ENV.strip().lower()
_PREFER_LOADED_STRICT_TOKENS = {"strict", "only", "2", "loaded-only", "loaded_only"}
_PREFER_LOADED_OFF_TOKENS = {"0", "false", "no", "off"}
PREFER_LOADED_MODELS = _PREFER_LOADED_RAW not in _PREFER_LOADED_OFF_TOKENS
STRICT_LOADED_MODELS = _PREFER_LOADED_RAW in _PREFER_LOADED_STRICT_TOKENS

# Tokens that mean "you pick a model for me". OpenClaw-specific ids are gated
# behind EXTRA_PLACEHOLDER_MODELS (see middle_layerMLX.py for the shared policy).
_CORE_PLACEHOLDER_MODELS = frozenset({
    "", "auto", "default",
    "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4.1",
    "claude-3-5-sonnet", "claude-3-opus",
})
_OPENCLAW_DEFAULT_PLACEHOLDERS = frozenset({
    "middlelayer", "middle-layer", "middle_layer",
    "mlxmiddlelayer", "mlx-middle-layer", "mlx_middle_layer", "mlx",
    "lmstudio", "openclaw",
})


def _build_effective_placeholder_models() -> frozenset[str]:
    raw = os.environ.get("EXTRA_PLACEHOLDER_MODELS")
    if raw is None:
        warnings.warn(
            "EXTRA_PLACEHOLDER_MODELS is unset: OpenClaw-specific placeholder model "
            "IDs remain enabled for one minor release. Set EXTRA_PLACEHOLDER_MODELS "
            "to a comma-separated list (or empty string to disable) for explicit "
            "control. Defaults change in 0.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        extra = _OPENCLAW_DEFAULT_PLACEHOLDERS
    elif raw.strip() == "":
        extra = frozenset()
    else:
        extra = frozenset(s.strip().lower() for s in raw.split(",") if s.strip())
    return frozenset(_CORE_PLACEHOLDER_MODELS | extra)


PLACEHOLDER_MODELS = _build_effective_placeholder_models()

# When the client asks for a specific model that is NOT loaded:
#   "fallback" (default) -> auto-pick another model + add X-Model-Resolution header
#   "error"              -> return 400 so the caller can react
ON_MODEL_MISS = os.environ.get("ON_MODEL_MISS", "fallback").lower()

# Optional preferred default. Match is case-insensitive substring or exact.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "").strip()

# Role registry. role -> ordered list of preferred model ids OR substrings.
# Configure via:
#   MODEL_ROLES_JSON='{"coder":["qwen2.5-coder"],"fast":["3b"]}'
#   MODEL_ROLES_FILE=/path/to/roles.json
# Resolution logic lives in ``middle_layer/resolver.py`` (Pass 3); the
# wrappers below keep the historical module-level surface stable.
from middle_layer import resolver as _resolver  # noqa: E402
from middle_layer.resolver import DEFAULT_MODEL_ROLES  # noqa: E402, F401

_HERE = os.path.dirname(os.path.abspath(__file__))


def _autodiscover_roles_file():
    """Back-compat wrapper over ``middle_layer.resolver.autodiscover_roles_file``,
    anchored at this script's directory."""
    return _resolver.autodiscover_roles_file(_HERE)


def _load_model_roles():
    return _resolver.load_model_roles(_HERE)


MODEL_ROLES, MODEL_ROLES_SOURCE = _load_model_roles()

# Swarm concurrency knobs. A typical Mac can run two reasonably-sized models
# in parallel; one big + one small is the safe default.
# Swarm primitives now live in ``middle_layer/swarm.py``. We re-export the
# names that pre-Pass-3 callers (and tests) referenced at module level so
# this is a pure relocation, not a behavior change. New code should import
# from ``middle_layer.swarm`` directly.
from middle_layer.cloud_escalation import (  # noqa: E402
    BigTaskThresholds,
    CloudEscalationClient,
)
from middle_layer.cloud_escalation import (
    extract_user_intent_text as _extract_user_intent_text_impl,
)
from middle_layer.swarm import (  # noqa: E402, F401
    # F401 is global to this block on purpose: every name here is a
    # back-compat re-export for pre-Pass-3 callers that historically
    # imported from middle_layer.py. Use middle_layer.swarm directly
    # in new code; this surface stays stable until 0.4.0.
    LM_STUDIO_PER_MODEL_INFLIGHT_CAP,
    MAX_PARALLEL_MODEL_CALLS,
    SWARM_CHAT_DEFAULT_JUDGE,
    SWARM_CHAT_DEFAULT_MODELS,
    SWARM_CHAT_DEFAULT_STRATEGY,
    SWARM_CHAT_ENABLED,
    SWARM_PER_CALL_TIMEOUT,
    SWARM_STREAM_CHUNK_CHARS,
    _per_model_semaphore,
    _per_model_semaphores,
)

_CLOUD = CloudEscalationClient(
    anthropic_api_key=ANTHROPIC_API_KEY,
    anthropic_base_url=ANTHROPIC_BASE_URL,
    anthropic_model=ANTHROPIC_MODEL,
    anthropic_version=ANTHROPIC_VERSION,
    use_litellm_for_anthropic=USE_LITELLM_FOR_ANTHROPIC,
    litellm_timeout_seconds=LITELLM_TIMEOUT_SECONDS,
    big_task=BigTaskThresholds(
        min_words=BIG_TASK_MIN_WORDS,
        min_chars=BIG_TASK_MIN_CHARS,
        min_bullets=BIG_TASK_MIN_BULLETS,
        min_step_markers=BIG_TASK_MIN_STEP_MARKERS,
    ),
    litellm_completion=litellm_completion,
    litellm_import_error=_litellm_import_error,
    default_chat_timeout=SWARM_PER_CALL_TIMEOUT,
)


def _litellm_available() -> bool:
    return _CLOUD.litellm_available()


def _litellm_model_for_anthropic(model_name: str) -> str:
    from middle_layer.cloud_escalation import litellm_model_for_anthropic

    return litellm_model_for_anthropic(model_name)


def _litellm_response_to_dict(resp):
    from middle_layer.cloud_escalation import litellm_response_to_dict

    return litellm_response_to_dict(resp)


def _call_litellm_chat(messages, model_override=None, **kwargs):
    return _CLOUD.call_litellm_chat(messages, model_override=model_override, **kwargs)


def _should_route_to_anthropic(endpoint: str, json_data: dict) -> bool:
    return _CLOUD.should_route_to_anthropic(endpoint, json_data)


def _extract_user_intent_text(json_data: dict) -> str:
    return _extract_user_intent_text_impl(json_data)


def _openai_messages_to_anthropic(json_data: dict) -> dict:
    from middle_layer.cloud_escalation import openai_messages_to_anthropic

    return openai_messages_to_anthropic(json_data, default_model=ANTHROPIC_MODEL)


def _anthropic_to_openai_chat_completion(anthropic_json: dict) -> dict:
    from middle_layer.cloud_escalation import anthropic_to_openai_chat_completion

    return anthropic_to_openai_chat_completion(
        anthropic_json, anthropic_model=ANTHROPIC_MODEL
    )


def _looks_like_code(text_lower: str) -> bool:
    from middle_layer.cloud_escalation import looks_like_code

    return looks_like_code(text_lower)


def _is_big_task(text: str) -> bool:
    from middle_layer.cloud_escalation import is_big_task

    return is_big_task(text, thresholds=_CLOUD.big_task)


def get_lmstudio_model_ids(force_refresh: bool = False):
    """
    Return (list_of_model_ids, error_message). Lists every currently loaded
    model id on the LM Studio server, in the order LM Studio reports them.
    Briefly cached (MODEL_LIST_TTL) so swarm fanouts don't hammer the API.
    """
    return _LMSTUDIO.get_model_ids(force_refresh=force_refresh)


def get_loaded_lmstudio_model_ids(force_refresh: bool = False):
    """Return (loaded_ids, error) using LM Studio's ``/api/v0/models`` endpoint
    (which exposes per-instance ``state``). Degrades gracefully when the
    endpoint isn't supported (older LM Studio) — see
    ``middle_layer.lmstudio_client.LMStudioClient.get_loaded_model_ids``.
    """
    return _LMSTUDIO.get_loaded_model_ids(force_refresh=force_refresh)


# Back-compat alias for callers that imported the underscore-prefixed name.
# (``is_chat_capable_model_id`` itself is re-exported from
# ``middle_layer.lmstudio_client`` at the top of this file.)
_is_chat_capable_model_id = is_chat_capable_model_id


def get_loaded_chat_capable_lmstudio_model_ids(force_refresh: bool = False):
    """Loaded-ids list filtered to chat-capable ids only. Pairs with
    ``get_loaded_lmstudio_model_ids`` (which returns *all* loaded ids,
    including embedding models). Used by swarm ``auto`` expansion so a
    default-shaped ``swarmCouncil`` call against a box that also has an
    embedding model loaded doesn't fan out to that embedding model.
    """
    ids, err = get_loaded_lmstudio_model_ids(force_refresh=force_refresh)
    if err:
        return ids, err
    return [m for m in ids if is_chat_capable_model_id(m)], None


def get_current_lmstudio_model():
    """
    Backwards-compatible single-model accessor. Returns (model_id, error)
    where model_id is the first currently loaded LM Studio model. Prefers a
    truly-loaded id when ``/api/v0/models`` is reachable; otherwise falls back
    to the first id in the configured-model list. Prefer
    ``get_lmstudio_model_ids`` / ``resolve_model_id`` for new code.
    """
    loaded, lerr = get_loaded_lmstudio_model_ids()
    if not lerr and loaded:
        return loaded[0], None
    ids, err = get_lmstudio_model_ids()
    if err:
        return None, err
    if not ids:
        return None, "No model is loaded in LM Studio."
    return ids[0], None


def _resolver_policy() -> _resolver.ResolverPolicy:
    """Snapshot the gateway's resolver knobs.

    Built per-call (cheap dataclass) so tests and the dashboard can mutate
    the module-level globals and the resolver sees the change immediately.
    """
    return _resolver.ResolverPolicy(
        roles=MODEL_ROLES,
        prefer_loaded=bool(PREFER_LOADED_MODELS),
        strict_loaded=bool(STRICT_LOADED_MODELS),
        default_model=DEFAULT_MODEL,
        placeholder_ids=PLACEHOLDER_MODELS,
    )


def _is_placeholder(name) -> bool:
    """True when `name` is empty / a generic placeholder / a known cloud id."""
    return _resolver.is_placeholder(name, PLACEHOLDER_MODELS)


def _match_one(needle: str, haystack):
    """First id in `haystack` matching `needle` (exact then substring, case-insensitive)."""
    return _resolver.match_one(needle, haystack)


def _resolve_role(role: str, available, loaded=None):
    """First model id matching any preference for ``role``.

    Delegates to ``middle_layer.resolver.resolve_role`` with this gateway's
    live policy (prefer-loaded / strict-loaded semantics documented there).
    """
    return _resolver.resolve_role(role, available, loaded, policy=_resolver_policy())


def resolve_model_id(requested, available=None, loaded=None):
    """
    Decide which loaded LM Studio model id to use for a request.

    Accepted shapes for `requested`:
      None / "" / "auto" / "default" / "middleLayer"  -> auto-pick
      "exact-model-id"                                -> exact, else substring
      "a,b,c"                                         -> priority list (first match wins)
      "role:coder"                                    -> registry lookup
      "*coder*" / "qwen*"                             -> wildcard substring
      mix any of the above in a comma-separated list, e.g. "role:coder,qwen*"

    When ``PREFER_LOADED_MODELS`` is on and LM Studio's ``/api/v0/models``
    endpoint reports any ``state=loaded`` ids, every match is attempted
    against the loaded subset first; not-loaded ids are only returned if the
    loaded subset cannot satisfy the request. The caller can pass
    ``available`` and ``loaded`` explicitly to avoid re-probing in a hot loop.

    Returns (model_id, error_message). On a soft miss (specific name asked but
    not loaded), error is non-None; the caller decides whether to fall back.
    """
    if available is None:
        available, err = get_lmstudio_model_ids()
        if err:
            return None, err
    if not available:
        return None, "No model is loaded in LM Studio."

    if loaded is None and PREFER_LOADED_MODELS:
        loaded, _lerr = get_loaded_lmstudio_model_ids()
        # _lerr is non-fatal: on probe failure we just proceed against `available`.

    return _resolver.resolve_model_id(requested, available, loaded, policy=_resolver_policy())


# ===========================================================================
# SWARM / MULTI-AGENT
# ---------------------------------------------------------------------------
# Design (high level):
#   - /swarm/models        : inventory + role registry (debug / discovery)
#   - /swarm/fanout        : same prompt -> N models in parallel, return all
#   - /swarm/vote          : fanout + best-of-n via a judge model (or
#                            "first-success" / longest-fallback)
#   - /swarm/pipeline      : sequential chain of model steps with
#                            {{previous}} / {{step_name}} substitution in
#                            the per-step `system` and `user` templates.
#
# A "model" inside a swarm spec can be:
#   - an exact LM Studio id        e.g. "qwen2.5-coder-32b-instruct"
#   - a comma list (priority)      e.g. "qwen2.5-coder,qwen2.5-7b"
#   - a role                       e.g. "role:coder"
#   - a wildcard substring         e.g. "*coder*"
#   - "anthropic" or               e.g. "anthropic:claude-4-opus-20250522"
#     "anthropic:<model>"          (requires ANTHROPIC_API_KEY)
#
# Concurrency is bounded by MAX_PARALLEL_MODEL_CALLS so a single Mac doesn't
# OOM. Fanout/vote use a ThreadPoolExecutor with that bound.
# ===========================================================================


def _lmstudio_chat_completion(model_id, messages, **kwargs):
    """Call LM Studio /v1/chat/completions for a single model.
    Returns (openai_shaped_response_json, error_str)."""
    kwargs.setdefault("timeout", SWARM_PER_CALL_TIMEOUT)
    return _LMSTUDIO.chat_completion(model_id, messages, **kwargs)


def _call_anthropic_chat(messages, model_override=None, **kwargs):
    """Call Anthropic / LiteLLM and return an OpenAI-shaped chat completion."""
    return _CLOUD.call_anthropic_chat(messages, model_override=model_override, **kwargs)


# Pure swarm helpers re-exported here so historical call sites (and tests
# that monkey-patch ``mod._classify_swarm_error`` etc.) keep working without
# updating to the new module path. New code should import from
# ``middle_layer.swarm`` directly.
from middle_layer.swarm import (  # noqa: E402, F401  # noqa: E402, F401
    # Back-compat re-exports — see the block above.
    _SWARM_AUTO_TOKENS,
    _SWARM_ERROR_KINDS,
    _classify_swarm_error,
    _extract_text,
    _extract_upstream_status,
    _is_auto_swarm_token,
    _normalize_agent_spec,
    _strip_upstream_prefix,
)
from middle_layer.swarm import SWARM_CHAT_AUTO_MAX as _SWARM_CHAT_AUTO_MAX  # noqa: E402
from middle_layer.swarm import expand_swarm_models as _swarm_expand_models  # noqa: E402


def _swarm_auto_pool():
    """Loaded-and-chat-capable pool used by swarm ``auto``/``loaded``/``*``
    sentinels. Tries the truly-loaded probe (``/api/v0/models``) first and
    filters out embedding models; falls back to the installed list
    (``/v1/models``, with the same chat-capable filter) when the loaded
    probe isn't reachable (older LM Studio, network blip, …).
    """
    loaded, err = get_loaded_chat_capable_lmstudio_model_ids()
    if not err and loaded:
        return loaded, None
    installed, ierr = get_lmstudio_model_ids()
    if ierr:
        return [], err or ierr
    return [m for m in installed if is_chat_capable_model_id(m)], None


def _expand_swarm_models(spec, available=None, *, apply_auto_cap: bool = False):
    """Thin gateway wrapper around ``middle_layer.swarm.expand_swarm_models``
    that wires in the LM Studio loaded-models probe so sentinel tokens
    (``auto`` / ``loaded`` / ``*`` / ``all`` / ``all-loaded``) can resolve
    against the actual LM Studio inventory, filtered to chat-capable ids
    only. ``apply_auto_cap`` is opt-in: callers that want the
    ``SWARM_CHAT_AUTO_MAX`` cap (default swarm chat completions) set it;
    the dedicated ``/swarm/fanout`` HTTP endpoint leaves it off so explicit
    ``models: "auto"`` callers still get every loaded chat model.
    """
    return _swarm_expand_models(
        spec,
        available=available,
        fetch_loaded=_swarm_auto_pool,
        max_auto_entries=_SWARM_CHAT_AUTO_MAX if apply_auto_cap else None,
    )


from middle_layer import swarm as _swarm_runner  # noqa: E402
from middle_layer.swarm import (  # noqa: E402, F401
    # Back-compat re-exports — see the block above.
    _SWARM_CHAT_CANONICAL,
    _SWARM_CHAT_INTENTS,
    _is_swarm_chat_model,
    _summarize_failed_candidates,
    _swarm_alias_warned,
    _swarm_chat_intent,
)

# Single ``SwarmDeps`` for this gateway. Built lazily so module-level globals
# defined further down (``ANTHROPIC_MODEL``, etc.) are already in scope.
_SWARM_DEPS: _swarm_runner.SwarmDeps | None = None


def _swarm_deps() -> _swarm_runner.SwarmDeps:
    """Lazily-built dependency bundle for the gateway-agnostic swarm runners.
    Cached after first call so repeat dispatches don't re-allocate.
    """
    global _SWARM_DEPS
    if _SWARM_DEPS is None:
        _SWARM_DEPS = _swarm_runner.SwarmDeps(
            chat_completion=_lmstudio_chat_completion,
            anthropic_chat=_call_anthropic_chat,
            resolve_model_id=resolve_model_id,
            get_available_models=get_lmstudio_model_ids,
            get_loaded_models=get_loaded_lmstudio_model_ids,
            extract_user_intent=_extract_user_intent_text,
            anthropic_default_model=ANTHROPIC_MODEL,
            prefer_loaded_models=PREFER_LOADED_MODELS,
        )
    return _SWARM_DEPS


def _run_one_agent(spec, default_messages, default_kwargs, available, loaded=None):
    """Back-compat thin wrapper around ``swarm.run_one_agent``. New code
    should call ``swarm.run_one_agent`` directly with a SwarmDeps."""
    return _swarm_runner.run_one_agent(
        spec, default_messages, default_kwargs, available, _swarm_deps(), loaded=loaded
    )


def _fanout(specs, messages, common_kwargs, max_parallel=None):
    """Back-compat thin wrapper around ``swarm.fanout``."""
    return _swarm_runner.fanout(
        specs, messages, common_kwargs, _swarm_deps(), max_parallel=max_parallel
    )


def _run_swarm_chat_completion(requested_model: str, json_data: dict, intent: str = "council"):
    """Back-compat thin wrapper around ``swarm.run_swarm_chat_completion``."""
    return _swarm_runner.run_swarm_chat_completion(
        requested_model, json_data, _swarm_deps(), intent=intent
    )


def _swarm_body_to_sse_response(body: dict, *, chunk_chars: int = SWARM_STREAM_CHUNK_CHARS):
    """Wrap the gateway-agnostic SSE generator in a Flask ``Response`` so
    streaming-only clients can consume swarm chat completions. Headers
    (``X-Model-Routed-To`` / ``X-Swarm-Strategy`` / ``X-Swarm-Winner``)
    come from ``swarm.swarm_response_headers``; we add transport headers
    (``Cache-Control``, ``X-Accel-Buffering``) here.
    """
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        **_swarm_runner.swarm_response_headers(body),
    }
    return Response(
        stream_with_context(
            _swarm_runner.stream_swarm_body_as_sse(body, chunk_chars=chunk_chars)
        ),
        status=200,
        mimetype="text/event-stream",
        headers=headers,
    )


from middle_layer.lmstudio_routes import (  # noqa: E402
    LmStudioRouteContext,
    register_lmstudio_routes,
)

register_lmstudio_routes(
    app,
    LmStudioRouteContext(
        lm_studio_url=LM_STUDIO_URL,
        lmstudio_client=_LMSTUDIO,
        middle_layer_api_key=MIDDLE_LAYER_API_KEY,
        model_roles=MODEL_ROLES,
        model_roles_source=MODEL_ROLES_SOURCE,
        prefer_loaded_models=bool(PREFER_LOADED_MODELS),
        strict_loaded_models=bool(STRICT_LOADED_MODELS),
        default_model=DEFAULT_MODEL,
        on_model_miss=ON_MODEL_MISS,
        anthropic_model=ANTHROPIC_MODEL,
        anthropic_enabled=bool(ANTHROPIC_API_KEY),
        enable_litellm_prefix_routing=bool(ENABLE_LITELLM_PREFIX_ROUTING),
        litellm_for_anthropic=bool(USE_LITELLM_FOR_ANTHROPIC),
        litellm_import_error=_litellm_import_error,
        swarm_chat_enabled=bool(SWARM_CHAT_ENABLED),
        max_parallel=MAX_PARALLEL_MODEL_CALLS,
        per_model_inflight_cap=LM_STUDIO_PER_MODEL_INFLIGHT_CAP,
        swarm_chat_default_models=SWARM_CHAT_DEFAULT_MODELS,
        swarm_chat_default_strategy=SWARM_CHAT_DEFAULT_STRATEGY,
        swarm_auto_tokens=_SWARM_AUTO_TOKENS,
        swarm_chat_canonical=_SWARM_CHAT_CANONICAL,
        swarm_chat_intents=_SWARM_CHAT_INTENTS,
        check_api_key=_check_api_key,
        apply_security_headers=_apply_security_headers,
        get_model_ids=get_lmstudio_model_ids,
        get_loaded_model_ids=get_loaded_lmstudio_model_ids,
        litellm_available=_litellm_available,
        should_route_to_anthropic=_should_route_to_anthropic,
        call_anthropic_chat=_call_anthropic_chat,
        call_litellm_chat=_call_litellm_chat,
        swarm_chat_intent=_swarm_chat_intent,
        run_swarm_chat_completion=_run_swarm_chat_completion,
        swarm_body_to_sse_response=_swarm_body_to_sse_response,
        resolve_model_id=resolve_model_id,
        is_placeholder=_is_placeholder,
        expand_swarm_models=_expand_swarm_models,
        fanout=_fanout,
        run_one_agent=_run_one_agent,
        extract_text=_extract_text,
        extract_user_intent=_extract_user_intent_text,
        lmstudio_chat_completion=_lmstudio_chat_completion,
        per_model_semaphore=_per_model_semaphore,
    ),
)


def main() -> None:
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        _enforce_safe_bind(host, MIDDLE_LAYER_API_KEY)
    except _PublicBindWithoutAuthError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Starting middle_layer on port {port}...")
    print(f"Listening on host: {host}")
    print(f"Max request body: {app.config['MAX_CONTENT_LENGTH']} bytes")
    print(f"LM Studio URL: {LM_STUDIO_URL}")
    if MIDDLE_LAYER_API_KEY:
        print("Auth: enabled (X-API-Key required)")
    else:
        print("Auth: disabled (set MIDDLE_LAYER_API_KEY to enable)")

    print(f"Model miss policy: {ON_MODEL_MISS}")
    if STRICT_LOADED_MODELS:
        print(
            "Loaded-model policy: STRICT (PREFER_LOADED_MODELS=strict) — "
            "MiddleLayer will only use LM Studio's currently-loaded models; "
            "role/DEFAULT_MODEL preferences never JIT-load installed-but-not-loaded ids."
        )
    elif PREFER_LOADED_MODELS:
        print("Loaded-model policy: prefer-loaded (fall back to installed set if no loaded match)")
    else:
        print("Loaded-model policy: legacy (no preference for loaded ids; first-match wins)")
    print(f"Max parallel model calls (swarm): {MAX_PARALLEL_MODEL_CALLS}")
    if _litellm_available():
        print("LiteLLM: available")
    else:
        print(f"LiteLLM: unavailable ({_litellm_import_error})")
    print(f"LiteLLM Anthropic routing: {'enabled' if USE_LITELLM_FOR_ANTHROPIC else 'disabled'}")
    print(f"LiteLLM prefix routing: {'enabled' if ENABLE_LITELLM_PREFIX_ROUTING else 'disabled'}")
    print(f"Swarm chat routing: {'enabled' if SWARM_CHAT_ENABLED else 'disabled'}")
    print(f"Swarm chat strategy: {SWARM_CHAT_DEFAULT_STRATEGY}")
    print(f"Swarm chat models: {SWARM_CHAT_DEFAULT_MODELS}")
    if DEFAULT_MODEL:
        print(f"Default model preference: {DEFAULT_MODEL}")
    if MODEL_ROLES:
        print("Roles configured:")
        for role, prefs in MODEL_ROLES.items():
            print(f"  - {role}: {prefs}")

    ids, error = get_lmstudio_model_ids(force_refresh=True)
    if error:
        print(f"WARN: {error}")
    elif not ids:
        print("WARN: LM Studio reachable but no models loaded.")
    else:
        print(f"OK: {len(ids)} model(s) loaded:")
        for mid in ids:
            print(f"  - {mid}")

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
