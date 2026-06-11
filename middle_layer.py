import os
import sys
import json
import time
import warnings
import requests
import re
import uuid
from flask import Flask, request, Response, stream_with_context

from middle_layer.security import apply_security_headers as _apply_security_headers
from middle_layer.security import check_api_key as _check_api_key
from middle_layer.security import enforce_safe_bind as _enforce_safe_bind
from middle_layer.security import PublicBindWithoutAuthError as _PublicBindWithoutAuthError
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

from middle_layer.lmstudio_client import LMStudioClient  # noqa: E402
from middle_layer.lmstudio_client import is_chat_capable_model_id  # noqa: E402, F401

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


def _litellm_available() -> bool:
    return litellm_completion is not None


def _litellm_model_for_anthropic(model_name: str) -> str:
    name = (model_name or "").strip()
    if "/" in name:
        return name
    return f"anthropic/{name}"


def _litellm_response_to_dict(resp):
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    # Best effort fallback for unexpected response objects.
    return json.loads(json.dumps(resp, default=str))


def _call_litellm_chat(messages, model_override=None, **kwargs):
    """Call LiteLLM chat completion and return OpenAI-shaped JSON."""
    if not _litellm_available():
        return None, f"LiteLLM not available: {_litellm_import_error or 'import failed'}"

    payload = {
        "model": model_override,
        "messages": messages or [],
        "stream": False,
    }
    for k in ("max_tokens", "temperature", "top_p", "stop"):
        if kwargs.get(k) is not None:
            payload[k] = kwargs[k]

    try:
        resp = litellm_completion(**payload, timeout=kwargs.get("timeout", LITELLM_TIMEOUT_SECONDS))
        return _litellm_response_to_dict(resp), None
    except Exception as e:  # noqa: BLE001
        return None, f"LiteLLM error: {e}"


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


def _looks_like_code(text_lower: str) -> bool:
    return bool(
        re.search(r"[{};()\[\]]", text_lower)
        or re.search(r"\b(def|class|function|var|let|const|import|from|#include|traceback|stack trace)\b", text_lower)
        or "```" in text_lower
    )


def _is_big_task(text: str) -> bool:
    t = (text or "").strip()
    tl = t.lower()

    words = re.findall(r"\w+", t)
    word_count = len(words)
    char_count = len(t)
    bullet_count = len(re.findall(r"^\s*([-*]|\d+\.)\s+", t, flags=re.MULTILINE))

    step_markers = [
        "step ", "steps", "phase", "phases", "roadmap", "milestone",
        "end-to-end", "from scratch", "system design", "architecture",
        "tradeoff", "trade-offs", "pros and cons", "migration plan",
        "rollout", "roll-out", "risk", "risks",
    ]
    step_score = sum(1 for m in step_markers if m in tl)

    # Long/multi-step requests are "big".
    if word_count >= BIG_TASK_MIN_WORDS or char_count >= BIG_TASK_MIN_CHARS:
        return True
    if bullet_count >= BIG_TASK_MIN_BULLETS:
        return True
    if step_score >= BIG_TASK_MIN_STEP_MARKERS:
        return True

    return False


def _extract_user_intent_text(json_data: dict) -> str:
    """
    Best-effort extraction of "what the user asked" from OpenAI-like payloads.
    Works for chat.completions and falls back gracefully.
    """
    parts = []

    messages = json_data.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "system") and content:
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # OpenAI "content parts" shape
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                            parts.append(p["text"])

    prompt = json_data.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)

    return "\n".join(parts).strip()


def _should_route_to_anthropic(endpoint: str, json_data: dict) -> bool:
    # Only route chat-completions-like calls.
    if endpoint not in ("chat/completions",):
        return False
    if not ANTHROPIC_API_KEY:
        return False

    text = _extract_user_intent_text(json_data)
    if not text:
        return False

    # Code/debug should stay local by default.
    if _looks_like_code(text.lower()):
        return False

    return _is_big_task(text)


def _openai_messages_to_anthropic(json_data: dict) -> dict:
    messages_in = json_data.get("messages", [])
    system_chunks = []
    out_messages = []

    if isinstance(messages_in, list):
        for m in messages_in:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")

            # Normalize OpenAI content to text.
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                texts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        texts.append(p["text"])
                text = "\n".join(texts).strip()

            if not text:
                continue

            if role == "system":
                system_chunks.append(text)
                continue
            if role in ("user", "assistant"):
                out_messages.append(
                    {
                        "role": role,
                        "content": [{"type": "text", "text": text}],
                    }
                )

    system_text = "\n".join(system_chunks).strip() if system_chunks else None

    max_tokens = json_data.get("max_tokens")
    if not isinstance(max_tokens, int):
        # OpenAI "max_tokens" may be absent; set a sane default for Anthropic.
        max_tokens = 1024

    temperature = json_data.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        temperature = None

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": out_messages,
    }
    if system_text:
        payload["system"] = system_text
    if temperature is not None:
        payload["temperature"] = temperature

    # NOTE: Tools/function-calling is not mapped here yet.
    return payload


def _anthropic_to_openai_chat_completion(anthropic_json: dict) -> dict:
    # Extract assistant text.
    text_parts = []
    content = anthropic_json.get("content")
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                text_parts.append(p["text"])
    assistant_text = "".join(text_parts)

    now = int(time.time())
    resp = {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": f"anthropic/{ANTHROPIC_MODEL}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
    }

    usage = anthropic_json.get("usage")
    if isinstance(usage, dict):
        # Anthropic: {input_tokens, output_tokens}
        it = usage.get("input_tokens")
        ot = usage.get("output_tokens")
        if isinstance(it, int) and isinstance(ot, int):
            resp["usage"] = {"prompt_tokens": it, "completion_tokens": ot, "total_tokens": it + ot}

    return resp


def _filtered_forward_headers():
    # Drop hop-by-hop headers and headers that are likely to be wrong if we mutate the body.
    excluded = {"host", "content-length", "connection", "transfer-encoding"}
    return {k: v for k, v in request.headers if k.lower() not in excluded}


def _build_flask_response(upstream_resp: requests.Response):
    excluded_headers = {'content-encoding', 'content-length', 'transfer-encoding', 'connection'}
    proxy_headers = [(name, value) for (name, value) in upstream_resp.headers.items()
                    if name.lower() not in excluded_headers]

    def generate():
        for chunk in upstream_resp.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(generate(), status=upstream_resp.status_code, headers=proxy_headers)


@app.route("/healthz", methods=["GET"])
def healthz():
    # Basic readiness: can we reach LM Studio and is at least one model loaded?
    ids, err = get_lmstudio_model_ids(force_refresh=True)
    loaded, lerr = get_loaded_lmstudio_model_ids(force_refresh=True)
    status = 200 if ids and not err else 503
    return Response(
        json.dumps(
            {
                "ok": status == 200,
                "lmstudio_model": ids[0] if ids else None,
                "lmstudio_models": ids,
                "lmstudio_loaded_models": loaded,
                "lmstudio_loaded_endpoint_supported": _LMSTUDIO.loaded_endpoint_supported,
                "lmstudio_loaded_error": lerr,
                "lmstudio_error": err,
                "model_roles": MODEL_ROLES,
                "model_roles_source": MODEL_ROLES_SOURCE,
                "prefer_loaded_models": bool(PREFER_LOADED_MODELS),
                "strict_loaded_models": bool(STRICT_LOADED_MODELS),
                "default_model": DEFAULT_MODEL or None,
                "max_parallel": MAX_PARALLEL_MODEL_CALLS,
                "per_model_inflight_cap": LM_STUDIO_PER_MODEL_INFLIGHT_CAP,
                "on_model_miss": ON_MODEL_MISS,
                "anthropic_enabled": bool(ANTHROPIC_API_KEY),
                "anthropic_model": ANTHROPIC_MODEL,
                "litellm_available": _litellm_available(),
                "litellm_for_anthropic": bool(USE_LITELLM_FOR_ANTHROPIC),
                "litellm_prefix_routing": bool(ENABLE_LITELLM_PREFIX_ROUTING),
                "litellm_import_error": None if _litellm_available() else _litellm_import_error,
                "swarm_chat_enabled": bool(SWARM_CHAT_ENABLED),
                "swarm_chat_default_models": SWARM_CHAT_DEFAULT_MODELS,
                "swarm_chat_default_strategy": SWARM_CHAT_DEFAULT_STRATEGY,
                "swarm_chat_auto_tokens": sorted(_SWARM_AUTO_TOKENS),
                "swarm_chat_canonical": _SWARM_CHAT_CANONICAL,
                "swarm_chat_aliases": {
                    name: {"intent": intent, "deprecated": deprecated}
                    for name, (intent, deprecated) in sorted(_SWARM_CHAT_INTENTS.items())
                },
            }
        ),
        status=status,
        mimetype="application/json",
    )

@app.before_request
def _auth_guard():
    if not MIDDLE_LAYER_API_KEY:
        return None
    if _check_api_key(request.headers, MIDDLE_LAYER_API_KEY):
        return None
    return Response(
        json.dumps({"error": "Unauthorized"}),
        status=401,
        mimetype="application/json",
    )


@app.after_request
def _security_headers(response):
    _apply_security_headers(response, path=request.path or "")
    return response


@app.route('/v1/<path:endpoint>', methods=['POST', 'GET'])
def proxy(endpoint):
    """OpenAI-compatible front door: local-first, Opus for big tasks."""
    
    # For GET requests (like /models), forward directly without modification
    if request.method == 'GET':
        resp = requests.request(
            method=request.method,
            url=f'{LM_STUDIO_URL}/v1/{endpoint}',
            headers=_filtered_forward_headers(),
            data=None,
            cookies=request.cookies,
            allow_redirects=False,
            stream=True,
        )

        return _build_flask_response(resp)
    
    # For POST requests, decide local vs Anthropic (only for chat/completions).
    headers = _filtered_forward_headers()
    data = request.get_data()
    
    if request.is_json:
        try:
            json_data = json.loads(data)

            # If this is a big task and Anthropic is configured, route to Opus.
            if _should_route_to_anthropic(endpoint, json_data):
                if json_data.get("stream") is True:
                    return Response(
                        json.dumps(
                            {
                                "error": "Streaming via Anthropic routing is not enabled in middle_layer.py yet. Set stream=false or route locally."
                            }
                        ),
                        status=501,
                        mimetype="application/json",
                    )

                llm_resp, llm_err = _call_anthropic_chat(
                    json_data.get("messages") or [],
                    max_tokens=json_data.get("max_tokens"),
                    temperature=json_data.get("temperature"),
                    top_p=json_data.get("top_p"),
                    stop=json_data.get("stop"),
                    timeout=60,
                )
                if llm_err or not llm_resp:
                    return Response(
                        json.dumps({"error": f"Anthropic routing failed: {llm_err}"}),
                        status=502,
                        mimetype="application/json",
                    )

                openai_like = llm_resp
                # Help clients see which backend was used.
                resp_headers = {"X-Model-Routed-To": f"anthropic/{ANTHROPIC_MODEL}"}
                return Response(json.dumps(openai_like), status=200, mimetype="application/json", headers=resp_headers)

            requested = json_data.get("model")
            if (
                endpoint == "chat/completions"
                and ENABLE_LITELLM_PREFIX_ROUTING
                and isinstance(requested, str)
                and requested.lower().startswith("litellm/")
            ):
                if json_data.get("stream") is True:
                    return Response(
                        json.dumps(
                            {
                                "error": "Streaming via litellm/ routing is not enabled in middle_layer.py yet. Set stream=false."
                            }
                        ),
                        status=501,
                        mimetype="application/json",
                    )
                routed_model = requested.split("/", 1)[1].strip()
                llm_resp, llm_err = _call_litellm_chat(
                    json_data.get("messages") or [],
                    model_override=routed_model,
                    max_tokens=json_data.get("max_tokens"),
                    temperature=json_data.get("temperature"),
                    top_p=json_data.get("top_p"),
                    stop=json_data.get("stop"),
                    timeout=60,
                )
                if llm_err or not llm_resp:
                    return Response(
                        json.dumps({"error": f"LiteLLM routing failed: {llm_err}"}),
                        status=502,
                        mimetype="application/json",
                    )
                return Response(
                    json.dumps(llm_resp),
                    status=200,
                    mimetype="application/json",
                    headers={"X-Model-Routed-To": f"litellm/{routed_model}"},
                )

            swarm_intent, swarm_canonical = (None, None)
            if endpoint == "chat/completions" and SWARM_CHAT_ENABLED:
                swarm_intent, swarm_canonical = _swarm_chat_intent(requested)

            if swarm_intent is not None:
                # ``pipeline`` is intentionally a 400 (not 502) — it's a
                # client misuse, not an upstream failure.
                if swarm_intent == "pipeline":
                    body, err, _ = _run_swarm_chat_completion(
                        requested, json_data, intent="pipeline"
                    )
                    return Response(
                        json.dumps({
                            "error": err,
                            "redirect": "POST /swarm/pipeline",
                        }),
                        status=400,
                        mimetype="application/json",
                    )

                wants_stream = json_data.get("stream") is True
                swarm_resp, swarm_err, swarm_err_details = _run_swarm_chat_completion(
                    requested, json_data, intent=swarm_intent
                )
                if swarm_err or not swarm_resp:
                    body: dict = {"error": f"Swarm routing failed: {swarm_err}"}
                    if swarm_err_details:
                        # Structured per-agent breakdown so callers can dispatch
                        # on error_kind without parsing the prose summary.
                        body["error_details"] = swarm_err_details
                    headers: dict = {}
                    if swarm_canonical and swarm_canonical.lower() != requested.strip().lower():
                        # Help the client move off a deprecated alias even on
                        # the failure path.
                        headers["X-Swarm-Canonical-Name"] = swarm_canonical
                    if isinstance(swarm_err_details, dict):
                        kinds = swarm_err_details.get("kinds") or {}
                        if kinds:
                            headers["X-Swarm-Error-Kinds"] = ",".join(
                                f"{k}={v}" for k, v in sorted(kinds.items())
                            )
                    return Response(
                        json.dumps(body),
                        status=502,
                        mimetype="application/json",
                        headers=headers or None,
                    )
                if wants_stream:
                    return _swarm_body_to_sse_response(swarm_resp)
                resp_headers = {
                    "X-Model-Routed-To": str(swarm_resp.get("model", "swarm/unknown")),
                    "X-Swarm-Intent": swarm_intent,
                }
                if swarm_canonical and swarm_canonical.lower() != requested.strip().lower():
                    resp_headers["X-Swarm-Canonical-Name"] = swarm_canonical
                return Response(
                    json.dumps(swarm_resp),
                    status=200,
                    mimetype="application/json",
                    headers=resp_headers,
                )
            
            # Resolve the requested model against what is actually loaded.
            model_id, error = resolve_model_id(requested)
            fallback_from = None

            if error or not model_id:
                # If the caller asked for a specific model that is not loaded,
                # either fall back (default) or surface the error.
                if not _is_placeholder(requested) and ON_MODEL_MISS == "fallback":
                    # Under STRICT_LOADED_MODELS, fall back only to a loaded
                    # id so the proxy never silently asks LM Studio to JIT a
                    # different installed model.
                    if STRICT_LOADED_MODELS:
                        fb_ids, fb_err = get_loaded_lmstudio_model_ids()
                    else:
                        fb_ids, fb_err = get_lmstudio_model_ids()
                    if not fb_err and fb_ids:
                        model_id = fb_ids[0]
                        fallback_from = requested
                        error = None
                if error or not model_id:
                    return Response(
                        json.dumps({"error": f"503 Service Unavailable - {error}"}),
                        status=503,
                        mimetype='application/json'
                    )

            # Inject the resolved model ID (LM Studio needs an exact id).
            json_data['model'] = model_id

            # Forward to LM Studio with injected model
            resp = requests.request(
                method=request.method,
                url=f'{LM_STUDIO_URL}/v1/{endpoint}',
                headers=headers,
                data=json.dumps(json_data).encode('utf-8'),
                cookies=request.cookies,
                allow_redirects=False,
                stream=True,
                timeout=300,
            )

            flask_resp = _build_flask_response(resp)
            flask_resp.headers["X-Model-Routed-To"] = f"local/{model_id}"
            if fallback_from:
                flask_resp.headers["X-Model-Resolution"] = (
                    f"fallback (requested '{fallback_from}', not loaded)"
                )
            return flask_resp
        
        except Exception as e:
            # If JSON parsing fails, forward as-is
            resp = requests.request(
                method=request.method,
                url=f'{LM_STUDIO_URL}/v1/{endpoint}',
                headers=headers,
                data=data,
                cookies=request.cookies,
                allow_redirects=False,
                stream=True,
                timeout=300,
            )

            return _build_flask_response(resp)
    
    else:
        # Non-JSON request - forward as-is
        resp = requests.request(
            method=request.method,
            url=f'{LM_STUDIO_URL}/v1/{endpoint}',
            headers=headers,
            data=data,
            cookies=request.cookies,
            allow_redirects=False,
            stream=True,
            timeout=300,
        )

        return _build_flask_response(resp)


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
    """Call Anthropic /v1/messages with OpenAI-shaped messages and translate
    the response back into an OpenAI chat completion. Returns (resp, error)."""
    if USE_LITELLM_FOR_ANTHROPIC and _litellm_available():
        model_name = _litellm_model_for_anthropic(model_override or ANTHROPIC_MODEL)
        return _call_litellm_chat(messages, model_override=model_name, **kwargs)

    if not ANTHROPIC_API_KEY:
        return None, "ANTHROPIC_API_KEY not set"

    pseudo = {"messages": messages}
    if kwargs.get("max_tokens") is not None:
        pseudo["max_tokens"] = kwargs["max_tokens"]
    if kwargs.get("temperature") is not None:
        pseudo["temperature"] = kwargs["temperature"]

    payload = _openai_messages_to_anthropic(pseudo)
    if model_override:
        payload["model"] = model_override

    try:
        r = requests.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            headers={
                "content-type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            data=json.dumps(payload).encode("utf-8"),
            timeout=kwargs.get("timeout", SWARM_PER_CALL_TIMEOUT),
        )
        if r.status_code >= 400:
            return None, f"Anthropic {r.status_code}: {r.text[:300]}"
        return _anthropic_to_openai_chat_completion(r.json()), None
    except Exception as e:
        return None, f"Anthropic error: {e}"


# Pure swarm helpers re-exported here so historical call sites (and tests
# that monkey-patch ``mod._classify_swarm_error`` etc.) keep working without
# updating to the new module path. New code should import from
# ``middle_layer.swarm`` directly.
from middle_layer.swarm import (  # noqa: E402, F401
    _SWARM_ERROR_KINDS,
    _classify_swarm_error,
    _extract_text,
    _extract_upstream_status,
    _normalize_agent_spec,
    _strip_upstream_prefix,
)


from middle_layer.swarm import (  # noqa: E402, F401
    # Back-compat re-exports — see the block above.
    _SWARM_AUTO_TOKENS,
    _is_auto_swarm_token,
)
from middle_layer.swarm import expand_swarm_models as _swarm_expand_models  # noqa: E402
from middle_layer.swarm import SWARM_CHAT_AUTO_MAX as _SWARM_CHAT_AUTO_MAX  # noqa: E402


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


from middle_layer.swarm import (  # noqa: E402, F401
    # Back-compat re-exports — see the block above.
    _SWARM_CHAT_CANONICAL,
    _SWARM_CHAT_INTENTS,
    _is_swarm_chat_model,
    _summarize_failed_candidates,
    _swarm_alias_warned,
    _swarm_chat_intent,
)
from middle_layer import swarm as _swarm_runner  # noqa: E402

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


@app.route("/swarm/models", methods=["GET"])
def swarm_models():
    ids, err = get_lmstudio_model_ids(force_refresh=True)
    return Response(
        json.dumps(
            {
                "models": ids,
                "roles": MODEL_ROLES,
                "default_model": DEFAULT_MODEL or None,
                "max_parallel": MAX_PARALLEL_MODEL_CALLS,
                "anthropic_available": bool(ANTHROPIC_API_KEY),
                "anthropic_model": ANTHROPIC_MODEL if ANTHROPIC_API_KEY else None,
                "litellm_available": _litellm_available(),
                "litellm_for_anthropic": bool(USE_LITELLM_FOR_ANTHROPIC),
                "swarm_chat_auto_tokens": sorted(_SWARM_AUTO_TOKENS),
                "error": err,
            }
        ),
        status=200 if not err else 503,
        mimetype="application/json",
    )


@app.route("/swarm/fanout", methods=["POST"])
def swarm_fanout():
    """Broadcast one prompt to N models in parallel. Returns every response.

    Body:
      {
        "models":   ["role:coder", "qwen2.5-7b", "anthropic"],
        "messages": [...],         # OpenAI shape
        "max_tokens": 512,         # optional, applied to every agent
        "temperature": 0.7,        # optional
        "max_parallel": 3          # optional override (capped by env)
      }

    "models" also accepts the sentinel "auto" (or "loaded" / "*" / "all"),
    either as a bare string or as one entry in the list. Sentinels expand
    inline to every model currently loaded in LM Studio (de-duped, ordered).
    """
    data = request.get_json(silent=True) or {}
    models = data.get("models") or []
    messages = data.get("messages") or []
    if not isinstance(models, (list, str)) or not models:
        return Response(json.dumps({"error": "models (list, or 'auto') is required"}),
                        status=400, mimetype="application/json")
    if not isinstance(messages, list) or not messages:
        return Response(json.dumps({"error": "messages (list) is required"}),
                        status=400, mimetype="application/json")

    models, exp_err = _expand_swarm_models(models)
    if exp_err:
        return Response(json.dumps({"error": exp_err}),
                        status=503, mimetype="application/json")
    if not models:
        return Response(
            json.dumps({"error": "no LM Studio models loaded; load at least one or pass explicit models"}),
            status=503, mimetype="application/json",
        )

    common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p")}
    common = {k: v for k, v in common.items() if v is not None}

    results, err = _fanout(
        models, messages, common, max_parallel=data.get("max_parallel")
    )
    if err:
        return Response(json.dumps({"error": err}), status=503, mimetype="application/json")

    return Response(
        json.dumps(
            {
                "id": f"swarm_{uuid.uuid4().hex}",
                "object": "swarm.fanout",
                "created": int(time.time()),
                "responses": results,
            }
        ),
        status=200,
        mimetype="application/json",
        headers={"X-Swarm-Models": ",".join((r or {}).get("model", "?") for r in results)},
    )


@app.route("/swarm/vote", methods=["POST"])
def swarm_vote():
    """Fanout + consensus. Returns an OpenAI chat.completion with the winner.

    Body:
      {
        "models":      [...],
        "messages":    [...],
        "strategy":    "best-of-n" | "first-success" | "longest",
        "judge":       "role:reasoner",        # only for best-of-n
        "judge_system": "You are a strict judge. ..."   # optional override
      }
    """
    data = request.get_json(silent=True) or {}
    models = data.get("models") or []
    messages = data.get("messages") or []
    strategy = (data.get("strategy") or "best-of-n").lower()

    models_ok = isinstance(models, (list, str)) and models
    if not models_ok or not isinstance(messages, list) or not messages:
        return Response(json.dumps({"error": "models and messages are required"}),
                        status=400, mimetype="application/json")

    models, exp_err = _expand_swarm_models(models)
    if exp_err:
        return Response(json.dumps({"error": exp_err}),
                        status=503, mimetype="application/json")
    if not models:
        return Response(
            json.dumps({"error": "no LM Studio models loaded; load at least one or pass explicit models"}),
            status=503, mimetype="application/json",
        )

    common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p")
              if data.get(k) is not None}

    candidates, err = _fanout(models, messages, common)
    if err:
        return Response(json.dumps({"error": err}), status=503, mimetype="application/json")

    successes = [c for c in candidates if c["ok"] and c.get("text")]
    if not successes:
        errs = "; ".join(c.get("error") or "unknown" for c in candidates)
        return Response(
            json.dumps({"error": f"all agents failed: {errs}", "candidates": candidates}),
            status=502, mimetype="application/json",
        )

    rationale = ""
    if strategy == "first-success":
        winner = successes[0]
        rationale = "first agent to return a non-empty response"
    elif strategy == "longest":
        winner = max(successes, key=lambda c: len(c.get("text", "")))
        rationale = "longest non-empty response"
    else:  # best-of-n
        labels = [chr(ord("A") + i) for i in range(len(successes))]
        rendered = "\n\n".join(
            f"[{labels[i]}] (model={successes[i]['model']})\n{successes[i]['text']}"
            for i in range(len(successes))
        )
        original_user = _extract_user_intent_text({"messages": messages})
        judge_system = data.get("judge_system") or (
            "You are a strict judge. Below are candidate responses to a user request "
            "from different models, labeled [A], [B], etc. Pick the single best one. "
            "Reply with ONLY the letter on its own line, then a one-sentence reason."
        )
        judge_messages = [
            {"role": "system", "content": judge_system},
            {"role": "user", "content": (
                f"Original request:\n{original_user}\n\n"
                f"Candidate responses:\n{rendered}\n\n"
                "Pick the best one (A, B, ...) and explain briefly."
            )},
        ]
        judge_request = data.get("judge") or "role:reasoner"
        avail, _ = get_lmstudio_model_ids()
        judge_id, jerr = resolve_model_id(judge_request, avail)

        if jerr or not judge_id:
            winner = max(successes, key=lambda c: len(c.get("text", "")))
            rationale = f"judge unavailable ({jerr or 'no model'}); picked longest"
        else:
            # See chat-route judge above — same per-model semaphore policy.
            with _per_model_semaphore(judge_id):
                jresp, jerr = _lmstudio_chat_completion(
                    judge_id, judge_messages, max_tokens=200, temperature=0.0
                )
            verdict = _extract_text(jresp)
            picked_idx = None
            if verdict:
                for i, lab in enumerate(labels):
                    if re.search(rf"(?mi)^\s*{re.escape(lab)}\b", verdict):
                        picked_idx = i
                        break
            if picked_idx is None:
                winner = max(successes, key=lambda c: len(c.get("text", "")))
                rationale = (
                    f"judge response unparseable; fell back to longest. "
                    f"Verdict: {verdict[:140]}"
                )
            else:
                winner = successes[picked_idx]
                rationale = verdict.strip()

    out = {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"swarm/{winner['model']}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": winner["text"]},
            "finish_reason": "stop",
        }],
        "swarm": {
            "strategy": strategy,
            "winner": winner["model"],
            "rationale": rationale,
            "candidates": candidates,
        },
    }
    return Response(
        json.dumps(out),
        status=200,
        mimetype="application/json",
        headers={"X-Swarm-Strategy": strategy, "X-Swarm-Winner": str(winner["model"])},
    )


@app.route("/swarm/pipeline", methods=["POST"])
def swarm_pipeline():
    """Sequential chain of models. Each step sees previous outputs via templates.

    Body:
      {
        "messages": [...],          # original user/system messages
        "steps": [
          {"name": "plan",   "model": "role:reasoner",
           "system": "Plan the steps to answer the user.",
           "max_tokens": 512},
          {"name": "code",   "model": "role:coder",
           "system": "Implement the plan:\n{{plan}}",
           "max_tokens": 1024},
          {"name": "review", "model": "role:reasoner",
           "system": "Critique and fix this implementation:\n{{code}}"}
        ]
      }

    The final step's output is returned as an OpenAI chat.completion.
    """
    data = request.get_json(silent=True) or {}
    steps = data.get("steps") or []
    messages = data.get("messages") or []
    if not isinstance(steps, list) or not steps or not isinstance(messages, list) or not messages:
        return Response(json.dumps({"error": "steps and messages are required"}),
                        status=400, mimetype="application/json")

    available, err = get_lmstudio_model_ids()
    if err:
        return Response(json.dumps({"error": err}), status=503, mimetype="application/json")

    history = []
    last_text = ""

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        name = step.get("name") or f"step_{idx}"
        ctx = {h["name"]: h["text"] for h in history}
        ctx["previous"] = last_text

        def _fmt(template):
            if not isinstance(template, str):
                return template
            # Support both {{name}} (Mustache-ish) and {name} (str.format).
            t = re.sub(r"\{\{(\w+)\}\}", r"{\1}", template)
            try:
                return t.format(**ctx)
            except (KeyError, IndexError):
                return template

        sys_prompt = _fmt(step.get("system") or "")
        user_template = step.get("user")

        agent_messages = []
        if sys_prompt:
            agent_messages.append({"role": "system", "content": sys_prompt})
        if user_template:
            agent_messages.append({"role": "user", "content": _fmt(user_template)})
        else:
            agent_messages += [
                m for m in messages
                if isinstance(m, dict) and m.get("role") != "system"
            ]

        kwargs = {k: step[k] for k in ("max_tokens", "temperature", "top_p")
                  if step.get(k) is not None}

        model_id, resp, e, latency = _run_one_agent(
            {"model": step.get("model")}, agent_messages, kwargs, available
        )
        if e or not resp:
            return Response(
                json.dumps({"error": f"step '{name}' failed: {e}", "history": history}),
                status=502, mimetype="application/json",
            )

        text = _extract_text(resp)
        history.append({
            "name": name,
            "model": model_id,
            "text": text,
            "latency_ms": latency,
        })
        last_text = text

    final = history[-1] if history else {"text": "", "model": "?"}
    return Response(
        json.dumps({
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"swarm/pipeline/{final['model']}",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final.get("text", "")},
                "finish_reason": "stop",
            }],
            "swarm": {"strategy": "pipeline", "history": history},
        }),
        status=200,
        mimetype="application/json",
        headers={
            "X-Swarm-Strategy": "pipeline",
            "X-Swarm-Steps": ",".join(h["name"] for h in history),
        },
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
