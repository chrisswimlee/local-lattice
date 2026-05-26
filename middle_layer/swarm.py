"""Swarm primitives for the LM Studio + MLX gateways.

Three intent categories of OpenAI-shaped swarm meta-models go through this
module:

  council   best-of-n with a judge model. Aliases: ``swarmCouncil``,
            ``swarmVote``, ``swarm/vote``, ``swarmIntelligence``
            (the last is **deprecated** and emits ``DeprecationWarning``).
  fanout    no judge; first successful candidate wins. Alias: ``swarm/fanout``.
  pipeline  rejected from the chat shape because pipeline needs ``stages[]``
            which OpenAI chat can't carry. Alias: ``swarm/pipeline`` →
            HTTP 400 redirecting the caller to ``POST /swarm/pipeline``.

This module owns:

* swarm configuration constants read from env (``SWARM_*``,
  ``MAX_PARALLEL_MODEL_CALLS``, ``LM_STUDIO_PER_MODEL_INFLIGHT_CAP``,
  ``SWARM_STREAM_CHUNK_CHARS``);
* per-model serialization (a semaphore keyed by resolved model id —
  prevents LM Studio from crashing large MoE models under concurrent
  inference jobs);
* error classification into a small stable enum (``error_kind``) so callers
  like the openclaw embedded-agent runtime can fail soft on a known-bad
  agent without parsing prose;
* intent dispatch + alias map (with one-shot deprecation warnings);
* sentinel ``swarm.models`` expansion (``auto`` / ``loaded`` / ``*`` /
  ``all`` / ``all-loaded``);
* aggregate-failure summarization (the structured ``error_details`` body
  returned alongside HTTP 502).

Stateful IO (the actual fanout, agent-running, judge-calling) is **not**
in this module yet — it lives in the gateway modules so this stays
dependency-free besides ``threading`` / ``re`` / ``warnings``. Pass-3 work
will lift those into a ``SwarmDeps``-injected runner here so both gateways
share one implementation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Hard wall-clock budget for any single agent's chat call, in seconds.
SWARM_PER_CALL_TIMEOUT = int(os.environ.get("SWARM_PER_CALL_TIMEOUT", "180"))

#: Master switch. Set to 0/false to disable swarm meta-models on
#: ``/v1/chat/completions`` (callers that send ``model=swarmCouncil`` will
#: fall through to the regular resolver, which 404s the unknown id).
SWARM_CHAT_ENABLED = os.environ.get("SWARM_CHAT_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

#: Default ``swarm.models`` when the request body omits the field.
#:
#: Dynamic-by-default: ``"auto"`` expands to whatever LM Studio currently has
#: loaded, so a default-shaped swarm fanout never JIT-loads a giant model
#: behind the operator's back. The legacy default was
#: ``"role:reasoner,role:coder,role:fast"`` — three role lookups that, on a
#: machine with role prefs matching installed-but-not-loaded ids, would
#: trigger LM Studio to spin up new models per request. AGENTS.md
#: non-negotiable #1: keep the legacy value reachable via env var for one
#: minor version and warn on unset so operators can flip back.
_SWARM_CHAT_DEFAULT_MODELS_DEFAULT = "auto"
_SWARM_CHAT_DEFAULT_MODELS_LEGACY = "role:reasoner,role:coder,role:fast"
_SWARM_CHAT_DEFAULT_MODELS_ENV = os.environ.get("SWARM_CHAT_DEFAULT_MODELS")
if _SWARM_CHAT_DEFAULT_MODELS_ENV is None:
    warnings.warn(
        "SWARM_CHAT_DEFAULT_MODELS is unset: defaulting to "
        f"{_SWARM_CHAT_DEFAULT_MODELS_DEFAULT!r} (was "
        f"{_SWARM_CHAT_DEFAULT_MODELS_LEGACY!r}) so default-shaped swarm "
        "fanouts use the currently-loaded set instead of three role lookups "
        "that may JIT-load installed-but-not-loaded models. Set "
        f"SWARM_CHAT_DEFAULT_MODELS={_SWARM_CHAT_DEFAULT_MODELS_LEGACY!r} "
        "to keep the legacy behavior; will be removed in 0.2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    _SWARM_CHAT_DEFAULT_MODELS_RAW = _SWARM_CHAT_DEFAULT_MODELS_DEFAULT
else:
    _SWARM_CHAT_DEFAULT_MODELS_RAW = _SWARM_CHAT_DEFAULT_MODELS_ENV
SWARM_CHAT_DEFAULT_MODELS = [
    m.strip()
    for m in _SWARM_CHAT_DEFAULT_MODELS_RAW.split(",")
    if m.strip()
]

#: Default winner-pick strategy for ``swarmCouncil`` / ``swarmVote`` etc.
#: ``swarm/fanout`` always defaults to ``"fanout"`` regardless of this value.
#:
#: ``best-of-n`` pairs with the curated ``"auto"`` model expansion
#: (loaded + chat-capable, capped at ``SWARM_CHAT_AUTO_MAX``) to give a
#: genuine swarm: a small diverse set of models, judged for consensus.
#: For latency-over-quality, callers can pass ``strategy: "first-success"``
#: per-request (which now actually returns on first success and cancels
#: in-flight peers, see :func:`fanout`).
SWARM_CHAT_DEFAULT_STRATEGY = os.environ.get(
    "SWARM_CHAT_DEFAULT_STRATEGY", "best-of-n"
).strip().lower()

#: Cap on how many loaded chat-capable ids the ``auto`` /``loaded`` / ``*``
#: sentinels expand to. Without a cap, a box with 17 loaded ids would fan
#: every default-shaped swarm out to all 17 — most of the latency wasted
#: on slow models the judge will never pick. Three is the sweet spot for
#: diversity-vs-cost: one reasoner + one coder + one fast is enough to get
#: a meaningful best-of-n vote without quadrupling p50 latency. Set to 0
#: to disable the cap (legacy behavior).
SWARM_CHAT_AUTO_MAX = max(0, int(os.environ.get("SWARM_CHAT_AUTO_MAX", "3")))

#: Default judge model for ``best-of-n`` strategy when the request body
#: doesn't override ``swarm.judge``.
SWARM_CHAT_DEFAULT_JUDGE = os.environ.get(
    "SWARM_CHAT_DEFAULT_JUDGE", "role:reasoner"
).strip()

#: Chunk size (in characters) for the synthetic SSE we emit when a
#: streaming client requests a swarm meta-model. The swarm itself is
#: inherently batch — we run it normally and slice the winner's text back as
#: OpenAI ``chat.completion.chunk`` frames so streaming-only clients don't
#: see 501.
SWARM_STREAM_CHUNK_CHARS = max(1, int(os.environ.get("SWARM_STREAM_CHUNK_CHARS", "64")))

#: Cap on the ThreadPoolExecutor that drives ``_fanout``. Tune down for
#: memory-tight Macs.
MAX_PARALLEL_MODEL_CALLS = int(os.environ.get("MAX_PARALLEL_MODEL_CALLS", "2"))

#: Maximum simultaneous LM Studio chat-completion requests targeting the
#: *same* resolved model id. LM Studio reliably crashes large-context MoE
#: models when fed two concurrent inference jobs (observed with
#: ``qwen3.5-122b-a10b`` at 128k ctx → "The model has crashed without
#: additional information"). Default is 1 so swarm fanouts whose specs all
#: resolve to the same loaded model are serialized into back-to-back calls
#: instead of crashing the runtime. Set higher (e.g. 2) when you know your
#: loaded models tolerate it.
LM_STUDIO_PER_MODEL_INFLIGHT_CAP = max(
    1, int(os.environ.get("LM_STUDIO_PER_MODEL_INFLIGHT_CAP", "1"))
)


# ---------------------------------------------------------------------------
# Per-model serialization
# ---------------------------------------------------------------------------

_per_model_semaphores: dict[str, threading.Semaphore] = {}
_per_model_semaphores_lock = threading.Lock()


def per_model_semaphore(model_id: str) -> threading.Semaphore:
    """Lazily allocate (and reuse) a per-model semaphore so concurrent swarm
    agents that resolve to the same id are serialized through the gateway.
    Public name is canonical; ``_per_model_semaphore`` is kept as a private
    alias for back-compat with the in-tree call sites.
    """
    with _per_model_semaphores_lock:
        sem = _per_model_semaphores.get(model_id)
        if sem is None:
            sem = threading.Semaphore(LM_STUDIO_PER_MODEL_INFLIGHT_CAP)
            _per_model_semaphores[model_id] = sem
        return sem


# Back-compat alias — historical call sites use the underscore-prefixed name.
_per_model_semaphore = per_model_semaphore


# ---------------------------------------------------------------------------
# ``swarm.models`` sentinel expansion
# ---------------------------------------------------------------------------
#
# Sentinel tokens that expand a swarm ``models`` list to whatever the gateway
# currently has loaded. Recognized in both ``swarm.models`` (per-request)
# and the ``SWARM_CHAT_DEFAULT_MODELS`` env var. Order-preserving and de-duped.
_SWARM_AUTO_TOKENS = frozenset({"auto", "loaded", "*", "all", "all-loaded"})


def is_auto_swarm_token(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in _SWARM_AUTO_TOKENS


_is_auto_swarm_token = is_auto_swarm_token  # back-compat alias


def expand_swarm_models(
    spec,
    available=None,
    *,
    fetch_loaded=None,
    max_auto_entries: int | None = None,
):
    """Expand sentinel tokens in a swarm models list to loaded model ids.

    Recognized tokens (case-insensitive): ``auto``, ``loaded``, ``*``,
    ``all``, ``all-loaded``. Each token is replaced inline with every model
    id currently loaded on the gateway, preserving order. Non-sentinel
    entries (exact ids, ``role:...``, ``*substr*``, ``anthropic[:model]``,
    etc.) pass through unchanged. Duplicate string entries are dropped to
    keep the fanout small.

    ``available`` may be passed eagerly when the caller already has the list
    in hand. If a sentinel is present and ``available`` is None, ``fetch_loaded``
    (a zero-arg callable returning ``(ids, error)``) is invoked. This keeps the
    module pure — the real gateway-specific probe lives in the gateway. The
    caller is expected to pass a probe that already returns only chat-capable
    ids when ``auto`` is meant to drive ``/v1/chat/completions`` fanouts.

    ``max_auto_entries`` caps how many ids each sentinel token contributes,
    so a default-shaped swarm against a box with N loaded chat models doesn't
    explode into an N-way fanout. ``None`` / ``0`` disables the cap. The cap
    is applied per-sentinel and only to ids contributed by the sentinel —
    explicit ids the caller listed alongside the sentinel are always kept.

    Returns ``(expanded_list, error_or_None)``. If a sentinel is present
    but the gateway reports zero loaded models, the sentinel contributes
    nothing; the caller should error if the resulting list is empty.
    """
    if isinstance(spec, str):
        items = [spec]
    elif isinstance(spec, list):
        items = list(spec)
    else:
        return None, "swarm.models must be a list or sentinel string"

    needs_loaded = any(is_auto_swarm_token(s) for s in items)
    loaded = available
    if needs_loaded and loaded is None:
        if fetch_loaded is None:
            return None, "swarm.models contains a sentinel but no loaded-list source was provided"
        loaded, err = fetch_loaded()
        if err:
            return None, err

    cap = max_auto_entries if (max_auto_entries and max_auto_entries > 0) else None

    out: list = []
    seen: set = set()
    for entry in items:
        if is_auto_swarm_token(entry):
            taken = 0
            for mid in (loaded or []):
                if not isinstance(mid, str) or mid in seen:
                    continue
                if cap is not None and taken >= cap:
                    break
                seen.add(mid)
                out.append(mid)
                taken += 1
            continue
        if isinstance(entry, str):
            key = entry.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        else:
            out.append(entry)
    return out, None


_expand_swarm_models = expand_swarm_models  # back-compat alias


# ---------------------------------------------------------------------------
# Structured swarm error classification
# ---------------------------------------------------------------------------
#
# Every swarm candidate that fails carries an ``error_kind`` from the small
# set below so callers (notably the openclaw embedded-agent runtime) can fail
# soft on a known-bad agent without parsing prose. Add new kinds rarely and
# document them here — clients pin against this enum.
#
#   no_models_loaded     LM Studio is reachable but has zero models loaded.
#   model_not_resolved   The agent's spec (role:..., id, glob) didn't match
#                        anything loaded *or* installed.
#   oom                  LM Studio refused to JIT-load due to insufficient RAM.
#   model_crashed        LM Studio reported the loaded model crashed mid-call.
#   empty_response       LM Studio returned 200 with an empty assistant
#                        ``content`` (typical with reasoning models when the
#                        whole token budget is consumed by reasoning_content).
#                        The raw response is still on the candidate so callers
#                        can fall back to ``reasoning_content`` if they want.
#   timeout              Our timeout (``SWARM_PER_CALL_TIMEOUT``) tripped.
#   connection_error     We couldn't reach LM Studio at all.
#   upstream_4xx         Other 4xx from LM Studio / Anthropic.
#   upstream_5xx         5xx from LM Studio / Anthropic (HTML error pages,
#                        Internal Server Error, etc.).
#   config_error         Local misconfiguration (no ``ANTHROPIC_API_KEY``,
#                        LiteLLM not installed, etc.). Caller's deployment
#                        problem.
#   anthropic_error      Anthropic adapter raised a non-HTTP exception.
#   litellm_error        LiteLLM adapter raised a non-HTTP exception.
#   unknown              Anything we couldn't classify.
SWARM_ERROR_KINDS = frozenset({
    "no_models_loaded", "model_not_resolved",
    "oom", "model_crashed", "empty_response",
    "timeout", "connection_error",
    "upstream_4xx", "upstream_5xx",
    "config_error", "anthropic_error", "litellm_error",
    "unknown",
})

_SWARM_ERROR_KINDS = SWARM_ERROR_KINDS  # back-compat alias

_OOM_PHRASES = (
    "insufficient system resources",
    "would likely overload your system",
    "model loading was stopped",
)
_CRASH_PHRASES = (
    "the model has crashed",
    "model has crashed without additional information",
)


def extract_upstream_status(error_str) -> int | None:
    """Pull the upstream HTTP status (LM Studio / Anthropic) out of an error
    string we generated upstream-side. Returns None if no status is encoded.
    """
    if not isinstance(error_str, str):
        return None
    m = re.match(r"^(?:LM Studio|Anthropic)\s+(\d{3})\s*:", error_str)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m2 = re.search(r"endpoint returned (\d{3})", error_str)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return None
    return None


_extract_upstream_status = extract_upstream_status  # back-compat alias


def classify_swarm_error(error_str, http_status: int | None = None) -> str:
    """Map a per-agent error string (with optional upstream HTTP status) to a
    stable ``error_kind`` from ``SWARM_ERROR_KINDS``. The mapping is
    deliberately string-based because every adapter (LM Studio, Anthropic,
    LiteLLM) embeds its own prose; we never want to break the caller contract
    when an upstream tweaks its message wording, so we match on phrase
    fragments and fall back to ``"unknown"`` for safety.
    """
    if not error_str:
        return "unknown"
    s = str(error_str).strip()
    sl = s.lower()

    if http_status is None:
        http_status = extract_upstream_status(s)

    # Connection / transport before HTTP status — a 4xx from our own retry of
    # a connection error wouldn't make sense here.
    if "cannot connect to lm studio" in sl or "connection refused" in sl:
        return "connection_error"
    if sl.startswith("timeout") or "timeout " in sl or "timed out" in sl:
        return "timeout"

    # Resolver-level failures (no loaded model / no match).
    if "no model is loaded" in sl:
        return "no_models_loaded"
    if "no loaded lm studio model matched" in sl or "did not match" in sl:
        return "model_not_resolved"
    if "swarm.models expanded to an empty set" in sl:
        return "no_models_loaded"

    # Configuration problems we can't fix at runtime.
    if "anthropic_api_key not set" in sl or "litellm not available" in sl:
        return "config_error"

    # OOM and model-crash are 4xx from LM Studio with specific phrases. Match
    # the phrases first so we don't lose specificity to the generic "upstream_4xx".
    if any(p in sl for p in _OOM_PHRASES):
        return "oom"
    if any(p in sl for p in _CRASH_PHRASES):
        return "model_crashed"

    # Generic upstream HTTP buckets.
    if isinstance(http_status, int):
        if 400 <= http_status < 500:
            return "upstream_4xx"
        if 500 <= http_status < 600:
            return "upstream_5xx"

    if "anthropic error:" in sl:
        return "anthropic_error"
    if "litellm error:" in sl:
        return "litellm_error"

    return "unknown"


_classify_swarm_error = classify_swarm_error  # back-compat alias


def strip_upstream_prefix(error_str) -> str:
    """Trim the ``LM Studio NNN: `` / ``Anthropic NNN: `` prefix so the
    structured ``error_detail`` field carries the upstream payload only.
    Falls back to the original string when no prefix is present.
    """
    if not isinstance(error_str, str):
        return ""
    m = re.match(r"^(?:LM Studio|Anthropic)\s+\d{3}\s*:\s*(.*)$", error_str, re.DOTALL)
    return (m.group(1) if m else error_str).strip()


_strip_upstream_prefix = strip_upstream_prefix  # back-compat alias


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def normalize_agent_spec(spec):
    """Accept a string or dict and return a normalized agent dict."""
    if isinstance(spec, str):
        return {"model": spec}
    if isinstance(spec, dict):
        return dict(spec)
    return {"model": str(spec)}


_normalize_agent_spec = normalize_agent_spec  # back-compat alias


def extract_text(openai_response) -> str:
    """Pull the assistant text out of an OpenAI chat.completion response."""
    if not isinstance(openai_response, dict):
        return ""
    choices = openai_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(msg, dict):
        return (msg.get("content") or "").strip()
    return ""


_extract_text = extract_text  # back-compat alias


def spec_to_agent_id(spec) -> str:
    """Stable, caller-visible label for a swarm agent. Preserves the original
    request shape (``role:reasoner``, ``anthropic:claude-...``, raw id, glob)
    so the caller can correlate per-candidate results back to the slot it
    asked for, even if resolution lands on a fallback model.
    """
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        m = spec.get("model")
        return str(m) if m is not None else "?"
    return str(spec)


# ---------------------------------------------------------------------------
# Swarm meta-model name → intent map
# ---------------------------------------------------------------------------
#
# Three legitimate intents reach ``/v1/chat/completions`` via the ``model:``
# field; everything else has a richer shape and lives on a dedicated route
# (``POST /swarm/fanout`` / ``/swarm/vote`` / ``/swarm/pipeline``).
#
#   council   best-of-n with a judge model. Default for every alias that
#             historically meant "vote / pick the winner".
#   fanout    no judge; return the first successful candidate. Useful when
#             the caller wants speed over consensus.
#   pipeline  not implementable as a chat meta-model — pipeline needs
#             ``stages[]`` which the OpenAI chat shape cannot carry. We keep
#             the alias so old clients get a *helpful* 400 redirecting them
#             to ``POST /swarm/pipeline`` instead of a silent fallthrough.
#
# Canonical names are listed first; alias entries marked ``deprecated`` emit
# one ``DeprecationWarning`` per process the first time they're seen, per the
# AGENTS.md non-negotiable on deprecation paths.
SWARM_CHAT_INTENTS: dict[str, tuple[str, bool]] = {
    # name (lowercased) -> (intent, deprecated)
    "swarmcouncil":        ("council", False),
    "swarmvote":           ("council", False),
    "swarm/vote":          ("council", False),
    "swarmintelligence":   ("council", True),   # openclaw runtime alias; will be removed in 0.2.0
    "swarm/fanout":        ("fanout", False),
    "swarm/pipeline":      ("pipeline", False),
}
SWARM_CHAT_CANONICAL = "swarmCouncil"

_SWARM_CHAT_INTENTS = SWARM_CHAT_INTENTS  # back-compat alias
_SWARM_CHAT_CANONICAL = SWARM_CHAT_CANONICAL  # back-compat alias

# Tracks which deprecated aliases have already produced a warning this
# process. Cleared by tests via ``_swarm_alias_warned.clear()``.
_swarm_alias_warned: set[str] = set()


def swarm_chat_intent(requested_model) -> tuple[str, str] | tuple[None, None]:
    """Return ``(intent, canonical_name)`` for a swarm meta-model, or
    ``(None, None)`` for a regular model id. ``intent`` is one of
    ``"council"``, ``"fanout"``, ``"pipeline"``. Emits a one-shot
    ``DeprecationWarning`` for deprecated aliases so old clients see the new
    name without breaking. Comparison is case-insensitive but otherwise
    exact — no substring matching, so a model literally named
    ``my-swarm-vote`` won't be intercepted.
    """
    if not isinstance(requested_model, str):
        return None, None
    name = requested_model.strip().lower()
    entry = SWARM_CHAT_INTENTS.get(name)
    if entry is None:
        return None, None
    intent, deprecated = entry
    if deprecated and name not in _swarm_alias_warned:
        _swarm_alias_warned.add(name)
        warnings.warn(
            f"swarm meta-model {requested_model!r} is a deprecated alias for "
            f"{SWARM_CHAT_CANONICAL!r}; will be removed in 0.2.0. "
            f"Update your client to send model={SWARM_CHAT_CANONICAL!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
    canonical = SWARM_CHAT_CANONICAL if intent == "council" else f"swarm/{intent}"
    return intent, canonical


_swarm_chat_intent = swarm_chat_intent  # back-compat alias


def is_swarm_chat_model(requested_model) -> bool:
    """Back-compat predicate. Prefer ``swarm_chat_intent`` for new code."""
    intent, _ = swarm_chat_intent(requested_model)
    return intent is not None


_is_swarm_chat_model = is_swarm_chat_model  # back-compat alias


# ---------------------------------------------------------------------------
# Aggregate failure summarization
# ---------------------------------------------------------------------------


def summarize_failed_candidates(candidates) -> dict:
    """Build the structured ``error_details`` body returned alongside the
    legacy prose error when every swarm agent fails. Shape is documented in
    ``docs/capabilities.md``; openclaw / other callers should treat
    ``error_kind`` values as a stable enum (see ``SWARM_ERROR_KINDS``).
    """
    agents: list[dict] = []
    kinds: dict[str, int] = {}
    statuses: dict[int, int] = {}
    for c in candidates or []:
        kind = c.get("error_kind") or "unknown"
        kinds[kind] = kinds.get(kind, 0) + 1
        st = c.get("http_status")
        if isinstance(st, int):
            statuses[st] = statuses.get(st, 0) + 1
        agents.append({
            "agent_id": c.get("agent_id"),
            "model": c.get("model"),
            "ok": bool(c.get("ok")),
            "error_kind": kind,
            "http_status": c.get("http_status"),
            "error_detail": c.get("error_detail") or c.get("error"),
            "latency_ms": c.get("latency_ms"),
        })
    # Choose the summary line based on whether the cohort is uniformly an
    # empty_response problem (caller likely needs to raise max_tokens) vs a
    # mixed failure mode (caller probably needs to retry / reload models).
    if kinds and set(kinds.keys()) == {"empty_response"}:
        summary = (
            "all swarm agents returned empty content "
            "(likely max_tokens too low for reasoning models; check "
            "reasoning_content on each candidate.response)"
        )
    else:
        summary = "all swarm agents failed"
    # Surface ALL distinct kinds so callers can decide between fail-fast and
    # retry-once policies without parsing the prose summary.
    return {
        "summary": summary,
        "agent_count": len(agents),
        "kinds": kinds,
        "upstream_statuses": statuses,
        "agents": agents,
    }


_summarize_failed_candidates = summarize_failed_candidates  # back-compat alias


# ---------------------------------------------------------------------------
# Gateway-agnostic IO runners
# ---------------------------------------------------------------------------
#
# The functions below run actual chat completions through whichever backend
# the gateway has wired up. They take a ``SwarmDeps`` so this module stays
# agnostic to LM Studio vs MLX vs anything else — middle_layer.py and
# middle_layerMLX.py both build their own SwarmDeps and call the same
# runners.


@dataclass(frozen=True)
class SwarmDeps:
    """Gateway-supplied callables and config the swarm runners need.

    Every callable matches the existing in-tree signature so we can wire
    middle_layer.py and middle_layerMLX.py to this module without changing
    their internal helpers.

    Attributes:
        chat_completion: ``(model_id, messages, **kwargs) -> (resp, err)``.
            For LM Studio this is the HTTP POST to ``/v1/chat/completions``;
            for MLX it's the in-process generate call.
        anthropic_chat: ``(messages, model_override=None, **kwargs) ->
            (resp, err)`` or ``None`` if Anthropic isn't wired up.
        resolve_model_id: ``(requested, available, loaded=None) -> (id, err)``.
        get_available_models: ``() -> (ids, err)``. The full installed set.
        get_loaded_models: ``() -> (ids, err)`` or ``None``. When non-None
            and ``prefer_loaded_models`` is True, ``run_swarm_chat_completion``
            primes ``run_one_agent`` with the loaded snapshot so the resolver
            picks already-loaded ids over JIT-loading.
        extract_user_intent: ``(json_data: dict) -> str``. Used to render the
            judge prompt.
        anthropic_default_model: Label for ``anthropic`` agents that don't
            override the model.
        prefer_loaded_models: Honors the gateway's PREFER_LOADED_MODELS knob.
    """

    chat_completion: Callable[..., tuple[Any, Optional[str]]]
    resolve_model_id: Callable[..., tuple[Any, Optional[str]]]
    get_available_models: Callable[[], tuple[list, Optional[str]]]
    extract_user_intent: Callable[[dict], str]
    anthropic_default_model: str
    anthropic_chat: Optional[Callable[..., tuple[Any, Optional[str]]]] = None
    get_loaded_models: Optional[Callable[[], tuple[list, Optional[str]]]] = None
    prefer_loaded_models: bool = True


def run_one_agent(spec, default_messages, default_kwargs, available, deps, loaded=None):
    """Run one agent. Returns ``(resolved_model_id_or_label, response, error,
    latency_ms)``. Same-model agents serialize through the per-model semaphore
    so a 3-way fanout that all resolves to one loaded id doesn't fire 2-3
    concurrent inference jobs at the upstream (which crashes large MoE
    models at high context).
    """
    spec = normalize_agent_spec(spec)
    requested = spec.get("model")
    requested_str = (requested or "").strip()

    msgs = spec.get("messages") or list(default_messages)
    sys_prompt = spec.get("system")
    if sys_prompt:
        msgs = [{"role": "system", "content": sys_prompt}] + [
            m for m in msgs if isinstance(m, dict) and m.get("role") != "system"
        ]

    kwargs = dict(default_kwargs)
    for k in ("max_tokens", "temperature", "top_p", "timeout"):
        if k in spec and spec[k] is not None:
            kwargs[k] = spec[k]

    # Anthropic participant.
    if requested_str.lower().startswith("anthropic"):
        if deps.anthropic_chat is None:
            return (
                requested_str or "?",
                None,
                "Anthropic adapter not wired into this gateway",
                0,
            )
        override = None
        if ":" in requested_str:
            override = requested_str.split(":", 1)[1].strip() or None
        label = f"anthropic/{override or deps.anthropic_default_model}"
        t0 = time.time()
        resp, err = deps.anthropic_chat(msgs, model_override=override, **kwargs)
        return label, resp, err, int((time.time() - t0) * 1000)

    # LM Studio (or MLX) participant.
    model_id, err = deps.resolve_model_id(requested, available, loaded=loaded)
    if err:
        return requested or "?", None, err, 0
    sem = per_model_semaphore(model_id)
    t0 = time.time()
    with sem:
        resp, err = deps.chat_completion(model_id, msgs, **kwargs)
    return model_id, resp, err, int((time.time() - t0) * 1000)


def fanout(
    specs,
    messages,
    common_kwargs,
    deps,
    max_parallel=None,
    *,
    early_exit_on_first_success: bool = False,
):
    """Run each spec in parallel (bounded). Returns ``(results_list, error)``.

    Result rows carry the structured-error fields (``agent_id``,
    ``error_kind``, ``http_status``, ``error_detail``) so the all-failed
    branch in :func:`run_swarm_chat_completion` can build the structured
    ``error_details`` body without a second pass.

    When ``early_exit_on_first_success=True``, the fanout stops scheduling
    new agents and attempts to cancel any not-yet-started futures the
    moment one agent returns ``ok=True`` with non-empty text. Already-
    running agents continue (Python ThreadPoolExecutor can't preempt them),
    but their results are discarded and the returned list reflects only
    the agents that had completed by then plus the winning one. This is
    the real "fastest wins" semantic — picking ``successes[0]`` in
    input order from a wait-for-all result is *not* the same thing.
    """
    if not specs:
        return None, "swarm requires at least one model"

    available, err = deps.get_available_models()
    if err:
        # Anthropic-only swarms can still proceed without LM Studio reachable.
        all_anthropic = all(
            isinstance(s, str) and s.lower().startswith("anthropic")
            or (isinstance(s, dict) and str(s.get("model", "")).lower().startswith("anthropic"))
            for s in specs
        )
        if not all_anthropic:
            return None, err
        available = []

    # Probe loaded ids once for the whole fanout so every agent sees the same
    # snapshot and we don't ask the upstream N times in parallel.
    loaded = None
    if deps.prefer_loaded_models and available and deps.get_loaded_models is not None:
        loaded, _lerr = deps.get_loaded_models()

    cap = MAX_PARALLEL_MODEL_CALLS
    if isinstance(max_parallel, int) and max_parallel > 0:
        cap = min(cap, max_parallel)
    results: list = [None] * len(specs)
    workers = max(1, min(cap, len(specs)))

    # We deliberately avoid the ``with ThreadPoolExecutor(...)`` context
    # manager here because its ``__exit__`` calls ``shutdown(wait=True)``,
    # which would block early-exit returns on any in-flight peer until
    # ``SWARM_PER_CALL_TIMEOUT`` (180s default) fires — defeating the
    # whole point of early-exit. Instead we manage shutdown explicitly so
    # the early-exit branch can ``shutdown(wait=False, cancel_futures=True)``
    # and return immediately. Orphaned threads finish in the background
    # bounded by the per-call timeout.
    pool = ThreadPoolExecutor(max_workers=workers)
    early_exited = False
    try:
        futs = {
            pool.submit(run_one_agent, spec, messages, common_kwargs, available, deps, loaded): i
            for i, spec in enumerate(specs)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                model_id, resp, e, latency = fut.result()
            except Exception as e:  # noqa: BLE001
                model_id, resp, latency = "?", None, 0
                e = str(e)
            text = extract_text(resp) if resp else ""
            api_ok = e is None and resp is not None
            error_kind = None
            http_status = None
            error_detail = None
            if not api_ok and e:
                http_status = extract_upstream_status(e)
                error_kind = classify_swarm_error(e, http_status=http_status)
                error_detail = strip_upstream_prefix(e)
            elif api_ok and not text:
                # 200 OK but the assistant ``content`` is empty. Common with
                # reasoning models when ``max_tokens`` is consumed entirely
                # by ``reasoning_content`` (we keep ``response`` on the
                # candidate so callers can recover the chain-of-thought).
                error_kind = "empty_response"
                error_detail = (
                    "upstream returned 200 with empty assistant content "
                    "(check reasoning_content / increase max_tokens)"
                )
                e = "empty assistant content"
            # Swarm logic treats "no usable text" as a fail (it can't vote
            # on nothing), so collapse api_ok + empty text into ok=False.
            ok = api_ok and bool(text)
            results[i] = {
                "agent_id": spec_to_agent_id(specs[i]),
                "model": model_id,
                "ok": ok,
                "error": e,
                "error_kind": error_kind,
                "http_status": http_status,
                "error_detail": error_detail,
                "latency_ms": latency,
                "response": resp,
                "text": text,
            }
            if early_exit_on_first_success and ok:
                early_exited = True
                break
    finally:
        # ``cancel_futures=True`` (3.9+) prevents not-yet-started peers
        # from running. ``wait=False`` on the early-exit branch returns
        # control to the caller without joining still-running peers;
        # those threads finish in the background, bounded by the
        # per-call timeout, and their results are simply discarded.
        pool.shutdown(wait=not early_exited, cancel_futures=True)
    return results, None


def run_swarm_chat_completion(requested_model, json_data, deps, intent: str = "council"):
    """Execute swarm logic and return ``(body, err_str, err_details)``.

    See ``docs/capabilities.md`` for the full request/response shape. Intent
    semantics:

      ``council``   → ``SWARM_CHAT_DEFAULT_STRATEGY`` (best-of-n with judge).
      ``fanout``    → ``"fanout"`` strategy by default (no judge ceremony;
                      first successful candidate wins).
      ``pipeline``  → reject with a 400 redirecting to ``POST /swarm/pipeline``
                      because the OpenAI chat shape can't carry ``stages[]``.

    ``err_details`` is non-None only on the *all-agents-failed* branch; the
    HTTP route handler surfaces it as ``error_details`` in the JSON 502 body
    so clients can dispatch on ``error_kind`` instead of parsing the prose
    summary. Pre-fanout validation failures still return the legacy
    ``(None, err_str, None)`` shape — there's no per-agent breakdown yet.
    """
    if intent == "pipeline":
        return None, (
            "swarm/pipeline cannot run on /v1/chat/completions because the "
            "OpenAI chat shape cannot carry 'stages[]'. Send your request to "
            "POST /swarm/pipeline (with {stages: [{model, prompt_prefix}, ...], "
            "input}) instead."
        ), None

    messages = json_data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None, "messages (list) is required for swarm chat", None

    common = {k: json_data.get(k) for k in ("max_tokens", "temperature", "top_p")}
    common = {k: v for k, v in common.items() if v is not None}

    swarm_cfg = json_data.get("swarm") if isinstance(json_data.get("swarm"), dict) else {}
    models = swarm_cfg.get("models") or SWARM_CHAT_DEFAULT_MODELS
    if swarm_cfg.get("strategy"):
        strategy = swarm_cfg["strategy"].lower()
    elif intent == "fanout":
        strategy = "fanout"
    else:
        strategy = SWARM_CHAT_DEFAULT_STRATEGY.lower()
    max_parallel = swarm_cfg.get("max_parallel")

    if not isinstance(models, (list, str)) or not models:
        return None, "swarm.models must be a non-empty list (or 'auto')", None

    # ``auto`` should expand against the truly-loaded, chat-capable set
    # (not the installed set), so a default-shaped swarm never adds an
    # embedding model or an installed-but-not-loaded id to the fanout. We
    # fall back to ``get_available_models`` when the gateway didn't wire a
    # loaded probe — in that case the strict resolver will still reject
    # not-loaded ids per-agent.
    def _fetch_auto_pool():
        if deps.get_loaded_models is not None:
            ids, err = deps.get_loaded_models()
            if err:
                return ids, err
            if ids:
                return ids, None
        return deps.get_available_models()

    models, exp_err = expand_swarm_models(
        models,
        fetch_loaded=_fetch_auto_pool,
        max_auto_entries=SWARM_CHAT_AUTO_MAX,
    )
    if exp_err:
        return None, exp_err, None
    if not models:
        return None, (
            "swarm.models expanded to an empty set: no models are loaded in "
            "LM Studio. Load at least one model (or pass an explicit swarm.models)."
        ), None

    # Strategies that pick the temporally-first successful candidate don't
    # need every agent's output, so let fanout cancel pending peers as soon
    # as one succeeds. ``best-of-n`` and ``longest`` still wait for all so
    # the judge / max-by-length actually has multiple candidates to compare.
    early_exit = strategy in ("first-success", "first_success", "fanout")

    candidates, err = fanout(
        models,
        messages,
        common,
        deps,
        max_parallel=max_parallel,
        early_exit_on_first_success=early_exit,
    )
    if err:
        return None, err, None

    # When ``fanout`` exits early, only completed agents are in the list
    # (the rest are ``None`` placeholders). Drop those so downstream
    # filtering doesn't trip on them.
    candidates = [c for c in candidates if c is not None]

    successes = [c for c in candidates if c["ok"] and c.get("text")]
    if not successes:
        errs = "; ".join(c.get("error") or "unknown" for c in candidates)
        return (
            None,
            f"all swarm agents failed: {errs}",
            summarize_failed_candidates(candidates),
        )

    winner = None
    rationale = ""

    if strategy in ("fanout",):
        winner = successes[0]
        rationale = "fanout completed; returning first successful response"
    elif strategy in ("first-success", "first_success"):
        winner = successes[0]
        rationale = "first agent to return a non-empty response"
    elif strategy == "longest":
        winner = max(successes, key=lambda c: len(c.get("text", "")))
        rationale = "longest non-empty response"
    elif len(successes) == 1:
        # best-of-n with a single survivor has nothing to compare; skip the
        # judge call entirely. Avoids spending a 200-token judge round on a
        # foregone conclusion AND prevents a busy judge model from blocking
        # the response.
        winner = successes[0]
        rationale = "single successful candidate; judge skipped"
    else:
        labels = [chr(ord("A") + i) for i in range(len(successes))]
        rendered = "\n\n".join(
            f"[{labels[i]}] (model={successes[i]['model']})\n{successes[i]['text']}"
            for i in range(len(successes))
        )
        original_user = deps.extract_user_intent({"messages": messages})
        judge_system = swarm_cfg.get("judge_system") or (
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
        judge_request = swarm_cfg.get("judge") or SWARM_CHAT_DEFAULT_JUDGE
        avail, _ = deps.get_available_models()
        judge_id, jerr = deps.resolve_model_id(judge_request, avail)

        if jerr or not judge_id:
            winner = max(successes, key=lambda c: len(c.get("text", "")))
            rationale = f"judge unavailable ({jerr or 'no model'}); picked longest"
        else:
            # Route the judge call through the same per-model semaphore the
            # agents use, so a busy judge model can't open a second concurrent
            # inference job against an already-loaded MoE.
            with per_model_semaphore(judge_id):
                jresp, jerr = deps.chat_completion(
                    judge_id, judge_messages, max_tokens=200, temperature=0.0
                )
            verdict = extract_text(jresp)
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
            "requested_model": requested_model,
        },
    }
    return out, None, None


def stream_swarm_body_as_sse(
    body: dict, *, chunk_chars: int = SWARM_STREAM_CHUNK_CHARS
) -> Iterator[str]:
    """Yield SSE-formatted ``data: ...`` lines from a swarm chat.completion
    body. The swarm itself is inherently batch (every candidate has to
    finish before the judge votes), so streaming clients get the winner's
    text sliced back into ``chat.completion.chunk`` deltas. Trailing
    ``data: [DONE]`` is always emitted so well-behaved consumers don't hang.

    Pure generator — no Flask types. The gateway wraps this into a
    streaming HTTP response.
    """
    response_id = body.get("id") or f"chatcmpl_{uuid.uuid4().hex}"
    created = int(body.get("created") or time.time())
    model = body.get("model") or "swarm/unknown"

    text = ""
    choices = body.get("choices") or []
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            text = str(msg.get("content") or "")

    swarm_meta = body.get("swarm") if isinstance(body.get("swarm"), dict) else None

    first = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    step = max(1, int(chunk_chars))
    if text:
        for i in range(0, len(text), step):
            piece = text[i: i + step]
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if swarm_meta is not None:
        final["swarm"] = swarm_meta
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def swarm_response_headers(body: dict) -> dict:
    """Build the gateway-agnostic ``X-Swarm-*`` / ``X-Model-Routed-To``
    headers for a swarm response. Caller (Flask gateway) merges these with
    its own transport headers (Cache-Control, etc.).
    """
    headers = {"X-Model-Routed-To": str(body.get("model") or "swarm/unknown")}
    swarm_meta = body.get("swarm") if isinstance(body.get("swarm"), dict) else None
    if isinstance(swarm_meta, dict):
        if swarm_meta.get("strategy"):
            headers["X-Swarm-Strategy"] = str(swarm_meta["strategy"])
        if swarm_meta.get("winner"):
            headers["X-Swarm-Winner"] = str(swarm_meta["winner"])
    return headers


__all__ = [
    # Constants
    "SWARM_PER_CALL_TIMEOUT",
    "SWARM_CHAT_ENABLED",
    "SWARM_CHAT_DEFAULT_MODELS",
    "SWARM_CHAT_DEFAULT_STRATEGY",
    "SWARM_CHAT_DEFAULT_JUDGE",
    "SWARM_CHAT_AUTO_MAX",
    "SWARM_STREAM_CHUNK_CHARS",
    "MAX_PARALLEL_MODEL_CALLS",
    "LM_STUDIO_PER_MODEL_INFLIGHT_CAP",
    "SWARM_ERROR_KINDS",
    "SWARM_CHAT_INTENTS",
    "SWARM_CHAT_CANONICAL",
    # Stateful helpers
    "per_model_semaphore",
    # Pure helpers
    "is_auto_swarm_token",
    "expand_swarm_models",
    "extract_upstream_status",
    "classify_swarm_error",
    "strip_upstream_prefix",
    "normalize_agent_spec",
    "extract_text",
    "spec_to_agent_id",
    "swarm_chat_intent",
    "is_swarm_chat_model",
    "summarize_failed_candidates",
    # IO runners (gateway-agnostic; take SwarmDeps)
    "SwarmDeps",
    "run_one_agent",
    "fanout",
    "run_swarm_chat_completion",
    "stream_swarm_body_as_sse",
    "swarm_response_headers",
]
