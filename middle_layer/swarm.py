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

import os
import re
import threading
import warnings


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
SWARM_CHAT_DEFAULT_MODELS = [
    m.strip()
    for m in os.environ.get(
        "SWARM_CHAT_DEFAULT_MODELS", "role:reasoner,role:coder,role:fast"
    ).split(",")
    if m.strip()
]

#: Default winner-pick strategy for ``swarmCouncil`` / ``swarmVote`` etc.
#: ``swarm/fanout`` always defaults to ``"fanout"`` regardless of this value.
SWARM_CHAT_DEFAULT_STRATEGY = os.environ.get(
    "SWARM_CHAT_DEFAULT_STRATEGY", "best-of-n"
).strip().lower()

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


def expand_swarm_models(spec, available=None, *, fetch_loaded=None):
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
    module pure — the real gateway-specific probe lives in the gateway.

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

    out: list = []
    seen: set = set()
    for entry in items:
        if is_auto_swarm_token(entry):
            for mid in (loaded or []):
                if not isinstance(mid, str) or mid in seen:
                    continue
                seen.add(mid)
                out.append(mid)
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


__all__ = [
    # Constants
    "SWARM_PER_CALL_TIMEOUT",
    "SWARM_CHAT_ENABLED",
    "SWARM_CHAT_DEFAULT_MODELS",
    "SWARM_CHAT_DEFAULT_STRATEGY",
    "SWARM_CHAT_DEFAULT_JUDGE",
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
]
