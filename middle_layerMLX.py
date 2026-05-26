"""
middle_layerMLX.py
==================

OpenAI-compatible Flask gateway that runs models directly via Apple's MLX
(`mlx_lm`) on Apple Silicon, with optional Anthropic Opus escalation and
multi-agent / swarm primitives that mirror middle_layer.py.

Key features:
  * Direct MLX inference (mlx_lm.generate / mlx_lm.stream_generate),
    proper chat-template application, full message-history forwarding.
  * MLXManager with per-alias load lock and per-model generation lock
    so concurrent requests don't corrupt model state.
  * model_profiles.json: per-alias / pattern defaults (context_window,
    max_tokens ceiling, sampler defaults, supports_vision, latency_tier, …)
    plus optional MLX_CONTEXT_OVER_BUDGET=trim and capability-based routing
    (vision, long prompt, X-MLX-Latency-Tier: fast).
  * Resolver supporting exact alias, comma-priority list, role:<name>,
    wildcard substring, and placeholder auto-pick.
  * Hybrid swarm endpoints that can mix local MLX models with Anthropic.
  * Swarm fanout uses a wall-clock deadline (SWARM_FANOUT_TIMEOUT or derived
    from SWARM_PER_CALL_TIMEOUT) and non-blocking executor shutdown so HTTP
    requests finish even if an agent stalls; SSE streams always emit [DONE].

Endpoints:
  GET  /healthz
  GET  /v1/models
  POST /v1/chat/completions          (OpenAI shape, streaming or not)
  POST /v1/completions               (OpenAI shape, streaming or not)
  DELETE /v1/models/<alias>           (explicit model unload)
  GET  /swarm/models
  POST /swarm/fanout
  POST /swarm/vote
  POST /swarm/pipeline
  POST /swarm/debate
  GET  /dashboard/                  (optional runtime UI; MLX_DASHBOARD_ENABLED=0 disables)
  GET  /dashboard/api/snapshot      (JSON metrics; auth if MIDDLE_LAYER_API_KEY set)
  POST /dashboard/api/preferences   (runtime default model + swarm presets JSON)
  POST /dashboard/api/models/load   (preload an MLX alias into LRU)

  Dashboard env: MLX_DASHBOARD_ENABLED, MLX_DASHBOARD_PREVIEW_CHARS, MLX_DASHBOARD_MAX_EVENTS,
  MLX_DASHBOARD_CAPTURE_PROMPTS, MLX_DASHBOARD_MAX_PROMPT_CHARS, MLX_DASHBOARD_MAX_ERROR_CHARS.

CLI usage:
  python middle_layerMLX.py serve                                     # full multi-model mode
  python middle_layerMLX.py serve --grab mlx-community/Qwen3-8B-MLX  # single-model grab
  python middle_layerMLX.py serve --grab /path/to/model               # single-model local
  python middle_layerMLX.py download mlx-community/Qwen3-8B-MLX      # download only

  On an interactive terminal, serve asks once which model to use as the
  session default (placeholder / auto resolution), unless DEFAULT_MODEL is
  set, or MLX_SKIP_STARTUP_MODEL_PROMPT=1, or you pass --no-pick-model.

Grab mode (single model, minimal surface):
  python middle_layerMLX.py serve --grab "hf:mlx-community/Some-Model-MLX-8bit"
  # or via env:
  export MLX_GRAB_MODEL="hf:mlx-community/Some-Model-MLX-8bit"
  export MLX_GRAB_DISPLAY_NAME="mlx"

  In grab mode, /v1/chat/completions ignores Anthropic escalation, ignores the
  resolver, and always runs the one mlx_lm-loaded model. Swarm routes 503.
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
import warnings
import uuid
import logging
import threading
import argparse
import contextlib
import requests
import copy
import heapq
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError
from flask import Flask, request, Response, stream_with_context

try:
    import mlx_dashboard as _mlx_dash
except ImportError:
    _mlx_dash = None  # type: ignore

# --- Optional CORS --------------------------------------------------------
try:
    from flask_cors import CORS as _FlaskCORS
except ImportError:
    _FlaskCORS = None

# --- Shared security helpers (Pass 4) -----------------------------------------
from middle_layer.security import apply_security_headers as _apply_security_headers  # noqa: E402
from middle_layer.security import check_api_key as _check_api_key  # noqa: E402
from middle_layer.security import enforce_safe_bind as _enforce_safe_bind  # noqa: E402
from middle_layer.security import PublicBindWithoutAuthError as _PublicBindWithoutAuthError  # noqa: E402
from middle_layer.security import resolve_max_request_bytes as _resolve_max_request_bytes  # noqa: E402

# --- Shared swarm primitives (Pass 3) -----------------------------------------
# The MLX gateway shares the error classifier, intent map, structured
# failure summarizer, and SSE generator with the LM Studio gateway. The IO
# runners (run_one_agent / fanout / run_swarm_chat_completion) stay
# MLX-specific because MLX has its own admission scheduler, fanout
# deadline, and judge-verdict parser that don't belong in the shared module.
from middle_layer.swarm import (  # noqa: E402, F401
    classify_swarm_error as _classify_swarm_error,
    extract_upstream_status as _extract_upstream_status,
    spec_to_agent_id as _spec_to_agent_id,
    strip_upstream_prefix as _strip_upstream_prefix,
    summarize_failed_candidates as _summarize_failed_candidates,
    swarm_chat_intent as _swarm_chat_intent,
    swarm_response_headers as _swarm_response_headers,
    SWARM_CHAT_CANONICAL as _SWARM_CHAT_CANONICAL,
    SWARM_CHAT_INTENTS as _SWARM_CHAT_INTENTS,
    SWARM_ERROR_KINDS as _SWARM_ERROR_KINDS,
)

# --- MLX (optional but expected) ---------------------------------------------
try:
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_lm = None  # type: ignore

# Newer mlx_lm exposes a sampler factory; older versions accept temp= directly.
try:
    from mlx_lm.sample_utils import make_sampler as _mlx_make_sampler  # type: ignore
except Exception:
    _mlx_make_sampler = None

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("middle_layerMLX")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = _resolve_max_request_bytes()

# =============================================================================
# CONFIGURATION
# =============================================================================


def _discover_model_root() -> str:
    """Return a sensible default MLX_MODEL_ROOT for this machine."""
    candidates = [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
        os.path.expanduser("~/.cache/mlx-models"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.expanduser("~/.cache/mlx-models")


MLX_MODEL_ROOT = os.environ.get("MLX_MODEL_ROOT", _discover_model_root())

# Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-4-opus-20250522")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2025-04-14")
ANTHROPIC_AUTO_ROUTE = os.environ.get("ANTHROPIC_AUTO_ROUTE", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

# Big-task escalation thresholds
BIG_TASK_MIN_WORDS = int(os.environ.get("BIG_TASK_MIN_WORDS", "80"))
BIG_TASK_MIN_CHARS = int(os.environ.get("BIG_TASK_MIN_CHARS", "500"))
BIG_TASK_MIN_BULLETS = int(os.environ.get("BIG_TASK_MIN_BULLETS", "4"))
BIG_TASK_MIN_STEP_MARKERS = int(os.environ.get("BIG_TASK_MIN_STEP_MARKERS", "3"))

# Auth (optional — header X-API-Key or Authorization: Bearer ...)
MIDDLE_LAYER_API_KEY = os.environ.get("MIDDLE_LAYER_API_KEY")

# MLX residency / concurrency
MAX_CONCURRENT_MODELS = int(os.environ.get("MAX_CONCURRENT_MODELS", "2"))
PRELOAD_MODELS = [s.strip() for s in os.environ.get("PRELOAD_MODELS", "").split(",") if s.strip()]

# Historical knob. Was advertised as "Flask threads" but never wired —
# ``app.run(host, port, threaded=True)`` doesn't take a worker cap, and
# this value was only logged. Real production concurrency is a job for
# the upstream WSGI server. Kept for one minor as a no-op with a
# ``DeprecationWarning`` when explicitly set, per AGENTS.md rule 1.
_MAX_WORKERS_ENV = os.environ.get("MAX_WORKERS")
if _MAX_WORKERS_ENV is not None:
    warnings.warn(
        "MAX_WORKERS is no longer honored: Flask's threaded=True does not "
        "take a worker cap, and this knob was only logged. Configure your "
        "upstream WSGI server (gunicorn --workers, uvicorn --workers, etc.) "
        "instead. The env var is ignored and will be removed in 0.2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
# Kept as a module-level symbol so any downstream import (notably the
# legacy startup banner) doesn't NameError mid-deprecation. Always 0
# (== "no in-process cap"). Do not read this for any new logic.
MAX_WORKERS = 0

# Swarm / multi-agent knobs
MAX_PARALLEL_MODEL_CALLS = int(os.environ.get("MAX_PARALLEL_MODEL_CALLS", "2"))
SWARM_PER_CALL_TIMEOUT = int(os.environ.get("SWARM_PER_CALL_TIMEOUT", "180"))
# 0 = derive from SWARM_PER_CALL_TIMEOUT and agent count (cap 3600s). Else wall-clock cap for whole fanout.
SWARM_FANOUT_TIMEOUT = int(os.environ.get("SWARM_FANOUT_TIMEOUT", "0"))
SWARM_CHAT_ENABLED = os.environ.get("SWARM_CHAT_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
SWARM_CHAT_DEFAULT_MODELS = [
    m.strip()
    for m in os.environ.get("SWARM_CHAT_DEFAULT_MODELS", "role:reasoner,role:coder,role:fast").split(",")
    if m.strip()
]
SWARM_CHAT_DEFAULT_STRATEGY = os.environ.get("SWARM_CHAT_DEFAULT_STRATEGY", "best-of-n").strip().lower()
SWARM_CHAT_DEFAULT_JUDGE = os.environ.get("SWARM_CHAT_DEFAULT_JUDGE", "role:reasoner").strip()
# Chunk size (in characters) for the synthetic SSE we emit when a streaming client
# requests a swarm meta-model. The swarm itself is inherently batch (needs every
# candidate before voting), so we stream the winner's text back in fixed slices.
SWARM_STREAM_CHUNK_CHARS = max(1, int(os.environ.get("SWARM_STREAM_CHUNK_CHARS", "64")))

# Default per-request generation knobs (overridable via JSON body).
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "1024"))
MAX_TOKENS_CEILING = int(os.environ.get("MAX_TOKENS_CEILING", "16384"))
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "300"))

# Per-model profiles (model_profiles.json); see get_model_profile().
MODEL_PROFILES_PATH = os.environ.get(
    "MODEL_PROFILES_FILE",
    os.path.join(os.path.dirname(__file__), "model_profiles.json"),
)

# Context over-budget: "error" (default) or "trim" (drop oldest user/assistant turns).
MLX_CONTEXT_OVER_BUDGET = os.environ.get("MLX_CONTEXT_OVER_BUDGET", "error").strip().lower()
MLX_CONTEXT_TRIM_BUFFER = int(os.environ.get("MLX_CONTEXT_TRIM_BUFFER", "8"))

# Rough pre-resolve prompt size for routing (chars / ratio ≈ tokens).
MLX_ROUTE_LONG_PROMPT_CHARS = int(os.environ.get("MLX_ROUTE_LONG_PROMPT_CHARS", "48000"))

# Per-model in-flight generation cap. Default changed in 0.1.x from 0
# (admission scheduler bypassed entirely) to 1 (one inference per alias,
# match the stable launcher). Direct ``python middle_layerMLX.py``
# invocations now get back-pressure by default instead of unbounded
# thread pile-up on ``gen_lock``. AGENTS.md rule 1: emit a one-shot
# ``DeprecationWarning`` when unset so operators can pin the legacy
# behavior with ``MLX_PER_MODEL_INFLIGHT_CAP=0``.
#
# ``MLX_PER_MODEL_ADMISSION_CAP`` is the historical name; honored as a
# fallback for one minor with a separate warning.
_legacy_admission_cap_raw = os.environ.get("MLX_PER_MODEL_ADMISSION_CAP")
if _legacy_admission_cap_raw is not None:
    warnings.warn(
        "MLX_PER_MODEL_ADMISSION_CAP is deprecated; use "
        "MLX_PER_MODEL_INFLIGHT_CAP. Honored as a fallback for now; "
        "will be removed in 0.2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
_legacy_admission_cap = int(_legacy_admission_cap_raw or "0")

_inflight_cap_env = os.environ.get("MLX_PER_MODEL_INFLIGHT_CAP")
if _inflight_cap_env is None and _legacy_admission_cap_raw is None:
    warnings.warn(
        "MLX_PER_MODEL_INFLIGHT_CAP is unset: defaulting to 1 (was 0). "
        "MLX gateway now applies per-alias admission control by default "
        "so direct invocations get the same back-pressure as the stable "
        "launcher. Set MLX_PER_MODEL_INFLIGHT_CAP=0 explicitly to keep "
        "the legacy 'no admission, rely only on gen_lock' behavior; "
        "will be removed in 0.2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    MLX_PER_MODEL_INFLIGHT_CAP = 1
elif _inflight_cap_env is None:
    MLX_PER_MODEL_INFLIGHT_CAP = _legacy_admission_cap
else:
    MLX_PER_MODEL_INFLIGHT_CAP = int(_inflight_cap_env)
MLX_QUEUE_MAX_PER_MODEL = int(os.environ.get("MLX_QUEUE_MAX_PER_MODEL", "32"))
MLX_QUEUE_MAX_TOTAL = int(os.environ.get("MLX_QUEUE_MAX_TOTAL", "128"))
MLX_QUEUE_WAIT_TIMEOUT_SEC = float(os.environ.get("MLX_QUEUE_WAIT_TIMEOUT_SEC", "20"))
MLX_QUEUE_RETRY_AFTER_SEC = int(os.environ.get("MLX_QUEUE_RETRY_AFTER_SEC", "2"))
MLX_QUEUE_RETRY_JITTER_SEC = int(os.environ.get("MLX_QUEUE_RETRY_JITTER_SEC", "1"))
MLX_QUEUE_PRIORITY_MIN = int(os.environ.get("MLX_QUEUE_PRIORITY_MIN", "-10"))
MLX_QUEUE_PRIORITY_MAX = int(os.environ.get("MLX_QUEUE_PRIORITY_MAX", "10"))
MLX_QUEUE_DEFAULT_PRIORITY = int(os.environ.get("MLX_QUEUE_DEFAULT_PRIORITY", "0"))

# CORS (opt-in: set CORS_ORIGINS to enable, e.g. "*" or "http://localhost:3000")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()

# Resolver placeholders / fallback policy
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
    """Merge core + optional extras. OpenClaw-specific ids default-on until 0.2.0."""
    raw = os.environ.get("EXTRA_PLACEHOLDER_MODELS")
    if raw is None:
        warnings.warn(
            "EXTRA_PLACEHOLDER_MODELS is unset: OpenClaw-specific placeholder model "
            "IDs remain enabled for one minor release. Set EXTRA_PLACEHOLDER_MODELS "
            "to a comma-separated list (or empty string to disable) for explicit "
            "control. Defaults change in 0.2.0.",
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
ON_MODEL_MISS = os.environ.get("ON_MODEL_MISS", "fallback").lower()  # "fallback" | "error"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "").strip()
MLX_FORCE_DEFAULT_MODEL = os.environ.get("MLX_FORCE_DEFAULT_MODEL", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Roles map a logical capability to one or more substrings/aliases (priority list).
DEFAULT_MODEL_ROLES = {
    "coder":    ["coder", "code"],
    "reasoner": ["72b", "70b", "qwen2.5", "llama-3.3", "deepseek-r1", "deepseek"],
    "fast":     ["3b", "7b", "phi", "mini", "small"],
    "vision":   ["vl", "vision", "llava"],
    "default":  [],
}


def _load_model_roles():
    raw = os.environ.get("MODEL_ROLES_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception as e:
            log.warning("MODEL_ROLES_JSON is not valid JSON: %s", e)
    path = os.environ.get("MODEL_ROLES_FILE")
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning("cannot load MODEL_ROLES_FILE=%s: %s", path, e)
    return dict(DEFAULT_MODEL_ROLES)


MODEL_ROLES = _load_model_roles()

# --- Per-model profiles (model_profiles.json) -------------------------------

_MODEL_PROFILES_DOC: dict | None = None


def _load_model_profiles_doc() -> dict:
    """Load model_profiles.json; returns dict with defaults, aliases, patterns."""
    global _MODEL_PROFILES_DOC
    if _MODEL_PROFILES_DOC is not None:
        return _MODEL_PROFILES_DOC
    defaults = {
        "context_window": 131072,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "max_tokens_ceiling": MAX_TOKENS_CEILING,
        "temperature_default": 0.7,
        "top_p_default": 0.95,
        "supports_vision": False,
        "supports_tools": True,
        "supports_json_mode": False,
        "latency_tier": "medium",
        "memory_gb_estimate": 8.0,
    }
    doc = {"defaults": dict(defaults), "aliases": {}, "patterns": []}
    path = MODEL_PROFILES_PATH
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                if isinstance(loaded.get("defaults"), dict):
                    doc["defaults"].update(loaded["defaults"])
                if isinstance(loaded.get("aliases"), dict):
                    doc["aliases"] = loaded["aliases"]
                if isinstance(loaded.get("patterns"), list):
                    doc["patterns"] = [p for p in loaded["patterns"] if isinstance(p, dict)]
        except Exception as e:
            log.warning("model_profiles load failed (%s): %s", path, e)
    _MODEL_PROFILES_DOC = doc
    return doc


def _shallow_merge_profile(base: dict, overlay: dict | None) -> dict:
    if not overlay:
        return base
    out = dict(base)
    for k, v in overlay.items():
        if v is not None:
            out[k] = v
    return out


def get_model_profile(alias: str) -> dict:
    """Effective profile for an MLX alias (defaults + patterns + exact alias + mlx_context_windows.json)."""
    doc = _load_model_profiles_doc()
    dfl = dict(doc["defaults"])
    alias_lc = (alias or "").lower()
    for pat in doc.get("patterns") or []:
        sub = (pat.get("substring") or "").lower()
        prof = pat.get("profile")
        if sub and sub in alias_lc and isinstance(prof, dict):
            dfl = _shallow_merge_profile(dfl, prof)
    exact = (doc.get("aliases") or {}).get(alias)
    if isinstance(exact, dict):
        dfl = _shallow_merge_profile(dfl, exact)
    cw_file = mlx_manager.context_windows.get(alias)
    if cw_file is None and isinstance(mlx_manager.context_windows, dict):
        for k, v in mlx_manager.context_windows.items():
            if isinstance(k, str) and k.lower() in alias_lc:
                cw_file = v
                break
    if isinstance(cw_file, int) and cw_file > 0:
        dfl["context_window"] = cw_file
    elif isinstance(cw_file, dict) and isinstance(cw_file.get("context_window"), int):
        dfl["context_window"] = cw_file["context_window"]
    return dfl


def _messages_contain_image_urls(messages) -> bool:
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    return True
        if isinstance(c, dict) and c.get("type") == "image_url":
            return True
    return False


def _rough_prompt_chars_for_routing(data: dict) -> int:
    n = 0
    messages = data.get("messages") or []
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str):
                n += len(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        n += len(p["text"])
    p = data.get("prompt")
    if isinstance(p, str):
        n += len(p)
    return n


def _infer_request_capabilities(data: dict, headers: dict | None) -> dict:
    """Hints for capability-based resolution (vision, long context, fast tier)."""
    caps: dict = {
        "needs_vision": _messages_contain_image_urls(data.get("messages") or []),
        "min_context_window": 0,
        "prefers_fast": False,
    }
    chars = _rough_prompt_chars_for_routing(data)
    if chars >= MLX_ROUTE_LONG_PROMPT_CHARS:
        est = max(8192, min(2_000_000, chars // 3 + 4096))
        caps["min_context_window"] = est
    if headers:
        tier = (headers.get("X-MLX-Latency-Tier") or headers.get("x-mlx-latency-tier") or "").strip().lower()
        if tier == "fast":
            caps["prefers_fast"] = True
    return caps


def _filter_aliases_by_capabilities(available: list[str], caps: dict) -> list[str]:
    if not available or not caps:
        return available
    out: list[str] = []
    for a in available:
        prof = get_model_profile(a)
        if caps.get("needs_vision") and not prof.get("supports_vision", False):
            continue
        mcw = int(caps.get("min_context_window") or 0)
        if mcw > 0 and int(prof.get("context_window") or 0) < mcw:
            continue
        if caps.get("prefers_fast"):
            tier = str(prof.get("latency_tier") or "").lower()
            mem = prof.get("memory_gb_estimate")
            if tier != "fast" and not (isinstance(mem, (int, float)) and mem <= 10.0):
                continue
        out.append(a)
    return out


# --- Admission scheduler (queue + in-flight semantics) ----------------------

_NO_DEADLINE = 10**18


class _AdmissionTicket:
    __slots__ = (
        "alias",
        "request_id",
        "priority",
        "stream",
        "deadline_mono",
        "enqueued_mono",
        "seq",
        "state",
        "wait_ms",
        "error_info",
    )

    def __init__(
        self,
        *,
        alias: str,
        request_id: str,
        priority: int,
        stream: bool,
        deadline_mono: float,
        enqueued_mono: float,
        seq: int,
    ) -> None:
        self.alias = alias
        self.request_id = request_id
        self.priority = priority
        self.stream = stream
        self.deadline_mono = deadline_mono
        self.enqueued_mono = enqueued_mono
        self.seq = seq
        self.state = "queued"  # queued | granted | expired | dropped
        self.wait_ms = 0
        self.error_info: dict | None = None


class _AdmissionScheduler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._inflight_by_alias: dict[str, int] = defaultdict(int)
        self._queued_by_alias: dict[str, list[tuple[int, float, int, _AdmissionTicket]]] = defaultdict(list)
        self._queued_count_by_alias: dict[str, int] = defaultdict(int)
        self._queued_alias_rr: deque[str] = deque()
        self._queued_total = 0
        self._seq = 0
        self._inflight_cap = max(0, int(MLX_PER_MODEL_INFLIGHT_CAP))
        self._queue_max_per_model = max(0, int(MLX_QUEUE_MAX_PER_MODEL))
        self._queue_max_total = max(0, int(MLX_QUEUE_MAX_TOTAL))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self._inflight_cap > 0,
                "inflight_cap_per_model": self._inflight_cap or None,
                "queue_max_per_model": self._queue_max_per_model or None,
                "queue_max_total": self._queue_max_total or None,
                "inflight_total": int(sum(self._inflight_by_alias.values())),
                "queued_total": int(self._queued_total),
                "inflight_by_alias": {
                    a: int(n) for a, n in self._inflight_by_alias.items() if n > 0
                },
                "queued_by_alias": {
                    a: int(n) for a, n in self._queued_count_by_alias.items() if n > 0
                },
            }

    def _queue_depth_locked(self, alias: str) -> int:
        return int(self._queued_count_by_alias.get(alias, 0))

    def _add_rr_alias_locked(self, alias: str) -> None:
        if alias not in self._queued_alias_rr:
            self._queued_alias_rr.append(alias)

    def _cleanup_alias_locked(self, alias: str) -> None:
        if self._queue_depth_locked(alias) > 0:
            return
        try:
            self._queued_alias_rr.remove(alias)
        except ValueError:
            pass
        self._queued_by_alias.pop(alias, None)
        self._queued_count_by_alias.pop(alias, None)

    def _drop_ticket_locked(self, ticket: _AdmissionTicket, error_info: dict) -> None:
        if ticket.state != "queued":
            return
        ticket.state = "dropped"
        ticket.error_info = error_info
        self._queued_total = max(0, self._queued_total - 1)
        self._queued_count_by_alias[ticket.alias] = max(
            0, self._queued_count_by_alias.get(ticket.alias, 0) - 1
        )
        self._cleanup_alias_locked(ticket.alias)

    def _expire_deadlines_locked(self, now: float) -> None:
        for alias, heap in list(self._queued_by_alias.items()):
            while heap:
                _nprio, deadline, _seq, ticket = heap[0]
                if ticket.state != "queued":
                    heapq.heappop(heap)
                    continue
                if deadline >= _NO_DEADLINE or deadline > now:
                    break
                heapq.heappop(heap)
                self._drop_ticket_locked(
                    ticket,
                    {
                        "type": "deadline_exceeded",
                        "status": 429,
                        "error": f"Queue deadline exceeded for model '{ticket.alias}'.",
                    },
                )
            self._cleanup_alias_locked(alias)

    def _pop_next_waiting_locked(self, alias: str, now: float) -> _AdmissionTicket | None:
        heap = self._queued_by_alias.get(alias)
        if not heap:
            return None
        while heap:
            _nprio, deadline, _seq, ticket = heapq.heappop(heap)
            if ticket.state != "queued":
                continue
            if deadline < _NO_DEADLINE and deadline <= now:
                self._drop_ticket_locked(
                    ticket,
                    {
                        "type": "deadline_exceeded",
                        "status": 429,
                        "error": f"Queue deadline exceeded for model '{alias}'.",
                    },
                )
                continue
            return ticket
        return None

    def _dispatch_locked(self, now: float) -> None:
        self._expire_deadlines_locked(now)
        if self._inflight_cap <= 0:
            return
        progressed = True
        while progressed and self._queued_alias_rr:
            progressed = False
            alias_count = len(self._queued_alias_rr)
            for _ in range(alias_count):
                if not self._queued_alias_rr:
                    break
                alias = self._queued_alias_rr[0]
                self._queued_alias_rr.rotate(-1)
                if self._inflight_by_alias.get(alias, 0) >= self._inflight_cap:
                    continue
                ticket = self._pop_next_waiting_locked(alias, now)
                if ticket is None:
                    self._cleanup_alias_locked(alias)
                    continue
                self._inflight_by_alias[alias] += 1
                self._queued_total = max(0, self._queued_total - 1)
                self._queued_count_by_alias[alias] = max(0, self._queued_count_by_alias.get(alias, 0) - 1)
                self._cleanup_alias_locked(alias)
                ticket.state = "granted"
                ticket.wait_ms = max(0, int((now - ticket.enqueued_mono) * 1000))
                progressed = True
            if progressed:
                self._cv.notify_all()

    def _clamp_priority(self, priority: int | None) -> int:
        if not isinstance(priority, int):
            priority = MLX_QUEUE_DEFAULT_PRIORITY
        return max(MLX_QUEUE_PRIORITY_MIN, min(MLX_QUEUE_PRIORITY_MAX, priority))

    def acquire(
        self,
        alias: str,
        *,
        request_id: str,
        priority: int | None,
        stream: bool,
        deadline_mono: float | None = None,
        max_wait_sec: float | None = None,
    ) -> tuple[bool, dict]:
        if self._inflight_cap <= 0:
            return True, {"queue_wait_ms": 0, "priority": 0}

        p = self._clamp_priority(priority)
        now = time.monotonic()
        wait_cap = MLX_QUEUE_WAIT_TIMEOUT_SEC if max_wait_sec is None else max(0.0, float(max_wait_sec))
        absolute_deadline = now + wait_cap
        if deadline_mono is not None:
            absolute_deadline = min(absolute_deadline, deadline_mono)

        with self._cv:
            self._dispatch_locked(now)
            if self._queue_max_total > 0 and self._queued_total >= self._queue_max_total:
                return False, {
                    "type": "queue_overloaded_total",
                    "status": 429,
                    "error": "Admission queue is full (global).",
                }
            if self._queue_max_per_model > 0 and self._queue_depth_locked(alias) >= self._queue_max_per_model:
                return False, {
                    "type": "queue_overloaded_model",
                    "status": 429,
                    "error": f"Admission queue is full for model '{alias}'.",
                }

            self._seq += 1
            ticket = _AdmissionTicket(
                alias=alias,
                request_id=request_id,
                priority=p,
                stream=bool(stream),
                deadline_mono=absolute_deadline,
                enqueued_mono=now,
                seq=self._seq,
            )
            deadline_key = ticket.deadline_mono if ticket.deadline_mono < _NO_DEADLINE else _NO_DEADLINE
            heapq.heappush(
                self._queued_by_alias[alias],
                (-ticket.priority, deadline_key, ticket.seq, ticket),
            )
            self._queued_total += 1
            self._queued_count_by_alias[alias] += 1
            self._add_rr_alias_locked(alias)
            self._dispatch_locked(now)

            while True:
                if ticket.state == "granted":
                    return True, {"queue_wait_ms": ticket.wait_ms, "priority": ticket.priority}
                if ticket.state == "dropped":
                    return False, ticket.error_info or {
                        "type": "queue_dropped",
                        "status": 429,
                        "error": f"Queue request dropped for model '{alias}'.",
                    }
                now = time.monotonic()
                if now >= ticket.deadline_mono:
                    self._drop_ticket_locked(
                        ticket,
                        {
                            "type": "queue_timeout",
                            "status": 429,
                            "error": (
                                f"Queue wait timeout exceeded for model '{alias}' "
                                f"({int(max(0.0, ticket.deadline_mono - ticket.enqueued_mono) * 1000)}ms budget)."
                            ),
                        },
                    )
                    self._cv.notify_all()
                    return False, ticket.error_info or {
                        "type": "queue_timeout",
                        "status": 429,
                        "error": f"Queue wait timeout exceeded for model '{alias}'.",
                    }
                self._dispatch_locked(now)
                remaining = max(0.001, min(0.5, ticket.deadline_mono - now))
                self._cv.wait(timeout=remaining)

    def release(self, alias: str) -> None:
        if self._inflight_cap <= 0:
            return
        with self._cv:
            self._inflight_by_alias[alias] = max(0, self._inflight_by_alias.get(alias, 0) - 1)
            if self._inflight_by_alias[alias] == 0:
                self._inflight_by_alias.pop(alias, None)
            self._dispatch_locked(time.monotonic())
            self._cv.notify_all()


_admission_scheduler = _AdmissionScheduler()


def _coerce_int(v, default: int | None = None) -> int | None:
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _queue_controls_from_request(data: dict | None, request_headers=None) -> dict:
    data = data or {}
    queue_cfg = data.get("queue")
    if not isinstance(queue_cfg, dict):
        queue_cfg = {}

    h = request_headers or {}
    prio = _coerce_int(
        queue_cfg.get("priority"),
        _coerce_int(h.get("X-MLX-Priority"), MLX_QUEUE_DEFAULT_PRIORITY),
    )
    max_wait_ms = _coerce_int(
        queue_cfg.get("max_queue_wait_ms"),
        _coerce_int(h.get("X-MLX-Max-Queue-Wait-Ms"), None),
    )
    deadline_ms = _coerce_int(
        queue_cfg.get("deadline_ms"),
        _coerce_int(h.get("X-MLX-Deadline-Ms"), None),
    )

    now_wall = time.time()
    deadline_mono = None
    if isinstance(deadline_ms, int) and deadline_ms > 0:
        # Accept either relative milliseconds or unix epoch milliseconds.
        if deadline_ms > 3_000_000_000:
            sec_left = max(0.0, deadline_ms / 1000.0 - now_wall)
            deadline_mono = time.monotonic() + sec_left
        else:
            deadline_mono = time.monotonic() + (deadline_ms / 1000.0)

    wait_sec = MLX_QUEUE_WAIT_TIMEOUT_SEC
    if isinstance(max_wait_ms, int) and max_wait_ms >= 0:
        wait_sec = max(0.0, max_wait_ms / 1000.0)

    return {
        "priority": prio if isinstance(prio, int) else MLX_QUEUE_DEFAULT_PRIORITY,
        "deadline_mono": deadline_mono,
        "max_wait_sec": wait_sec,
    }


def _admission_retry_hint(alias: str, error_info: dict, *, queue_wait_ms: int | None = None) -> tuple[dict, dict]:
    snap = _admission_scheduler.snapshot()
    retry_after = max(1, int(MLX_QUEUE_RETRY_AFTER_SEC))
    alias_q = int((snap.get("queued_by_alias") or {}).get(alias, 0))
    alias_inflight = int((snap.get("inflight_by_alias") or {}).get(alias, 0))
    payload = {
        "error": error_info.get("error") or "Admission queue overloaded.",
        "type": error_info.get("type") or "admission_rejected",
        "status": int(error_info.get("status") or 429),
        "retry": {
            "retry_after_sec": retry_after,
            "jitter_sec": max(0, int(MLX_QUEUE_RETRY_JITTER_SEC)),
            "strategy": "exponential_backoff_with_jitter",
        },
        "queue": {
            "model": alias,
            "queued_for_model": alias_q,
            "inflight_for_model": alias_inflight,
            "queued_total": int(snap.get("queued_total") or 0),
            "inflight_total": int(snap.get("inflight_total") or 0),
            "waited_ms": int(queue_wait_ms or 0),
        },
    }
    headers = {
        "Retry-After": str(retry_after),
        "X-MLX-Retry-Jitter-Sec": str(max(0, int(MLX_QUEUE_RETRY_JITTER_SEC))),
        "X-MLX-Queue-Depth": str(alias_q),
        "X-MLX-InFlight": str(alias_inflight),
    }
    return payload, headers


def _admission_acquire(
    alias: str,
    *,
    request_id: str,
    stream: bool,
    queue_controls: dict | None = None,
) -> tuple[bool, dict]:
    queue_controls = queue_controls or {}
    return _admission_scheduler.acquire(
        alias,
        request_id=request_id,
        priority=queue_controls.get("priority"),
        stream=bool(stream),
        deadline_mono=queue_controls.get("deadline_mono"),
        max_wait_sec=queue_controls.get("max_wait_sec"),
    )


def _admission_release(alias: str) -> None:
    _admission_scheduler.release(alias)


def _mlx_generate_text_timed(
    model, tokenizer, formatted_prompt, max_tokens, temperature=None, top_p=None,
    timeout_sec: int | None = None,
):
    """Run MLX generation synchronously.

    NOTE:
      mlx_lm generation is not safely cancellable mid-flight. The previous
      implementation returned early on timeout while generation continued in
      a background worker, which could violate per-model serialization.

      To preserve correctness, we run in-thread and treat timeout_sec as a
      soft budget for logging only.
    """
    t0 = time.perf_counter()
    out = _mlx_generate_text(model, tokenizer, formatted_prompt, max_tokens, temperature, top_p)
    if timeout_sec and timeout_sec > 0:
        elapsed = time.perf_counter() - t0
        if elapsed > float(timeout_sec):
            log.warning(
                "MLX generation exceeded soft timeout (%.2fs > %ss). "
                "Request completed to preserve model safety.",
                elapsed,
                timeout_sec,
            )
    return out


def _mlx_load_model(path: str):
    """Load MLX model + tokenizer, preferring fix_mistral_regex when supported."""
    try:
        return mlx_lm.load(path, fix_mistral_regex=True)
    except TypeError:
        # Older mlx_lm versions may not support this kwarg.
        return mlx_lm.load(path)


# Whether to force a ``gc.collect()`` after every Metal cache clear.
# Off by default — gc.collect() is noticeable wall time on macOS, and
# the Metal allocator usually reclaims promptly once Python refs drop.
# Operators on memory-tight Macs can opt in to trade latency for tighter
# RSS reclamation immediately after eviction.
MLX_FORCE_GC_ON_EVICT = os.environ.get("MLX_FORCE_GC_ON_EVICT", "0").strip().lower() not in {
    "0", "false", "no", "off", ""
}


def _try_clear_mlx_metal_cache() -> str | None:
    """Best-effort Metal cache release after MLX eviction or unload.

    The mlx_lm public API has churned across versions; feature-detect
    the available teardown call and silently skip on unsupported
    variants. Returns the name of the function actually called (for
    logging) or ``None`` if nothing was available.

    Eviction never fails on teardown errors — at worst we leak some
    Metal allocator slack until the next allocator pressure point.
    """
    try:
        import mlx.core as _mx  # type: ignore
    except ImportError:
        return None
    try:
        if hasattr(_mx, "metal") and hasattr(_mx.metal, "clear_cache"):
            _mx.metal.clear_cache()
            return "mx.metal.clear_cache"
        if hasattr(_mx, "clear_cache"):
            _mx.clear_cache()
            return "mx.clear_cache"
    except Exception as e:  # noqa: BLE001
        log.debug("Metal cache teardown failed (non-fatal): %s", e)
    return None


def _post_evict_cleanup(reason: str, alias: str) -> None:
    """Run after a model is evicted from the MLXManager registry —
    either by LRU eviction during a load or by an explicit (possibly
    deferred) unload. Triggers the best-effort Metal cache release
    and, if MLX_FORCE_GC_ON_EVICT is set, a Python-level gc.collect().

    Every step here is wrapped in try/except: post-evict cleanup is
    *opportunistic*. A failure to clear the Metal cache must never
    prevent the eviction from completing — at worst we leak some
    allocator slack until the next pressure point.
    """
    try:
        cleared = _try_clear_mlx_metal_cache()
        if cleared:
            log.debug("Post-evict cleanup '%s' (%s): cleared via %s", alias, reason, cleared)
    except Exception as e:  # noqa: BLE001
        log.debug(
            "Post-evict metal teardown for '%s' (%s) raised (non-fatal): %s",
            alias, reason, e,
        )
    if MLX_FORCE_GC_ON_EVICT:
        try:
            import gc as _gc
            collected = _gc.collect()
            log.debug(
                "Post-evict gc.collect '%s' (%s): freed %d objects",
                alias, reason, collected,
            )
        except Exception as e:  # noqa: BLE001
            log.debug(
                "Post-evict gc.collect for '%s' (%s) raised (non-fatal): %s",
                alias, reason, e,
            )


_MLX_OOM_HINT = (
    "Detected probable memory pressure. Try the stable launcher "
    "(./start_middle_layerMLX_5001_stable.sh) or lower MAX_CONCURRENT_MODELS, "
    "MLX_PER_MODEL_INFLIGHT_CAP, and max_tokens."
)


def _is_probable_oom_error(exc: BaseException | str) -> bool:
    txt = str(exc).lower()
    markers = (
        "out of memory",
        "oom",
        "mps backend out of memory",
        "resource exhausted",
        "killed",
        "std::bad_alloc",
        "allocation failed",
    )
    return any(m in txt for m in markers)


def _mlx_error_with_guidance(exc: BaseException | str, prefix: str) -> str:
    msg = f"{prefix}: {exc}"
    if _is_probable_oom_error(exc):
        return f"{msg}. {_MLX_OOM_HINT}"
    return msg


# --- CORS setup ---------------------------------------------------------------
if CORS_ORIGINS:
    if _FlaskCORS is not None:
        _FlaskCORS(app, origins=CORS_ORIGINS.split(","))
        log.info("CORS enabled: origins=%s", CORS_ORIGINS)
    else:
        @app.after_request
        def _cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = CORS_ORIGINS
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            return response

        @app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
        @app.route("/<path:path>", methods=["OPTIONS"])
        def _cors_preflight(path):
            return Response("", status=204, headers={
                "Access-Control-Allow-Origin": CORS_ORIGINS,
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                "Access-Control-Max-Age": "86400",
            })

        log.info("CORS enabled (manual): origins=%s", CORS_ORIGINS)


# =============================================================================
# MLX MODEL MANAGEMENT
# =============================================================================


class MLXManager:
    """Discovers, loads, and serializes access to MLX models on disk.

    `loaded_models[alias]` -> (model, tokenizer, gen_lock, last_used).

    Concurrency rules:
      * `_registry_lock` protects the OrderedDict (eviction, insert, lookup),
        the per-alias inflight refcount, and the deferred-unload set.
      * Each model has its own `gen_lock`; callers MUST hold it while running
        `mlx_lm.generate` / `mlx_lm.stream_generate` on that model.
      * Per-alias `_loading_locks[alias]` prevents two threads from loading
        the same model concurrently. Pruned on unload + when inflight==0.

    In-flight pinning (added in the swarm-intelligence-effectiveness +
    audit hardening pass):
      * Every inference site MUST go through ``acquire_inference_handle``
        (the context manager) instead of calling ``load_model`` directly.
        That increments an inflight refcount for the alias.
      * ``_ensure_capacity_locked`` skips pinned aliases when picking an
        eviction victim, so an LRU eviction never yanks a model out from
        under a running generation. (If *every* resident alias is pinned,
        a new load proceeds anyway and logs a warning — exceeding the
        configured cap is preferred to deadlocking the caller.)
      * ``unload_model`` is pin-aware: if the alias is in-flight, the
        unload is *deferred* and fires when the last holder releases.
    """

    def __init__(self, root_path):
        self.root_path = os.path.expanduser(root_path)
        self.registry: dict[str, str] = {}
        self.loaded_models: "OrderedDict[str, tuple]" = OrderedDict()
        self._registry_lock = threading.Lock()
        self._loading_locks: dict[str, threading.Lock] = {}
        self._last_load_errors: dict[str, str] = {}
        # Per-alias inflight refcount. Incremented by
        # ``acquire_inference_handle`` on entry, decremented on exit.
        # Protected by ``_registry_lock``.
        self._inflight: dict[str, int] = {}
        # Aliases whose ``unload_model`` was called while pinned. The
        # actual dict eviction happens when the last holder releases.
        self._unload_pending: set[str] = set()
        # Aliases that were evicted under a locked block and need a
        # post-evict Metal cleanup pass once the lock is released.
        # Drained by ``_drain_post_evict_cleanup`` which the caller
        # invokes *after* releasing ``_registry_lock`` so Metal
        # teardown doesn't block other manager operations.
        self._pending_post_evict_cleanup: list[str] = []
        self.context_windows: dict = {}
        self._scan()
        self._load_context_windows()

    # ---------- discovery ------------------------------------------------
    def _scan(self):
        if not os.path.exists(self.root_path):
            log.warning("MLX root does not exist: %s", self.root_path)
            return
        log.info("Scanning MLX models in: %s", self.root_path)
        for entry in os.scandir(self.root_path):
            if not entry.is_dir():
                continue
            cfg = os.path.join(entry.path, "config.json")
            if not os.path.exists(cfg):
                # Could be a publisher dir with model subdirs (LM Studio layout).
                try:
                    for sub in os.scandir(entry.path):
                        if sub.is_dir() and os.path.exists(os.path.join(sub.path, "config.json")):
                            alias = f"{entry.name}/{sub.name}"
                            self.registry[alias] = sub.path
                            log.info("  Found: %s", alias)
                except Exception:
                    pass
                continue
            alias = entry.name
            self.registry[alias] = entry.path
            log.info("  Found: %s", alias)

    def _load_context_windows(self):
        cfg = os.path.join(os.path.dirname(__file__), "mlx_context_windows.json")
        if os.path.exists(cfg):
            try:
                with open(cfg, "r") as f:
                    self.context_windows = json.load(f)
            except Exception:
                self.context_windows = {}

    def get_available_aliases(self):
        return list(self.registry.keys())

    def get_loaded_aliases(self):
        with self._registry_lock:
            return list(self.loaded_models.keys())

    def get_last_load_error(self, alias: str) -> str | None:
        with self._registry_lock:
            return self._last_load_errors.get(alias)

    def get_memory_stats(self):
        with self._registry_lock:
            return {
                "loaded": list(self.loaded_models.keys()),
                "count": len(self.loaded_models),
                "max": MAX_CONCURRENT_MODELS,
            }

    # ---------- loading (thread-safe) ------------------------------------
    def _evict_alias_locked(self, alias: str) -> None:
        """Caller MUST hold _registry_lock. Drop ``alias`` from the
        loaded dict and prune its per-alias load lock. Does NOT check
        inflight pinning — callers must do that.
        """
        self.loaded_models.pop(alias, None)
        self._loading_locks.pop(alias, None)
        self._unload_pending.discard(alias)

    def _drain_post_evict_cleanup(self, reason: str = "eviction") -> None:
        """Run ``_post_evict_cleanup`` for every alias queued under
        the registry lock. MUST be called outside the lock so Metal
        teardown doesn't block other manager operations.
        """
        with self._registry_lock:
            pending = list(self._pending_post_evict_cleanup)
            self._pending_post_evict_cleanup.clear()
        for alias in pending:
            _post_evict_cleanup(reason, alias)

    def _ensure_capacity_locked(self):
        """Caller MUST hold _registry_lock.

        Pick the oldest *unpinned* resident alias to evict until we're
        under the cap. If every resident alias is currently in-flight,
        let the new load proceed and exceed the cap rather than
        deadlocking — log loudly so the operator sees they need to
        raise ``MAX_CONCURRENT_MODELS`` or reduce request concurrency.

        Each successful eviction also triggers ``_post_evict_cleanup``
        which best-effort-clears the Metal allocator cache (and
        optionally runs ``gc.collect()`` when
        ``MLX_FORCE_GC_ON_EVICT=1``). The cleanup runs *after*
        releasing the registry lock so we don't block other callers
        on Metal teardown.
        """
        evicted_aliases: list[str] = []
        while len(self.loaded_models) >= MAX_CONCURRENT_MODELS:
            victim = None
            for alias in self.loaded_models:  # OrderedDict iteration is oldest-first
                if self._inflight.get(alias, 0) == 0:
                    victim = alias
                    break
            if victim is None:
                pinned = {
                    a: self._inflight.get(a, 0)
                    for a in self.loaded_models
                }
                log.warning(
                    "MAX_CONCURRENT_MODELS=%d exceeded: all %d resident "
                    "aliases are in-flight (%s). New load proceeding; "
                    "raise MAX_CONCURRENT_MODELS or reduce request "
                    "concurrency to stay within the cap.",
                    MAX_CONCURRENT_MODELS,
                    len(self.loaded_models),
                    pinned,
                )
                return
            self._evict_alias_locked(victim)
            evicted_aliases.append(victim)
            log.info("LRU eviction: '%s'", victim)
        # Defer the cleanup call to a sibling helper (called by load_model
        # after dropping the registry lock) so we don't pay Metal teardown
        # cost while holding the lock. We can't safely call it inline here
        # because callers nest this inside a locked block.
        if evicted_aliases:
            self._pending_post_evict_cleanup.extend(evicted_aliases)

    def unload_model(self, alias) -> dict:
        """Explicitly unload a model from memory.

        Returns ``{"unloaded": bool, "deferred": bool}``. If the alias
        is currently in-flight, the unload is deferred and fires when
        the last holder releases via ``acquire_inference_handle``'s
        context exit. Operators see ``deferred=True`` immediately so
        the HTTP DELETE caller knows the action was accepted.
        """
        with self._registry_lock:
            self._last_load_errors.pop(alias, None)
            if alias not in self.loaded_models:
                return {"unloaded": False, "deferred": False}
            if self._inflight.get(alias, 0) > 0:
                self._unload_pending.add(alias)
                log.info(
                    "Unload deferred for '%s' (inflight=%d); will fire on "
                    "last release",
                    alias,
                    self._inflight[alias],
                )
                return {"unloaded": False, "deferred": True}
            self._evict_alias_locked(alias)
            self._pending_post_evict_cleanup.append(alias)
            log.info(
                "Unloaded model '%s' (resident: %s)",
                alias,
                list(self.loaded_models.keys()),
            )
        # Drain Metal cleanup outside the registry lock so we don't
        # block other manager operations on Metal teardown.
        self._drain_post_evict_cleanup(reason="unload")
        return {"unloaded": True, "deferred": False}

    def pin_alias(self, alias: str) -> None:
        """Increment the inflight refcount for ``alias``. Caller MUST
        pair this with exactly one ``release_pin(alias)`` in a
        ``finally`` block. Use ``acquire_inference_handle`` when the
        load + pin + generation all happen in the same synchronous
        scope; use this lower-level method when the pin must outlive
        the function call (e.g. a Flask streaming generator)."""
        with self._registry_lock:
            self._inflight[alias] = self._inflight.get(alias, 0) + 1

    def release_pin(self, alias: str) -> None:
        """Decrement the inflight refcount for ``alias``. If the
        refcount hits zero and an unload was deferred while pinned,
        the actual eviction fires here."""
        deferred_drained = False
        with self._registry_lock:
            remaining = self._inflight.get(alias, 1) - 1
            if remaining <= 0:
                self._inflight.pop(alias, None)
                if alias in self._unload_pending:
                    self._evict_alias_locked(alias)
                    self._pending_post_evict_cleanup.append(alias)
                    deferred_drained = True
                    log.info(
                        "Deferred unload of '%s' fired (resident: %s)",
                        alias,
                        list(self.loaded_models.keys()),
                    )
            else:
                self._inflight[alias] = remaining
        if deferred_drained:
            self._drain_post_evict_cleanup(reason="deferred-unload")

    @contextlib.contextmanager
    def acquire_inference_handle(self, alias):
        """Reserve an alias for inference. Yields
        ``(model, tokenizer, gen_lock)`` or ``None`` if the load
        failed (caller checks the yielded value).

        While the context is active the alias is *pinned*:
        ``_ensure_capacity_locked`` will not evict it, and
        ``unload_model`` defers the actual drop until the last holder
        releases. This is the canonical entry point for every
        synchronous inference call site; for Flask streaming paths
        (where the generator outlives the function call), use
        ``pin_alias`` / ``release_pin`` directly and tie the release
        to the generator's ``finally`` block.
        """
        handle = self.load_model(alias)
        if handle is None:
            yield None
            return
        self.pin_alias(alias)
        try:
            yield handle
        finally:
            self.release_pin(alias)

    def load_model(self, alias):
        """Returns (model, tokenizer, gen_lock) or None.

        Safe for concurrent callers with different or identical aliases.
        """
        if not MLX_AVAILABLE:
            return None

        # Fast-path cache check.
        with self._registry_lock:
            if alias in self.loaded_models:
                model, tokenizer, gen_lock, _ = self.loaded_models[alias]
                self.loaded_models.move_to_end(alias)
                self.loaded_models[alias] = (model, tokenizer, gen_lock, time.time())
                return model, tokenizer, gen_lock
            path = self.registry.get(alias)
            if not path:
                return None
            per_alias = self._loading_locks.setdefault(alias, threading.Lock())

        # Serialize per-alias load. Other aliases may load concurrently.
        with per_alias:
            with self._registry_lock:
                if alias in self.loaded_models:
                    model, tokenizer, gen_lock, _ = self.loaded_models[alias]
                    self.loaded_models.move_to_end(alias)
                    self.loaded_models[alias] = (model, tokenizer, gen_lock, time.time())
                    return model, tokenizer, gen_lock

            log.info("Loading MLX model '%s' from %s", alias, path)
            try:
                model, tokenizer = _mlx_load_model(path)
            except Exception as e:
                err = _mlx_error_with_guidance(e, f"Failed to load MLX model '{alias}'")
                with self._registry_lock:
                    self._last_load_errors[alias] = err
                log.error("%s", err)
                return None

            with self._registry_lock:
                self._ensure_capacity_locked()
                gen_lock = threading.Lock()
                self.loaded_models[alias] = (model, tokenizer, gen_lock, time.time())
                self._last_load_errors.pop(alias, None)
            # Drain Metal cleanup for anything evicted during
            # _ensure_capacity_locked, outside the lock.
            self._drain_post_evict_cleanup(reason="lru-eviction")
            log.info("Loaded '%s' (resident: %s)", alias, self.get_loaded_aliases())
            return model, tokenizer, gen_lock


mlx_manager = MLXManager(MLX_MODEL_ROOT)

# -----------------------------------------------------------------------------
# Optional "grab mode": one local path OR one Hugging Face repo → mlx_lm.load
# -----------------------------------------------------------------------------

_GRAB: tuple | None = None
"""When set by init_mlx_grab_model(): (model, tokenizer, gen_lock, abs_path, label)."""


def init_mlx_grab_model() -> str | None:
    """If env MLX_GRAB_MODEL is set, resolve or download weights and load once.

    Accepted values:
      - Absolute or ~-expanded directory containing config.json
      - hf:ORG/MODEL  or  ORG/MODEL  (Hugging Face repo id; downloaded on startup)

    Returns an error string, or None on success. Leaves _GRAB == None if unset.
    """
    global _GRAB

    if _GRAB is not None:
        return None

    spec = os.environ.get("MLX_GRAB_MODEL", "").strip()
    if not spec:
        return None

    if not MLX_AVAILABLE:
        return "MLX_GRAB_MODEL is set but mlx_lm is not installed (pip install mlx-lm)."

    if spec.lower().startswith("hf:"):
        spec = spec[3:].strip()

    expanded = os.path.expanduser(spec)
    path: str | None = None

    if os.path.isdir(expanded) and os.path.isfile(os.path.join(expanded, "config.json")):
        path = os.path.abspath(expanded)
    elif re.fullmatch(r"[\w.-]+/[\w.-]+", spec):
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError:
            return "MLX_GRAB_MODEL is a Hugging Face repo but huggingface_hub is not installed (pip install huggingface_hub)."

        cache_root = os.path.expanduser(
            os.environ.get("MLX_GRAB_CACHE", os.path.join("~", ".cache", "mlx-middle-layer-grab"))
        )
        local_name = spec.replace("/", "__")
        dest = os.path.join(cache_root, local_name)
        os.makedirs(dest, exist_ok=True)
        log.info("MLX_GRAB_MODEL: downloading %r -> %s", spec, dest)
        snapshot_download(repo_id=spec, local_dir=dest, local_dir_use_symlinks=False)
        if not os.path.isfile(os.path.join(dest, "config.json")):
            return f"Download finished but config.json missing under {dest}"
        path = os.path.abspath(dest)
    else:
        return (
            f"MLX_GRAB_MODEL={spec!r} is neither a local model directory with config.json "
            f"nor a Hugging Face repo id like mlx-community/Qwen3-Next-MLX-8bit."
        )

    label = (os.environ.get("MLX_GRAB_DISPLAY_NAME") or "mlx").strip() or "mlx"

    log.info("MLX_GRAB_MODEL: loading MLX weights from %s", path)
    try:
        model, tokenizer = _mlx_load_model(path)
    except Exception as e:
        return f"_mlx_load_model({path!r}) failed: {e}"

    lock = threading.Lock()
    _GRAB = (model, tokenizer, lock, path, label)
    log.info("Grab mode ready — serving as model id %r (path %s)", label, path)
    return None


def _swarm_disabled_in_grab():
    return Response(
        json.dumps(
            {
                "error": "Swarm routes are disabled when MLX_GRAB_MODEL is set "
                "(single-model grab mode). Unset MLX_GRAB_MODEL to use /swarm/*."
            }
        ),
        status=503,
        mimetype="application/json",
    )


def _handle_grab_chat(data: dict, dash_rid: str | None = None, request_headers=None):
    """Run the single grabbed model; ignores client model + Anthropic."""
    assert _GRAB is not None
    if not dash_rid:
        dash_rid = str(uuid.uuid4())
    model, tokenizer, gen_lock, _path, label = _GRAB
    req_str = str(data.get("model")) if data.get("model") is not None else str(label)

    queue_controls = _queue_controls_from_request(data, request_headers)
    ok_ad, ad_meta = _admission_acquire(
        str(label),
        request_id=dash_rid,
        stream=bool(data.get("stream") is True),
        queue_controls=queue_controls,
    )
    if not ok_ad:
        payload, hint_headers = _admission_retry_hint(str(label), ad_meta)
        return Response(
            json.dumps(payload),
            status=int(payload.get("status") or 429),
            mimetype="application/json",
            headers=hint_headers,
        )
    queue_wait_ms = int(ad_meta.get("queue_wait_ms") or 0)

    def _grab_release():
        _admission_release(str(label))

    profile = get_model_profile(str(label))
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    temperature = _safe_float(data.get("temperature"))
    top_p = _safe_float(data.get("top_p"))
    temperature, top_p = _apply_sampler_defaults(temperature, top_p, profile)

    msgs, formatted, prompt_tok, trim_err = _trim_messages_for_context(
        tokenizer,
        messages,
        data.get("prompt") or "",
        int(profile.get("context_window") or 8192),
        _clamp_max_tokens(data.get("max_tokens"), profile),
        MLX_CONTEXT_TRIM_BUFFER,
    )
    if trim_err:
        _grab_release()
        return Response(json.dumps({"error": trim_err}), status=400, mimetype="application/json")
    if not formatted:
        _grab_release()
        return Response(json.dumps({"error": "empty prompt"}),
                        status=400, mimetype="application/json")

    max_tokens = _clamp_max_tokens(data.get("max_tokens"), profile)
    cw = int(profile.get("context_window") or 8192)
    max_tokens, _wc, impossible = _budget_max_tokens(
        prompt_tok, max_tokens, cw, MLX_CONTEXT_TRIM_BUFFER,
    )
    if impossible:
        _grab_release()
        return Response(
            json.dumps({
                "error": (
                    f"No completion budget left (prompt_tokens≈{prompt_tok}, context_window={cw}). "
                    "Shorten the prompt or raise context_window in model_profiles.json."
                ),
            }),
            status=400,
            mimetype="application/json",
        )

    response_id = f"chatcmpl_{uuid.uuid4().hex}"

    if data.get("stream") is True:
        def stream_generator():
            done_sent = False
            created = int(time.time())
            acc: list[str] = []
            if _mlx_dash is not None:
                _mlx_dash.metrics_store.active_enter(label)
            t_stream = time.perf_counter()

            def _err_chunk(exc: BaseException):
                return {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": label,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
                    "error": str(exc),
                }

            def _record_grab_stream(status: str, err_msg: str | None, comp_text: str):
                if _mlx_dash is None:
                    return
                elapsed_ms = max(1, int((time.perf_counter() - t_stream) * 1000))
                ct = _count_tokens(tokenizer, comp_text) if comp_text else 0
                pt = _count_tokens(tokenizer, formatted)
                prev = _mlx_dash.build_preview(
                    msgs,
                    formatted_prompt=formatted if getattr(_mlx_dash, "MLX_DASHBOARD_CAPTURE_PROMPTS", False) else None,
                )
                _mlx_dash.record_event(
                    request_id=dash_rid,
                    parent_request_id=None,
                    agent_slot=None,
                    route_kind="chat_grab_stream",
                    requested_model=req_str,
                    resolved_model=str(label),
                    backend="mlx",
                    stream=True,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    latency_ms=elapsed_ms,
                    status=status,
                    error_message=err_msg,
                    preview=prev,
                )

            try:
                with gen_lock:
                    first = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": label,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(first)}\n\n"
                    try:
                        for piece in _mlx_stream_text(model, tokenizer, formatted, max_tokens, temperature, top_p):
                            if piece:
                                acc.append(piece)
                            chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": label,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception as e:
                        guided_err = _mlx_error_with_guidance(e, "MLX generation error")
                        yield f"data: {json.dumps(_err_chunk(guided_err))}\n\n"
                        yield "data: [DONE]\n\n"
                        done_sent = True
                        _record_grab_stream("error", guided_err, "".join(acc))
                        return

                    final = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": label,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"
                    done_sent = True
                    _record_grab_stream("ok", None, "".join(acc))
            except GeneratorExit:
                raise
            except BaseException as e:
                log.exception("grab stream failed")
                if not done_sent:
                    guided_err = _mlx_error_with_guidance(e, "MLX stream failure")
                    try:
                        yield f"data: {json.dumps(_err_chunk(guided_err))}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception:
                        log.warning("grab stream could not emit terminal [DONE]", exc_info=True)
                    _record_grab_stream("error", guided_err, "".join(acc))
                return
            finally:
                if _mlx_dash is not None:
                    _mlx_dash.metrics_store.active_exit(label)
                _grab_release()

        return Response(
            stream_with_context(stream_generator()),
            mimetype="text/event-stream",
            headers={
                "X-Model-Routed-To": f"mlx-grab/{label}",
                "X-MLX-Grab-Path": _path,
                "X-MLX-Queue-Wait-Ms": str(queue_wait_ms),
            },
        )

    if _mlx_dash is not None:
        _mlx_dash.metrics_store.active_enter(label)
    t0 = time.perf_counter()
    try:
        with gen_lock:
            try:
                text, prompt_tok, comp_tok = _mlx_generate_text_timed(
                    model, tokenizer, formatted, max_tokens, temperature, top_p,
                    timeout_sec=GENERATION_TIMEOUT,
                )
            except TimeoutError as e:
                elapsed = int((time.perf_counter() - t0) * 1000)
                if _mlx_dash is not None:
                    _mlx_dash.record_event(
                        request_id=dash_rid,
                        parent_request_id=None,
                        agent_slot=None,
                        route_kind="chat_grab",
                        requested_model=req_str,
                        resolved_model=str(label),
                        backend="mlx",
                        stream=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=max(1, elapsed),
                        status="error",
                        error_message=str(e),
                        preview=_mlx_dash.build_preview(msgs),
                    )
                return Response(json.dumps({"error": str(e)}),
                                status=504, mimetype="application/json")
            except Exception as e:
                elapsed = int((time.perf_counter() - t0) * 1000)
                if _mlx_dash is not None:
                    _mlx_dash.record_event(
                        request_id=dash_rid,
                        parent_request_id=None,
                        agent_slot=None,
                        route_kind="chat_grab",
                        requested_model=req_str,
                        resolved_model=str(label),
                        backend="mlx",
                        stream=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=max(1, elapsed),
                        status="error",
                        error_message=str(e),
                        preview=_mlx_dash.build_preview(msgs),
                    )
                return Response(json.dumps({"error": _mlx_error_with_guidance(e, "MLX generation error")}),
                                status=500, mimetype="application/json")
    finally:
        if _mlx_dash is not None:
            _mlx_dash.metrics_store.active_exit(label)
        _grab_release()

    elapsed = int((time.perf_counter() - t0) * 1000)
    if _mlx_dash is not None:
        _mlx_dash.record_event(
            request_id=dash_rid,
            parent_request_id=None,
            agent_slot=None,
            route_kind="chat_grab",
            requested_model=req_str,
            resolved_model=str(label),
            backend="mlx",
            stream=False,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            latency_ms=max(1, elapsed),
            status="ok",
            error_message=None,
            preview=_mlx_dash.build_preview(msgs),
        )

    body = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": label,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text or ""},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": prompt_tok, "completion_tokens": comp_tok, "total_tokens": prompt_tok + comp_tok},
    }
    return Response(
        json.dumps(body),
        status=200,
        mimetype="application/json",
        headers={
            "X-Model-Routed-To": f"mlx-grab/{label}",
            "X-MLX-Latency-Ms": str(elapsed),
            "X-MLX-Grab-Path": _path,
            "X-MLX-Queue-Wait-Ms": str(queue_wait_ms),
        },
    )


# =============================================================================
# RESOLVER (alias / role / wildcard / priority list)
# =============================================================================


def _is_placeholder(name) -> bool:
    if name is None:
        return True
    if not isinstance(name, str):
        return True
    return name.strip().lower() in PLACEHOLDER_MODELS


def _match_one(needle: str, haystack):
    if not needle:
        return None
    n = needle.strip().lower()
    for mid in haystack:
        if mid.lower() == n:
            return mid
    for mid in haystack:
        if n in mid.lower():
            return mid
    return None


def _resolve_role(role: str, available):
    prefs = MODEL_ROLES.get(role.lower(), [])
    if isinstance(prefs, str):
        prefs = [prefs]
    for p in prefs:
        m = _match_one(p, available)
        if m:
            return m
    return None


def resolve_model_alias(requested, available=None, capabilities: dict | None = None):
    """Resolve a requested model spec to an MLX alias actually on disk.

    Same grammar as middle_layer.resolve_model_id:
      None / "" / "auto" / placeholder           -> auto-pick
      "exact-alias"                              -> exact, then substring
      "a,b,c"                                    -> priority list
      "role:coder"                               -> registry lookup
      "*coder*" / "qwen*"                        -> wildcard substring

    If `capabilities` is set (from _infer_request_capabilities), available aliases
    are filtered by profile metadata (vision, min context_window, fast tier).
    Returns (alias, error_message).
    """
    if available is None:
        available = mlx_manager.get_available_aliases()
    if not available:
        return None, "No MLX models available on disk."

    pool = list(available)
    if capabilities:
        filtered = _filter_aliases_by_capabilities(pool, capabilities)
        strict = capabilities.get("needs_vision") or int(capabilities.get("min_context_window") or 0) > 0
        if filtered:
            pool = filtered
        elif strict:
            reasons = []
            if capabilities.get("needs_vision"):
                reasons.append("needs_vision")
            if int(capabilities.get("min_context_window") or 0) > 0:
                reasons.append(f"min_context_window>={capabilities.get('min_context_window')}")
            return None, (
                "No MLX model satisfies routing constraints (" + ", ".join(reasons) + "). "
                "Update model_profiles.json or install a matching model."
            )

    if _is_placeholder(requested):
        if _mlx_dash is not None:
            rt = _mlx_dash.metrics_store.get_runtime_default_model()
            if rt:
                m = _match_one(rt, pool)
                if m:
                    return m, None
        if DEFAULT_MODEL:
            m = _match_one(DEFAULT_MODEL, pool)
            if m:
                return m, None
        m = _resolve_role("default", pool)
        if m:
            return m, None
        return pool[0], None

    candidates = [c.strip() for c in str(requested).split(",") if c.strip()]
    for cand in candidates:
        cand_lc = cand.lower()
        if cand_lc.startswith("role:"):
            m = _resolve_role(cand_lc.split(":", 1)[1], pool)
            if m:
                return m, None
            continue
        if "*" in cand:
            m = _match_one(cand.replace("*", ""), pool)
            if m:
                return m, None
            continue
        m = _match_one(cand, pool)
        if m:
            return m, None

    return None, f"No MLX model matched '{requested}'. Available: {pool}"


def _maybe_interactive_startup_model_pick(aliases: list[str], *, no_pick: bool) -> None:
    """Set DEFAULT_MODEL once from stdin when running serve on a TTY.

    Skipped for grab mode, non-TTY, --no-pick-model, MLX_SKIP_STARTUP_MODEL_PROMPT,
    or when DEFAULT_MODEL is already set (env).
    """
    global DEFAULT_MODEL

    if _GRAB is not None or not aliases:
        return
    if no_pick:
        return
    if os.environ.get("MLX_SKIP_STARTUP_MODEL_PROMPT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if (DEFAULT_MODEL or "").strip():
        return
    if not sys.stdin.isatty():
        return

    if len(aliases) == 1:
        only = aliases[0]
        DEFAULT_MODEL = only
        os.environ["DEFAULT_MODEL"] = only
        log.info("Single MLX model — session default: %s", only)
        return

    print("\nWhich MLX model should be the default for this session?", file=sys.stderr)
    print("(Used when clients send auto / mlx / empty model id.)\n", file=sys.stderr)
    for i, a in enumerate(aliases, 1):
        print(f"  {i}) {a}", file=sys.stderr)
    print("", file=sys.stderr)

    sel = aliases[0]
    while True:
        try:
            raw = input("Enter number or alias substring [1]: ").strip()
        except EOFError:
            sel = aliases[0]
            log.info("EOF on model prompt — defaulting to %s", sel)
            break
        if not raw:
            sel = aliases[0]
            break
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(aliases):
                sel = aliases[idx - 1]
                break
            print(f"  Please enter 1–{len(aliases)}.", file=sys.stderr)
            continue
        m = _match_one(raw, aliases)
        if m:
            sel = m
            break
        print(f"  No model matched {raw!r}. Try again.", file=sys.stderr)

    DEFAULT_MODEL = sel
    os.environ["DEFAULT_MODEL"] = sel
    log.info("Session default model (placeholder resolution): %s", sel)


def _maybe_interactive_startup_model_root_pick(current_root: str, *, no_pick: bool) -> str:
    """Ask once for MLX model root on TTY startup."""
    if no_pick:
        return current_root
    if os.environ.get("MLX_SKIP_STARTUP_ROOT_PROMPT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return current_root
    if not sys.stdin.isatty():
        return current_root

    print("\nWhere are your MLX model files located?", file=sys.stderr)
    print(f"Press Enter to keep current: {current_root}\n", file=sys.stderr)

    while True:
        try:
            raw = input(f"MLX model root [{current_root}]: ").strip()
        except EOFError:
            log.info("EOF on model-root prompt — keeping %s", current_root)
            return current_root
        if not raw:
            return current_root
        expanded = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(expanded):
            return expanded
        print(f"  Path does not exist or is not a directory: {expanded}", file=sys.stderr)


# =============================================================================
# OPENAI <-> MLX: prompt formatting & sampler
# =============================================================================


def _build_chat_prompt(tokenizer, messages, fallback_prompt=""):
    """Apply a chat template if available; otherwise fall back to a join
    that's at least readable to most models. Returns the formatted prompt
    string ready for mlx_lm.generate / stream_generate.
    """
    if isinstance(messages, list) and messages:
        try:
            apply = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply):
                # tokenize=False asks for a string back, not token ids.
                return apply(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
        except Exception as e:
            log.warning("apply_chat_template failed: %s; falling back.", e)

        # Fallback: simple "<role>: <content>" join with a final assistant cue.
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # OpenAI content-parts shape
                content = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if content:
                parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    return fallback_prompt or ""


def _build_sampler_kwargs(temperature=None, top_p=None):
    """Return a kwargs dict suitable for mlx_lm.generate/stream_generate.
    Prefers the modern `sampler=` interface and falls back to legacy `temp=`.
    """
    out = {}
    if _mlx_make_sampler is not None:
        sampler_kw = {}
        if temperature is not None:
            sampler_kw["temp"] = float(temperature)
        if top_p is not None:
            sampler_kw["top_p"] = float(top_p)
        if sampler_kw:
            try:
                out["sampler"] = _mlx_make_sampler(**sampler_kw)
                return out
            except Exception:
                pass
    if temperature is not None:
        out["temp"] = float(temperature)
    return out


def _clamp_max_tokens(requested, profile: dict | None = None) -> int:
    """Clamp max_tokens using global env caps merged with per-model profile."""
    ceiling = MAX_TOKENS_CEILING
    default = DEFAULT_MAX_TOKENS
    if profile:
        pceil = profile.get("max_tokens_ceiling")
        if isinstance(pceil, int) and pceil > 0:
            ceiling = min(ceiling, pceil)
        pdef = profile.get("default_max_tokens")
        if isinstance(pdef, int) and pdef > 0:
            default = pdef
    if not isinstance(requested, int) or requested < 1:
        return default
    return min(requested, ceiling)


def _apply_sampler_defaults(
    temperature,
    top_p,
    profile: dict | None,
):
    if profile:
        if temperature is None:
            t = profile.get("temperature_default")
            if isinstance(t, (int, float)):
                temperature = float(t)
        if top_p is None:
            p = profile.get("top_p_default")
            if isinstance(p, (int, float)):
                top_p = float(p)
    return temperature, top_p


def _safe_float(val, default=None):
    """Coerce to float if numeric, else return default."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _count_tokens(tokenizer, text: str) -> int:
    """Best-effort token count. Returns 0 on failure."""
    try:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            ids = encode(text)
            return len(ids) if ids else 0
    except Exception:
        pass
    return 0


def _trim_messages_for_context(
    tokenizer,
    messages: list,
    fallback_prompt: str,
    context_window: int,
    max_tokens: int,
    buffer: int,
) -> tuple[list, str, int, str | None]:
    """Drop oldest non-system chat turns until prompt tokens + max_tokens + buffer <= context_window.

    Returns (new_messages, formatted_prompt, prompt_tokens, error_or_none).
    """
    if not isinstance(messages, list) or not messages:
        fmt = fallback_prompt or ""
        pt = _count_tokens(tokenizer, fmt)
        if pt + max_tokens + buffer > context_window:
            return [], fmt, pt, (
                f"Prompt exceeds context_window={context_window} "
                f"(tokens≈{pt}, max_tokens={max_tokens}). Shorten the prompt."
            )
        return [], fmt, pt, None

    msgs = copy.deepcopy(messages)
    err: str | None = None
    while True:
        formatted = _build_chat_prompt(tokenizer, msgs, fallback_prompt=fallback_prompt)
        if not formatted:
            return msgs, "", 0, "empty prompt after context trim"
        pt = _count_tokens(tokenizer, formatted)
        if pt + max_tokens + buffer <= context_window:
            return msgs, formatted, pt, None
        if MLX_CONTEXT_OVER_BUDGET != "trim":
            err = (
                f"Context budget exceeded: prompt_tokens≈{pt}, max_tokens={max_tokens}, "
                f"context_window={context_window}. Set MLX_CONTEXT_OVER_BUDGET=trim to drop oldest turns, "
                "or reduce messages / max_tokens."
            )
            return msgs, formatted, pt, err
        # Remove oldest non-system message (preserve leading system blocks).
        idx = None
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") != "system":
                idx = i
                break
        if idx is None:
            err = (
                f"Cannot trim enough to fit context_window={context_window} "
                f"(prompt_tokens≈{pt})."
            )
            return msgs, formatted, pt, err
        msgs.pop(idx)
        if len(msgs) == 0:
            err = "Context trim removed all messages."
            return msgs, formatted, pt, err


def _budget_max_tokens(
    prompt_tokens: int,
    requested_max: int,
    context_window: int,
    buffer: int,
) -> tuple[int, bool, bool]:
    """Clamp max_tokens so prompt + completion (+ buffer) fits context_window.

    Returns (clamped_max_tokens, was_clamped, impossible) where impossible means
    no room remains for any completion.
    """
    room = context_window - prompt_tokens - buffer
    if room < 1:
        return 1, True, True
    if requested_max <= room:
        return requested_max, False, False
    return room, True, False


def _mlx_generate_text(model, tokenizer, formatted_prompt, max_tokens, temperature=None, top_p=None):
    """Direct synchronous MLX generation. Returns (text, prompt_tokens, completion_tokens)."""
    kwargs = {"max_tokens": int(max_tokens)}
    kwargs.update(_build_sampler_kwargs(temperature, top_p))
    try:
        result = mlx_lm.generate(model, tokenizer, prompt=formatted_prompt, **kwargs)
    except TypeError:
        for k in ("sampler", "top_p"):
            kwargs.pop(k, None)
        result = mlx_lm.generate(model, tokenizer, prompt=formatted_prompt, **kwargs)

    text = result if isinstance(result, str) else str(result)
    prompt_tokens = _count_tokens(tokenizer, formatted_prompt)
    completion_tokens = _count_tokens(tokenizer, text)
    return text, prompt_tokens, completion_tokens


def _mlx_stream_text(model, tokenizer, formatted_prompt, max_tokens, temperature=None, top_p=None):
    """Yields incremental text chunks from MLX streaming generation."""
    if not hasattr(mlx_lm, "stream_generate"):
        text, _, _ = _mlx_generate_text(model, tokenizer, formatted_prompt, max_tokens, temperature, top_p)
        yield text
        return

    kwargs = {"max_tokens": int(max_tokens)}
    kwargs.update(_build_sampler_kwargs(temperature, top_p))
    try:
        gen = mlx_lm.stream_generate(model, tokenizer, prompt=formatted_prompt, **kwargs)
    except TypeError:
        for k in ("sampler", "top_p"):
            kwargs.pop(k, None)
        gen = mlx_lm.stream_generate(model, tokenizer, prompt=formatted_prompt, **kwargs)

    for chunk in gen:
        # mlx_lm versions have varied: GenerationResponse w/ .text, or str, or tuple.
        text = getattr(chunk, "text", None)
        if text is None:
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, tuple) and chunk:
                text = str(chunk[0])
            else:
                text = str(chunk)
        if text:
            yield text


# =============================================================================
# UTILITIES
# =============================================================================


def _looks_like_code(text_lower: str) -> bool:
    return bool(
        re.search(r"[{};()\[\]]", text_lower)
        or re.search(r"\b(def|class|function|var|let|const|import|from|#include|traceback|stack trace)\b", text_lower)
        or "```" in text_lower
    )


def _is_big_task(text: str) -> bool:
    t = (text or "").strip()
    tl = t.lower()
    words = len(re.findall(r"\w+", t))
    if words >= BIG_TASK_MIN_WORDS or len(t) >= BIG_TASK_MIN_CHARS:
        return True
    bullets = len(re.findall(r"^\s*([-*]|\d+\.)\s+", t, flags=re.MULTILINE))
    if bullets >= BIG_TASK_MIN_BULLETS:
        return True
    markers = ("step ", "phase", "roadmap", "milestone", "end-to-end", "system design",
               "architecture", "tradeoff", "trade-offs", "migration plan", "rollout", "risks")
    if sum(1 for m in markers if m in tl) >= BIG_TASK_MIN_STEP_MARKERS:
        return True
    return False


def _extract_user_intent_text(json_data: dict) -> str:
    parts = []
    messages = json_data.get("messages", [])
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
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                            parts.append(p["text"])
    prompt = json_data.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)
    return "\n".join(parts).strip()


def _should_route_to_anthropic(endpoint: str, json_data: dict) -> bool:
    if endpoint != "chat/completions" or not ANTHROPIC_API_KEY or not ANTHROPIC_AUTO_ROUTE:
        return False
    text = _extract_user_intent_text(json_data)
    return bool(text) and not _looks_like_code(text.lower()) and _is_big_task(text)


def _active_default_model() -> str:
    """Return runtime/session default model preference, if any."""
    if _mlx_dash is not None:
        try:
            rt = _mlx_dash.metrics_store.get_runtime_default_model()
            if isinstance(rt, str) and rt.strip():
                return rt.strip()
        except Exception:
            pass
    return (DEFAULT_MODEL or "").strip()


# ---- OpenAI <-> Anthropic translation (mirrors middle_layer.py) -------------

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
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                texts = [p["text"] for p in content
                         if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)]
                text = "\n".join(texts).strip()
            if not text:
                continue
            if role == "system":
                system_chunks.append(text)
            elif role in ("user", "assistant"):
                out_messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    max_tokens = json_data.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = 1024

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": out_messages,
    }
    if system_chunks:
        payload["system"] = "\n".join(system_chunks).strip()
    temperature = json_data.get("temperature")
    if isinstance(temperature, (int, float)):
        payload["temperature"] = temperature
    return payload


def _anthropic_to_openai_chat_completion(anthropic_json: dict) -> dict:
    text_parts = []
    content = anthropic_json.get("content")
    if isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                text_parts.append(p["text"])
    assistant_text = "".join(text_parts)

    resp = {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"anthropic/{ANTHROPIC_MODEL}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant_text},
            "finish_reason": "stop",
        }],
    }
    usage = anthropic_json.get("usage")
    if isinstance(usage, dict):
        it, ot = usage.get("input_tokens"), usage.get("output_tokens")
        if isinstance(it, int) and isinstance(ot, int):
            resp["usage"] = {"prompt_tokens": it, "completion_tokens": ot, "total_tokens": it + ot}
    return resp


def _call_anthropic_chat(messages, model_override=None, **kwargs):
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


# =============================================================================
# CORE: run a single MLX inference (used by /v1/* and /swarm/*)
# =============================================================================


def _mlx_chat_completion(alias, messages, max_tokens=None, temperature=None, top_p=None, prompt=None):
    """Returns (openai_shaped_response, error). Acquires per-model gen_lock."""
    if not MLX_AVAILABLE:
        return None, "MLX not available (mlx_lm not installed)"

    ok_ad, ad_meta = _admission_acquire(
        alias,
        request_id=f"swarm_{uuid.uuid4().hex}",
        stream=False,
        queue_controls=None,
    )
    if not ok_ad:
        payload, _headers = _admission_retry_hint(alias, ad_meta)
        return None, payload.get("error") or "Admission rejected"
    try:
        with mlx_manager.acquire_inference_handle(alias) as handle:
            if handle is None:
                load_err = mlx_manager.get_last_load_error(alias)
                if load_err:
                    return None, load_err
                return None, f"Could not load MLX model '{alias}'"
            model, tokenizer, gen_lock = handle

            profile = get_model_profile(alias)
            temperature, top_p = _apply_sampler_defaults(temperature, top_p, profile)
            msgs, formatted, prompt_tok, trim_err = _trim_messages_for_context(
                tokenizer,
                messages if isinstance(messages, list) else [],
                prompt or "",
                int(profile.get("context_window") or 8192),
                _clamp_max_tokens(max_tokens, profile),
                MLX_CONTEXT_TRIM_BUFFER,
            )
            if trim_err:
                return None, trim_err
            if not formatted:
                return None, "empty prompt"

            mt = _clamp_max_tokens(max_tokens, profile)
            cw = int(profile.get("context_window") or 8192)
            mt, _clamped, impossible = _budget_max_tokens(prompt_tok, mt, cw, MLX_CONTEXT_TRIM_BUFFER)
            if impossible:
                return None, (
                    f"No completion budget left (prompt_tokens≈{prompt_tok}, context_window={cw}). "
                    "Shorten the prompt or choose a larger-context model."
                )

            t0 = time.time()
            with gen_lock:
                try:
                    text, prompt_tok, comp_tok = _mlx_generate_text_timed(
                        model, tokenizer, formatted, mt, temperature, top_p,
                        timeout_sec=GENERATION_TIMEOUT,
                    )
                except TimeoutError as e:
                    return None, str(e)
                except Exception as e:
                    return None, _mlx_error_with_guidance(e, "MLX generation error")
            elapsed = int((time.time() - t0) * 1000)

            return {
                "id": f"chatcmpl_{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": alias,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text or ""},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": comp_tok,
                    "total_tokens": prompt_tok + comp_tok,
                },
                "_meta": {"latency_ms": elapsed, "backend": "mlx"},
            }, None
    finally:
        _admission_release(alias)


def _extract_text(openai_response) -> str:
    if not isinstance(openai_response, dict):
        return ""
    choices = openai_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(msg, dict):
        return (msg.get("content") or "").strip()
    return ""


# =============================================================================
# SWARM AGENT RUNNER (MLX + Anthropic)
# =============================================================================


def _normalize_agent_spec(spec):
    if isinstance(spec, str):
        return {"model": spec}
    if isinstance(spec, dict):
        return dict(spec)
    return {"model": str(spec)}


def _run_one_agent(spec, default_messages, default_kwargs, available):
    """Returns (label, openai_response, error, latency_ms)."""
    spec = _normalize_agent_spec(spec)
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

    # Anthropic participant
    if requested_str.lower().startswith("anthropic"):
        override = None
        if ":" in requested_str:
            override = requested_str.split(":", 1)[1].strip() or None
        label = f"anthropic/{override or ANTHROPIC_MODEL}"
        t0 = time.time()
        resp, err = _call_anthropic_chat(msgs, model_override=override, **kwargs)
        return label, resp, err, int((time.time() - t0) * 1000)

    # MLX participant
    caps = _infer_request_capabilities({"messages": msgs}, None)
    alias, err = resolve_model_alias(requested, available, capabilities=caps)
    if err:
        return requested or "?", None, err, 0

    t0 = time.time()
    resp, err = _mlx_chat_completion(
        alias,
        msgs,
        max_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
    )
    return alias, resp, err, int((time.time() - t0) * 1000)


def _swarm_fanout_budget_seconds(spec_count: int) -> float:
    """Wall-clock budget for collecting all fanout agent results."""
    n = max(1, int(spec_count))
    if SWARM_FANOUT_TIMEOUT > 0:
        return float(SWARM_FANOUT_TIMEOUT)
    per = float(max(30, SWARM_PER_CALL_TIMEOUT))
    return float(min(3600.0, per * n))


def _parse_judge_verdict(verdict, labels):
    """Extract the chosen candidate index from a judge model's free-form reply.

    The prompt asks the judge to "reply with ONLY the letter on its own line",
    but real models routinely answer with ``**A**``, ``Answer: A``, ``[A]``,
    ``I pick A because…``, or just embed the letter in prose. A brittle
    ``^A\\b`` regex falls through to the "longest" fallback on any of those,
    which silently degrades vote into a length contest.

    Patterns are tried strict-first to keep ambiguous answers from being
    over-eagerly matched. Returns the index into ``labels`` of the selected
    candidate, or ``None`` if no label can be identified.
    """
    if not isinstance(verdict, str) or not verdict.strip():
        return None
    if not labels:
        return None
    text = verdict.strip()
    label_class = "[" + "".join(re.escape(lab) for lab in labels) + "]"

    patterns = (
        # Bare label on its own line, optionally bold / bracketed / parenthesized.
        rf"(?m)^\s*\**\(?\[?({label_class})\]?\)?\**\s*[.\):,!?\-]?\s*$",
        # Label at start of a line (with possible trailing prose).
        rf"(?m)^\s*\**({label_class})\**\b",
        # Explicit declarations: "answer is A", "pick A", "winner: A", etc.
        rf"(?i)(?:answer|pick|choose|winner|verdict|best|select|chosen|choice)"
        rf"(?:\s+is)?[:\s]+\**\(?\[?({label_class})\]?\)?\**\b",
        # Bold / bracketed / parenthesized letter anywhere.
        rf"\*\*({label_class})\*\*",
        rf"\[({label_class})\]",
        rf"\(({label_class})\)",
        # Last resort: first standalone label letter occurrence anywhere.
        rf"\b({label_class})\b",
    )

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            letter = m.group(1).upper()
            try:
                return labels.index(letter)
            except ValueError:
                continue
    return None


def _substitute_pipeline_template(template, ctx):
    """Substitute ``{{name}}`` placeholders without triggering ``str.format``.

    The previous implementation used ``re.sub`` to map ``{{name}}`` -> ``{name}``
    then ran ``template.format(**ctx)``, which exploded on any earlier stage
    output that contained literal ``{`` or ``}`` (e.g. Python f-strings, dict
    literals, JSON). The ``except (KeyError, IndexError)`` branch silently
    returned the original template with placeholders intact, so downstream
    stages received literal ``{{previous}}`` instead of the prior text.

    This version does direct replacement on ``{{name}}`` only, so values
    containing braces are inert. Unknown keys are left as their literal
    ``{{name}}`` placeholder (matches the prior behavior's intent).
    """
    if not isinstance(template, str):
        return template
    if not template or "{{" not in template:
        return template
    result = template
    if isinstance(ctx, dict):
        for key, value in ctx.items():
            if not isinstance(key, str):
                continue
            needle = "{{" + key + "}}"
            if needle in result:
                result = result.replace(needle, "" if value is None else str(value))
    return result


# Sentinel tokens that expand a swarm ``models`` list to whatever the MLX
# manager has currently in memory. Recognized in both ``swarm.models``
# (per-request) and the ``SWARM_CHAT_DEFAULT_MODELS`` env var. ``available``
# / ``configured`` map to the broader registry (lazy-loadable aliases) so
# users can opt into the wider fanout when nothing is preloaded.
_SWARM_AUTO_LOADED_TOKENS = frozenset({"auto", "loaded", "*", "all", "all-loaded"})
_SWARM_AUTO_AVAILABLE_TOKENS = frozenset({"available", "configured", "all-available"})
_SWARM_AUTO_TOKENS = _SWARM_AUTO_LOADED_TOKENS | _SWARM_AUTO_AVAILABLE_TOKENS


def _is_auto_swarm_token(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in _SWARM_AUTO_TOKENS


def _expand_swarm_models(spec):
    """Expand sentinel tokens in a swarm models list to MLX aliases.

    ``auto`` / ``loaded`` / ``*`` / ``all`` / ``all-loaded`` -> currently
    loaded MLX aliases (``mlx_manager.get_loaded_aliases()``). If nothing is
    loaded yet, falls back to every configured alias so the swarm still has
    something to fan out to (MLX lazy-loads on demand).

    ``available`` / ``configured`` / ``all-available`` -> every configured
    alias in the registry (``mlx_manager.get_available_aliases()``).

    Non-sentinel entries (exact aliases, ``role:...``, wildcards,
    ``anthropic[:model]``, etc.) pass through unchanged. Duplicate string
    entries are dropped to keep the fanout small.

    Returns ``(expanded_list, error_or_None)``. The caller should treat an
    empty result as a hard error.
    """
    if isinstance(spec, str):
        items = [spec]
    elif isinstance(spec, list):
        items = list(spec)
    else:
        return None, "swarm.models must be a list or sentinel string"

    loaded_cache = None
    available_cache = None

    out = []
    seen = set()
    for entry in items:
        if isinstance(entry, str) and entry.strip().lower() in _SWARM_AUTO_LOADED_TOKENS:
            if loaded_cache is None:
                loaded_cache = mlx_manager.get_loaded_aliases() or []
                if not loaded_cache:
                    # Lazy-load fallback: registry knows what *can* run.
                    loaded_cache = mlx_manager.get_available_aliases() or []
            for alias in loaded_cache:
                if not isinstance(alias, str) or alias in seen:
                    continue
                seen.add(alias)
                out.append(alias)
            continue
        if isinstance(entry, str) and entry.strip().lower() in _SWARM_AUTO_AVAILABLE_TOKENS:
            if available_cache is None:
                available_cache = mlx_manager.get_available_aliases() or []
            for alias in available_cache:
                if not isinstance(alias, str) or alias in seen:
                    continue
                seen.add(alias)
                out.append(alias)
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


def _fanout(specs, messages, common_kwargs, max_parallel=None, parent_request_id=None, route_kind="swarm_fanout"):
    if not specs:
        return None, "swarm requires at least one model"

    available = mlx_manager.get_available_aliases()
    if not available:
        # Anthropic-only swarms can still proceed.
        all_anthropic = all(
            (isinstance(s, str) and s.lower().startswith("anthropic"))
            or (isinstance(s, dict) and str(s.get("model", "")).lower().startswith("anthropic"))
            for s in specs
        )
        if not all_anthropic:
            return None, "No MLX models available on disk."

    cap = MAX_PARALLEL_MODEL_CALLS
    if isinstance(max_parallel, int) and max_parallel > 0:
        cap = min(cap, max_parallel)
    workers = max(1, min(cap, len(specs)))

    def _record_slot(fut, slot: int):
        try:
            label, resp, err, latency = fut.result()
        except Exception as exc:  # noqa: BLE001
            label, resp, latency = "?", None, 0
            err = str(exc)
        text = _extract_text(resp) if resp else ""
        api_ok = err is None and resp is not None
        error_kind = None
        http_status = None
        error_detail = None
        if not api_ok and err:
            http_status = _extract_upstream_status(err)
            error_kind = _classify_swarm_error(err, http_status=http_status)
            error_detail = _strip_upstream_prefix(err)
        elif api_ok and not text:
            error_kind = "empty_response"
            error_detail = (
                "MLX returned a response with empty assistant content "
                "(check reasoning_content / increase max_tokens)"
            )
            err = "empty assistant content"
        ok = api_ok and bool(text)
        results[slot] = {
            "agent_id": _spec_to_agent_id(specs[slot]),
            "model": label,
            "ok": ok,
            "error": err,
            "error_kind": error_kind,
            "http_status": http_status,
            "error_detail": error_detail,
            "latency_ms": latency,
            "response": resp,
            "text": text,
        }
        if _mlx_dash is not None and parent_request_id:
            spec = specs[slot]
            req_m = spec.get("model") if isinstance(spec, dict) else (spec if isinstance(spec, str) else None)
            pt, ct, tt = _mlx_dash.usage_from_openai_response(resp if isinstance(resp, dict) else None)
            backend = "anthropic" if str(label).lower().startswith("anthropic") else "mlx"
            _mlx_dash.record_event(
                request_id=f"{parent_request_id}_slot{slot}",
                parent_request_id=parent_request_id,
                agent_slot=slot,
                route_kind=route_kind,
                requested_model=str(req_m) if req_m is not None else None,
                resolved_model=str(label) if label is not None else None,
                backend=backend,
                stream=False,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                latency_ms=latency,
                status="ok" if err is None and resp is not None else "error",
                error_message=str(err) if err else None,
                preview=_mlx_dash.build_preview(messages),
            )

    results = [None] * len(specs)
    deadline = time.monotonic() + _swarm_fanout_budget_seconds(len(specs))
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            ex.submit(_run_one_agent, spec, messages, common_kwargs, available): i
            for i, spec in enumerate(specs)
        }
        pending = set(futures.keys())
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            slice_timeout = min(60.0, remaining)
            done, pending = wait(pending, timeout=slice_timeout, return_when=FIRST_COMPLETED)
            for fut in done:
                _record_slot(fut, futures[fut])
        for fut in pending:
            i = futures[fut]
            fut.cancel()
            timeout_err = (
                "Timed out waiting for agent (fanout deadline; "
                "set SWARM_FANOUT_TIMEOUT or SWARM_PER_CALL_TIMEOUT)."
            )
            results[i] = {
                "agent_id": _spec_to_agent_id(specs[i]),
                "model": "?",
                "ok": False,
                "error": timeout_err,
                "error_kind": "timeout",
                "http_status": None,
                "error_detail": timeout_err,
                "latency_ms": 0,
                "response": None,
                "text": "",
            }
            if _mlx_dash is not None and parent_request_id:
                spec = specs[i]
                req_m = spec.get("model") if isinstance(spec, dict) else (spec if isinstance(spec, str) else None)
                _mlx_dash.record_event(
                    request_id=f"{parent_request_id}_slot{i}_timeout",
                    parent_request_id=parent_request_id,
                    agent_slot=i,
                    route_kind=route_kind,
                    requested_model=str(req_m) if req_m is not None else None,
                    resolved_model=None,
                    backend="mlx",
                    stream=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=0,
                    status="error",
                    error_message=results[i]["error"],
                    preview=_mlx_dash.build_preview(messages),
                )
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            ex.shutdown(wait=False)
    return results, None


def _is_swarm_chat_model(requested_model) -> bool:
    """Back-compat predicate. Now delegates to the shared intent map so MLX
    and the LM Studio gateway recognize the exact same alias set, including
    ``swarmIntelligence`` (deprecated) which used to fall through here.
    """
    intent, _ = _swarm_chat_intent(requested_model)
    return intent is not None


def _run_swarm_chat_completion(
    requested_model: str,
    data: dict,
    parent_request_id: str | None = None,
    intent: str = "council",
):
    """Run swarm fanout/vote semantics and return ``(body, err, err_details)``.

    See ``docs/capabilities.md`` for the contract. Intent semantics match
    the LM Studio gateway:

      ``council``   → ``SWARM_CHAT_DEFAULT_STRATEGY`` (best-of-n with judge).
      ``fanout``    → ``"fanout"`` strategy by default (no judge ceremony).
      ``pipeline``  → 400 redirecting the caller to ``POST /swarm/pipeline``.
    """
    if intent == "pipeline":
        return None, (
            "swarm/pipeline cannot run on /v1/chat/completions because the "
            "OpenAI chat shape cannot carry 'stages[]'. Send your request to "
            "POST /swarm/pipeline (with {stages: [{model, prompt_prefix}, ...], "
            "input}) instead."
        ), None

    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None, "messages (list) is required for swarm chat", None

    common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p")}
    common = {k: v for k, v in common.items() if v is not None}

    swarm_cfg = data.get("swarm") if isinstance(data.get("swarm"), dict) else {}
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

    models, exp_err = _expand_swarm_models(models)
    if exp_err:
        return None, exp_err, None
    if not models:
        return None, (
            "swarm.models expanded to an empty set: no MLX models loaded or "
            "configured. Pass an explicit list or load/register at least one alias."
        ), None

    candidates, err = _fanout(
        models,
        messages,
        common,
        max_parallel=max_parallel,
        parent_request_id=parent_request_id,
        route_kind="chat_swarm_vote",
    )
    if err:
        return None, err, None

    successes = [c for c in candidates if c["ok"] and c.get("text")]
    if not successes:
        errs = "; ".join(c.get("error") or "unknown" for c in candidates)
        return (
            None,
            f"all swarm agents failed: {errs}",
            _summarize_failed_candidates(candidates),
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
        # judge call entirely. Same rationale as the LM Studio gateway.
        winner = successes[0]
        rationale = "single successful candidate; judge skipped"
    else:
        labels = [chr(ord("A") + i) for i in range(len(successes))]
        rendered = "\n\n".join(
            f"[{labels[i]}] (model={successes[i]['model']})\n{successes[i]['text']}"
            for i in range(len(successes))
        )
        original_user = _extract_user_intent_text({"messages": messages})
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
        avail = mlx_manager.get_available_aliases()
        judge_alias, jerr = resolve_model_alias(judge_request, avail)

        if jerr or not judge_alias:
            winner = max(successes, key=lambda c: len(c.get("text", "")))
            rationale = f"judge unavailable ({jerr or 'no model'}); picked longest"
        else:
            jresp, jerr = _mlx_chat_completion(judge_alias, judge_messages, max_tokens=200, temperature=0.0)
            if _mlx_dash is not None and parent_request_id:
                jlat = 0
                if isinstance(jresp, dict):
                    jlat = int((jresp.get("_meta") or {}).get("latency_ms") or 0)
                pt, ct, tt = _mlx_dash.usage_from_openai_response(jresp if isinstance(jresp, dict) else None)
                _mlx_dash.record_event(
                    request_id=f"{parent_request_id}_judge",
                    parent_request_id=parent_request_id,
                    agent_slot=None,
                    route_kind="chat_swarm_vote_judge",
                    requested_model=str(judge_request),
                    resolved_model=str(judge_alias),
                    backend="mlx",
                    stream=False,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    latency_ms=jlat,
                    status="ok" if not jerr else "error",
                    error_message=str(jerr) if jerr else None,
                    preview=_mlx_dash.build_preview(judge_messages),
                )
            verdict = _extract_text(jresp)
            picked_idx = _parse_judge_verdict(verdict, labels)
            if picked_idx is None:
                winner = max(successes, key=lambda c: len(c.get("text", "")))
                rationale = (
                    f"judge response unparseable; fell back to longest. "
                    f"Verdict: {(verdict or '')[:140]}"
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


def _swarm_body_to_sse_response(body: dict, *, chunk_chars: int = SWARM_STREAM_CHUNK_CHARS):
    """Wrap a non-stream swarm chat.completion as OpenAI SSE chunks.

    The swarm flow is inherently batch (every candidate must finish before the
    judge votes), so callers that asked for ``stream: true`` get the winner's
    text fed back as ``chat.completion.chunk`` deltas in fixed-size slices.
    Trailing ``data: [DONE]`` is always emitted, even on empty content.
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

    def _gen():
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
                piece = text[i : i + step]
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

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Model-Routed-To": str(model),
    }
    if isinstance(swarm_meta, dict):
        if swarm_meta.get("strategy"):
            headers["X-Swarm-Strategy"] = str(swarm_meta["strategy"])
        if swarm_meta.get("winner"):
            headers["X-Swarm-Winner"] = str(swarm_meta["winner"])
    return Response(
        stream_with_context(_gen()),
        status=200,
        mimetype="text/event-stream",
        headers=headers,
    )


# =============================================================================
# AUTH GUARD
# =============================================================================


@app.before_request
def _auth_guard():
    if not MIDDLE_LAYER_API_KEY:
        return None
    p = request.path or ""
    if request.method == "GET" and (
        p == "/dashboard/"
        or (p.startswith("/dashboard/") and not p.startswith("/dashboard/api/"))
    ):
        return None
    if _check_api_key(request.headers, MIDDLE_LAYER_API_KEY):
        return None
    return Response(
        json.dumps({"error": "Unauthorized"}),
        status=401,
        mimetype="application/json",
    )


@app.before_request
def _stamp_start_time():
    request._mlx_start = time.time()


@app.after_request
def _log_request(response):
    elapsed = int((time.time() - getattr(request, "_mlx_start", time.time())) * 1000)
    model_routed = response.headers.get("X-Model-Routed-To", "-")
    log.info(
        "%s %s -> %d (%dms) model=%s",
        request.method, request.path, response.status_code, elapsed, model_routed,
    )
    return response


@app.after_request
def _security_headers(response):
    _apply_security_headers(response, path=request.path or "")
    return response


# =============================================================================
# OPENAI-COMPATIBLE ROUTES
# =============================================================================


@app.route("/healthz", methods=["GET"])
def healthz():
    if _GRAB is not None:
        _m, _t, _l, pth, lbl = _GRAB
        ok = MLX_AVAILABLE
        return Response(
            json.dumps({
                "ok": ok,
                "mode": "grab",
                "mlx_available": MLX_AVAILABLE,
                "mlx_grab_path": pth,
                "mlx_grab_display_name": lbl,
                "anthropic_enabled": False,
                "note": "MLX_GRAB_MODEL mode: single model only; swarm routes disabled.",
                "dashboard": "http://<host>:<port>/dashboard/" if _mlx_dash is not None else None,
            }),
            status=200 if ok else 503,
            mimetype="application/json",
        )

    available = mlx_manager.get_available_aliases()
    loaded = mlx_manager.get_loaded_aliases()
    ok = MLX_AVAILABLE and bool(available)
    return Response(
        json.dumps({
            "ok": ok,
            "mlx_available": MLX_AVAILABLE,
            "mlx_root": mlx_manager.root_path,
            "mlx_models": available,
            "mlx_loaded": loaded,
            "memory": mlx_manager.get_memory_stats(),
            "model_roles": MODEL_ROLES,
            "default_model": DEFAULT_MODEL or None,
            "max_parallel": MAX_PARALLEL_MODEL_CALLS,
            "swarm_fanout_timeout_sec": SWARM_FANOUT_TIMEOUT or None,
            "swarm_fanout_budget_example_sec": _swarm_fanout_budget_seconds(3),
            "on_model_miss": ON_MODEL_MISS,
            "anthropic_enabled": bool(ANTHROPIC_API_KEY),
            "anthropic_model": ANTHROPIC_MODEL,
            "swarm_chat_enabled": bool(SWARM_CHAT_ENABLED),
            "swarm_chat_default_models": SWARM_CHAT_DEFAULT_MODELS,
            "swarm_chat_default_strategy": SWARM_CHAT_DEFAULT_STRATEGY,
            "swarm_chat_auto_tokens_loaded": sorted(_SWARM_AUTO_LOADED_TOKENS),
            "swarm_chat_auto_tokens_available": sorted(_SWARM_AUTO_AVAILABLE_TOKENS),
            "swarm_chat_canonical": _SWARM_CHAT_CANONICAL,
            "swarm_chat_aliases": {
                name: {"intent": intent, "deprecated": deprecated}
                for name, (intent, deprecated) in sorted(_SWARM_CHAT_INTENTS.items())
            },
            "dashboard": "http://<host>:<port>/dashboard/" if _mlx_dash is not None else None,
            "model_profiles_file": MODEL_PROFILES_PATH if os.path.isfile(MODEL_PROFILES_PATH) else None,
            "mlx_per_model_admission_cap_legacy": _legacy_admission_cap or None,
            "mlx_per_model_inflight_cap": MLX_PER_MODEL_INFLIGHT_CAP or None,
            "mlx_queue_max_per_model": MLX_QUEUE_MAX_PER_MODEL or None,
            "mlx_queue_max_total": MLX_QUEUE_MAX_TOTAL or None,
            "mlx_queue_wait_timeout_sec": MLX_QUEUE_WAIT_TIMEOUT_SEC,
            "mlx_context_over_budget": MLX_CONTEXT_OVER_BUDGET,
            "generation_timeout_sec": GENERATION_TIMEOUT,
            "admission": _admission_scheduler.snapshot(),
        }),
        status=200 if ok else 503,
        mimetype="application/json",
    )


@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible model list."""
    now = int(time.time())
    if _GRAB is not None:
        lbl = _GRAB[4]
        return Response(
            json.dumps({
                "object": "list",
                "data": [{"id": lbl, "object": "model", "created": now, "owned_by": "mlx-grab"}],
            }),
            status=200,
            mimetype="application/json",
        )

    aliases = mlx_manager.get_available_aliases()
    return Response(
        json.dumps({
            "object": "list",
            "data": [
                {"id": a, "object": "model", "created": now, "owned_by": "mlx"}
                for a in aliases
            ],
        }),
        status=200,
        mimetype="application/json",
    )


@app.route("/v1/models/<path:alias>", methods=["DELETE"])
def unload_model(alias):
    """Explicitly unload a model from memory to free RAM.

    Returns:
      200 + ``unloaded=True``  → model was resident and dropped immediately.
      202 + ``deferred=True``  → model was in-flight; unload will fire when
                                 the last holder releases. (HTTP 202 Accepted
                                 is the correct shape for "request received,
                                 action queued".)
      404 + ``unloaded=False`` → model was not loaded to begin with.
    """
    if _GRAB is not None:
        return Response(
            json.dumps({"error": "Cannot unload in grab mode (single-model)."}),
            status=400, mimetype="application/json",
        )
    result = mlx_manager.unload_model(alias)
    if result["unloaded"]:
        return Response(
            json.dumps({
                "ok": True,
                "unloaded": alias,
                "deferred": False,
                "resident": mlx_manager.get_loaded_aliases(),
            }),
            status=200, mimetype="application/json",
        )
    if result["deferred"]:
        return Response(
            json.dumps({
                "ok": True,
                "unloaded": None,
                "deferred": True,
                "alias": alias,
                "note": (
                    f"Model '{alias}' is in-flight; unload deferred until "
                    "the last active request releases."
                ),
                "resident": mlx_manager.get_loaded_aliases(),
            }),
            status=202, mimetype="application/json",
        )
    return Response(
        json.dumps({
            "ok": False,
            "error": f"Model '{alias}' is not currently loaded.",
            "resident": mlx_manager.get_loaded_aliases(),
        }),
        status=404, mimetype="application/json",
    )


def _handle_chat_request(data, request_headers=None):
    """Shared body for /v1/chat/completions and /v1/completions."""
    if not MLX_AVAILABLE:
        return Response(json.dumps({"error": "MLX unavailable"}),
                        status=503, mimetype="application/json")

    if not data:
        return Response(json.dumps({"error": "Invalid JSON"}),
                        status=400, mimetype="application/json")

    dash_rid = str(uuid.uuid4())
    caps = _infer_request_capabilities(data, request_headers)

    # Single-model "grab" mode: no Anthropic, no resolver — always the one load.
    if _GRAB is not None:
        return _handle_grab_chat(data, dash_rid=dash_rid, request_headers=request_headers)

    requested = data.get("model")
    swarm_intent, swarm_canonical = (None, None)
    if SWARM_CHAT_ENABLED:
        swarm_intent, swarm_canonical = _swarm_chat_intent(requested)

    if swarm_intent is not None:
        # ``pipeline`` is a 400 (caller misuse), not a 502 (upstream
        # failure) — same convention as the LM Studio gateway.
        if swarm_intent == "pipeline":
            _body, err_msg, _ = _run_swarm_chat_completion(
                str(requested), data, parent_request_id=dash_rid, intent="pipeline",
            )
            return Response(
                json.dumps({"error": err_msg, "redirect": "POST /swarm/pipeline"}),
                status=400,
                mimetype="application/json",
            )

        wants_stream = data.get("stream") is True
        body, swarm_err, swarm_err_details = _run_swarm_chat_completion(
            str(requested),
            data,
            parent_request_id=dash_rid,
            intent=swarm_intent,
        )
        if swarm_err or not body:
            err_body: dict = {"error": f"Swarm routing failed: {swarm_err}"}
            if swarm_err_details:
                err_body["error_details"] = swarm_err_details
            err_headers: dict = {}
            if swarm_canonical and swarm_canonical.lower() != str(requested).strip().lower():
                err_headers["X-Swarm-Canonical-Name"] = swarm_canonical
            if isinstance(swarm_err_details, dict):
                kinds = swarm_err_details.get("kinds") or {}
                if kinds:
                    err_headers["X-Swarm-Error-Kinds"] = ",".join(
                        f"{k}={v}" for k, v in sorted(kinds.items())
                    )
            return Response(
                json.dumps(err_body),
                status=502,
                mimetype="application/json",
                headers=err_headers or None,
            )
        if wants_stream:
            return _swarm_body_to_sse_response(body)
        resp_headers = {
            "X-Model-Routed-To": str(body.get("model", "swarm/unknown")),
            "X-Swarm-Intent": swarm_intent,
        }
        if swarm_canonical and swarm_canonical.lower() != str(requested).strip().lower():
            resp_headers["X-Swarm-Canonical-Name"] = swarm_canonical
        return Response(
            json.dumps(body),
            status=200,
            mimetype="application/json",
            headers=resp_headers,
        )

    # Anthropic escalation for big non-code tasks (matches middle_layer.py)
    if _should_route_to_anthropic("chat/completions", data):
        if data.get("stream") is True:
            return Response(
                json.dumps({"error": "Streaming via Anthropic routing not enabled. Set stream=false."}),
                status=501, mimetype="application/json",
            )
        anthropic_payload = _openai_messages_to_anthropic(data)
        t_anth = time.perf_counter()
        try:
            r = requests.post(
                f"{ANTHROPIC_BASE_URL}/v1/messages",
                headers={
                    "content-type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                data=json.dumps(anthropic_payload).encode("utf-8"),
                timeout=GENERATION_TIMEOUT,
            )
            elapsed_ms = int((time.perf_counter() - t_anth) * 1000)
            if r.status_code >= 400:
                if _mlx_dash is not None:
                    _mlx_dash.record_event(
                        request_id=dash_rid,
                        parent_request_id=None,
                        agent_slot=None,
                        route_kind="chat_anthropic",
                        requested_model=str(data.get("model")),
                        resolved_model=str(ANTHROPIC_MODEL),
                        backend="anthropic",
                        stream=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=elapsed_ms,
                        status="error",
                        error_message=r.text[:500] if r.text else f"HTTP {r.status_code}",
                        preview=_mlx_dash.build_preview(data.get("messages")),
                    )
                return Response(r.content, status=r.status_code,
                                mimetype=r.headers.get("content-type", "application/json"))
            body = _anthropic_to_openai_chat_completion(r.json())
            if _mlx_dash is not None:
                pt, ct, tt = _mlx_dash.usage_from_openai_response(body)
                _mlx_dash.record_event(
                    request_id=dash_rid,
                    parent_request_id=None,
                    agent_slot=None,
                    route_kind="chat_anthropic",
                    requested_model=str(data.get("model")),
                    resolved_model=str(ANTHROPIC_MODEL),
                    backend="anthropic",
                    stream=False,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    latency_ms=elapsed_ms,
                    status="ok",
                    error_message=None,
                    preview=_mlx_dash.build_preview(data.get("messages")),
                )
            return Response(
                json.dumps(body),
                status=200, mimetype="application/json",
                headers={"X-Model-Routed-To": f"anthropic/{ANTHROPIC_MODEL}"},
            )
        except Exception as e:
            if _mlx_dash is not None:
                elapsed_ms = int((time.perf_counter() - t_anth) * 1000)
                _mlx_dash.record_event(
                    request_id=dash_rid,
                    parent_request_id=None,
                    agent_slot=None,
                    route_kind="chat_anthropic",
                    requested_model=str(data.get("model")),
                    resolved_model=str(ANTHROPIC_MODEL),
                    backend="anthropic",
                    stream=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=elapsed_ms,
                    status="error",
                    error_message=str(e),
                    preview=_mlx_dash.build_preview(data.get("messages")),
                )
            return Response(json.dumps({"error": f"Anthropic error: {e}"}),
                            status=502, mimetype="application/json")

    # Resolve which MLX alias to run
    forced_default_from = None
    if MLX_FORCE_DEFAULT_MODEL:
        active_default = _active_default_model()
        if active_default:
            forced_default_from = requested
            requested = active_default
    alias, err = resolve_model_alias(requested, capabilities=caps)
    fallback_from = None
    if err:
        if not _is_placeholder(requested) and ON_MODEL_MISS == "fallback":
            available = mlx_manager.get_available_aliases()
            if available:
                alias = available[0]
                fallback_from = requested
                err = None
        if err:
            return Response(json.dumps({"error": err}),
                            status=503, mimetype="application/json")

    queue_controls = _queue_controls_from_request(data, request_headers)
    ok_ad, ad_meta = _admission_acquire(
        alias,
        request_id=dash_rid,
        stream=bool(data.get("stream") is True),
        queue_controls=queue_controls,
    )
    if not ok_ad:
        payload, hint_headers = _admission_retry_hint(alias, ad_meta)
        return Response(
            json.dumps(payload),
            status=int(payload.get("status") or 429),
            mimetype="application/json",
            headers=hint_headers,
        )
    queue_wait_ms = int(ad_meta.get("queue_wait_ms") or 0)

    def _admission_err_response(msg: str, code: int = 400):
        _admission_release(alias)
        return Response(json.dumps({"error": msg}), status=code, mimetype="application/json")

    handle = mlx_manager.load_model(alias)
    if handle is None:
        load_err = mlx_manager.get_last_load_error(alias)
        if load_err:
            return _admission_err_response(load_err, 503)
        return _admission_err_response(f"Could not load MLX model '{alias}'", 503)

    # Pin the alias for the duration of this request (and the streaming
    # generator's lifetime if streaming). Released in the streaming
    # generator's finally OR the non-streaming finally below — never
    # both. We pin AFTER load_model so the load itself can still
    # trigger LRU eviction of a different alias if needed.
    mlx_manager.pin_alias(alias)
    pin_released = False

    def _release_pin_once():
        nonlocal pin_released
        if not pin_released:
            pin_released = True
            mlx_manager.release_pin(alias)

    def _pin_err_response(msg: str, code: int = 400):
        _release_pin_once()
        return _admission_err_response(msg, code)

    model, tokenizer, gen_lock = handle
    profile = get_model_profile(alias)
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    temperature = _safe_float(data.get("temperature"))
    top_p = _safe_float(data.get("top_p"))
    temperature, top_p = _apply_sampler_defaults(temperature, top_p, profile)

    msgs, formatted, prompt_tok, trim_err = _trim_messages_for_context(
        tokenizer,
        messages,
        data.get("prompt") or "",
        int(profile.get("context_window") or 8192),
        _clamp_max_tokens(data.get("max_tokens"), profile),
        MLX_CONTEXT_TRIM_BUFFER,
    )
    if trim_err:
        return _pin_err_response(trim_err, 400)
    if not formatted:
        return _pin_err_response("empty prompt", 400)

    max_tokens = _clamp_max_tokens(data.get("max_tokens"), profile)
    cw = int(profile.get("context_window") or 8192)
    max_tokens, _was_clamped, impossible = _budget_max_tokens(
        prompt_tok, max_tokens, cw, MLX_CONTEXT_TRIM_BUFFER,
    )
    if impossible:
        return _pin_err_response(
            f"No completion budget left (prompt_tokens≈{prompt_tok}, context_window={cw}). "
            "Shorten the prompt or choose a larger-context model.",
            400,
        )

    response_id = f"chatcmpl_{uuid.uuid4().hex}"

    # ---- Streaming path ---------------------------------------------------
    if data.get("stream") is True:
        req_str = str(requested) if requested is not None else None

        def stream_generator():
            done_sent = False
            created = int(time.time())
            acc: list[str] = []
            if _mlx_dash is not None:
                _mlx_dash.metrics_store.active_enter(alias)
            t_stream = time.perf_counter()

            def _err_chunk(exc: BaseException):
                return {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": alias,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
                    "error": str(exc),
                }

            def _record_stream(status: str, err_msg: str | None, comp_text: str):
                if _mlx_dash is None:
                    return
                elapsed_ms = max(1, int((time.perf_counter() - t_stream) * 1000))
                ct = _count_tokens(tokenizer, comp_text) if comp_text else 0
                pt = _count_tokens(tokenizer, formatted)
                prev = _mlx_dash.build_preview(
                    msgs,
                    formatted_prompt=formatted if getattr(_mlx_dash, "MLX_DASHBOARD_CAPTURE_PROMPTS", False) else None,
                )
                _mlx_dash.record_event(
                    request_id=dash_rid,
                    parent_request_id=None,
                    agent_slot=None,
                    route_kind="chat_mlx_stream",
                    requested_model=req_str,
                    resolved_model=str(alias),
                    backend="mlx",
                    stream=True,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    latency_ms=elapsed_ms,
                    status=status,
                    error_message=err_msg,
                    preview=prev,
                )

            try:
                try:
                    with gen_lock:
                        # Initial role chunk so most clients render immediately.
                        first = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(first)}\n\n"
                        try:
                            for piece in _mlx_stream_text(model, tokenizer, formatted, max_tokens, temperature, top_p):
                                if piece:
                                    acc.append(piece)
                                chunk = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": alias,
                                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        except Exception as e:
                            guided_err = _mlx_error_with_guidance(e, "MLX generation error")
                            yield f"data: {json.dumps(_err_chunk(guided_err))}\n\n"
                            yield "data: [DONE]\n\n"
                            done_sent = True
                            _record_stream("error", guided_err, "".join(acc))
                            return

                        final = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(final)}\n\n"
                        yield "data: [DONE]\n\n"
                        done_sent = True
                        _record_stream("ok", None, "".join(acc))
                except GeneratorExit:
                    raise
                except BaseException as e:
                    log.exception("mlx chat stream failed")
                    if not done_sent:
                        guided_err = _mlx_error_with_guidance(e, "MLX stream failure")
                        try:
                            yield f"data: {json.dumps(_err_chunk(guided_err))}\n\n"
                            yield "data: [DONE]\n\n"
                        except Exception:
                            log.warning("mlx chat stream could not emit terminal [DONE]", exc_info=True)
                        _record_stream("error", guided_err, "".join(acc))
                    return
                finally:
                    if _mlx_dash is not None:
                        _mlx_dash.metrics_store.active_exit(alias)
            finally:
                _admission_release(alias)
                _release_pin_once()

        headers = {"X-Model-Routed-To": f"mlx/{alias}"}
        if fallback_from:
            headers["X-Model-Resolution"] = f"fallback (requested '{fallback_from}', not present)"
        headers["X-MLX-Queue-Wait-Ms"] = str(queue_wait_ms)
        return Response(stream_with_context(stream_generator()),
                        mimetype="text/event-stream", headers=headers)

    # ---- Non-streaming path ----------------------------------------------
    req_str = str(requested) if requested is not None else None
    if _mlx_dash is not None:
        _mlx_dash.metrics_store.active_enter(alias)
    t0 = time.perf_counter()
    try:
        with gen_lock:
            try:
                text, prompt_tok, comp_tok = _mlx_generate_text_timed(
                    model, tokenizer, formatted, max_tokens, temperature, top_p,
                    timeout_sec=GENERATION_TIMEOUT,
                )
            except TimeoutError as e:
                elapsed = int((time.perf_counter() - t0) * 1000)
                if _mlx_dash is not None:
                    _mlx_dash.record_event(
                        request_id=dash_rid,
                        parent_request_id=None,
                        agent_slot=None,
                        route_kind="chat_mlx",
                        requested_model=req_str,
                        resolved_model=str(alias),
                        backend="mlx",
                        stream=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=max(1, elapsed),
                        status="error",
                        error_message=str(e),
                        preview=_mlx_dash.build_preview(msgs),
                    )
                return Response(json.dumps({"error": str(e)}),
                                status=504, mimetype="application/json")
            except Exception as e:
                elapsed = int((time.perf_counter() - t0) * 1000)
                if _mlx_dash is not None:
                    _mlx_dash.record_event(
                        request_id=dash_rid,
                        parent_request_id=None,
                        agent_slot=None,
                        route_kind="chat_mlx",
                        requested_model=req_str,
                        resolved_model=str(alias),
                        backend="mlx",
                        stream=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=max(1, elapsed),
                        status="error",
                        error_message=str(e),
                        preview=_mlx_dash.build_preview(msgs),
                    )
                return Response(json.dumps({"error": _mlx_error_with_guidance(e, "MLX generation error")}),
                                status=500, mimetype="application/json")
    finally:
        if _mlx_dash is not None:
            _mlx_dash.metrics_store.active_exit(alias)
        _admission_release(alias)
        _release_pin_once()

    elapsed = int((time.perf_counter() - t0) * 1000)

    if _mlx_dash is not None:
        _mlx_dash.record_event(
            request_id=dash_rid,
            parent_request_id=None,
            agent_slot=None,
            route_kind="chat_mlx",
            requested_model=req_str,
            resolved_model=str(alias),
            backend="mlx",
            stream=False,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            latency_ms=max(1, elapsed),
            status="ok",
            error_message=None,
            preview=_mlx_dash.build_preview(msgs),
        )

    body = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text or ""},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": prompt_tok, "completion_tokens": comp_tok, "total_tokens": prompt_tok + comp_tok},
    }
    headers = {
        "X-Model-Routed-To": f"mlx/{alias}",
        "X-MLX-Latency-Ms": str(elapsed),
        "X-MLX-Queue-Wait-Ms": str(queue_wait_ms),
    }
    if forced_default_from is not None:
        headers["X-Model-Resolution"] = (
            f"forced_default (requested '{forced_default_from}', using '{requested}')"
        )
    if fallback_from:
        headers["X-Model-Resolution"] = f"fallback (requested '{fallback_from}', not present)"
    return Response(json.dumps(body), status=200, mimetype="application/json", headers=headers)


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    return _handle_chat_request(request.get_json(silent=True), request.headers)


@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-style text completions endpoint.

    Internally routes through chat inference for one consistent MLX path,
    then adapts the response shape to text_completion.
    """
    data = request.get_json(silent=True) or {}
    if data.get("stream") is True:
        return Response(
            json.dumps({"error": "Streaming not implemented for /v1/completions. Use /v1/chat/completions with stream=true."}),
            status=501,
            mimetype="application/json",
        )
    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(p for p in prompt if isinstance(p, str))
    if not prompt:
        return Response(json.dumps({"error": "prompt required"}),
                        status=400, mimetype="application/json")
    bridged = dict(data)
    bridged["messages"] = [{"role": "user", "content": prompt}]
    bridged.pop("prompt", None)
    chat_resp = _handle_chat_request(bridged, request.headers)

    # Pass through non-JSON / error payloads unchanged.
    if int(chat_resp.status_code) >= 400:
        return chat_resp

    try:
        chat_body = json.loads(chat_resp.get_data(as_text=True) or "{}")
    except Exception:
        return chat_resp

    choices = chat_body.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    first_msg = first_choice.get("message") if isinstance(first_choice, dict) else {}
    text = first_msg.get("content", "") if isinstance(first_msg, dict) else ""
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else "stop"
    usage = chat_body.get("usage")
    model = chat_body.get("model")

    body = {
        "id": chat_body.get("id", f"cmpl_{uuid.uuid4().hex}"),
        "object": "text_completion",
        "created": int(chat_body.get("created") or time.time()),
        "model": model,
        "choices": [{
            "text": text or "",
            "index": 0,
            "logprobs": None,
            "finish_reason": finish_reason or "stop",
        }],
    }
    if isinstance(usage, dict):
        body["usage"] = usage

    headers = {}
    for key in ("X-Model-Routed-To", "X-MLX-Latency-Ms", "X-MLX-Queue-Wait-Ms", "X-Model-Resolution"):
        val = chat_resp.headers.get(key)
        if val is not None:
            headers[key] = val

    return Response(json.dumps(body), status=200, mimetype="application/json", headers=headers)


# =============================================================================
# SWARM ROUTES
# =============================================================================


@app.route("/swarm/models", methods=["GET"])
def swarm_models():
    if _GRAB is not None:
        return Response(
            json.dumps({
                "mode": "grab",
                "models": [_GRAB[4]],
                "path": _GRAB[3],
                "swarm": "disabled",
            }),
            status=200,
            mimetype="application/json",
        )

    aliases = mlx_manager.get_available_aliases()
    return Response(
        json.dumps({
            "models": aliases,
            "loaded": mlx_manager.get_loaded_aliases(),
            "roles": MODEL_ROLES,
            "default_model": DEFAULT_MODEL or None,
            "max_parallel": MAX_PARALLEL_MODEL_CALLS,
            "max_concurrent_models": MAX_CONCURRENT_MODELS,
            "anthropic_available": bool(ANTHROPIC_API_KEY),
            "anthropic_model": ANTHROPIC_MODEL if ANTHROPIC_API_KEY else None,
            "swarm_chat_enabled": bool(SWARM_CHAT_ENABLED),
            "swarm_chat_default_models": SWARM_CHAT_DEFAULT_MODELS,
            "swarm_chat_default_strategy": SWARM_CHAT_DEFAULT_STRATEGY,
            "swarm_chat_auto_tokens_loaded": sorted(_SWARM_AUTO_LOADED_TOKENS),
            "swarm_chat_auto_tokens_available": sorted(_SWARM_AUTO_AVAILABLE_TOKENS),
        }),
        status=200, mimetype="application/json",
    )


@app.route("/swarm/fanout", methods=["POST"])
def swarm_fanout():
    """Broadcast one prompt to N models in parallel.

    Body:
      {"models": ["role:coder", "qwen2.5-7b", "anthropic"],
       "messages": [...], "max_tokens": 512, "max_parallel": 3}
    """
    if _GRAB is not None:
        return _swarm_disabled_in_grab()

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
            json.dumps({"error": "no MLX models loaded or configured; pass explicit models"}),
            status=503, mimetype="application/json",
        )

    common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p")}
    common = {k: v for k, v in common.items() if v is not None}

    swarm_parent_id = str(uuid.uuid4())
    results, err = _fanout(
        models, messages, common, max_parallel=data.get("max_parallel"),
        parent_request_id=swarm_parent_id, route_kind="swarm_fanout",
    )
    if err:
        return Response(json.dumps({"error": err}),
                        status=503, mimetype="application/json")

    return Response(
        json.dumps({
            "id": f"swarm_{uuid.uuid4().hex}",
            "object": "swarm.fanout",
            "created": int(time.time()),
            "responses": results,
        }),
        status=200, mimetype="application/json",
        headers={"X-Swarm-Models": ",".join((r or {}).get("model", "?") for r in results)},
    )


@app.route("/swarm/vote", methods=["POST"])
def swarm_vote():
    """Fanout + consensus. Returns an OpenAI chat.completion."""
    if _GRAB is not None:
        return _swarm_disabled_in_grab()

    data = request.get_json(silent=True) or {}
    models = data.get("models") or []
    messages = data.get("messages") or []
    strategy = (data.get("strategy") or "best-of-n").lower()
    models_ok = isinstance(models, (list, str)) and models
    if not (models_ok and isinstance(messages, list) and messages):
        return Response(json.dumps({"error": "models and messages are required"}),
                        status=400, mimetype="application/json")

    models, exp_err = _expand_swarm_models(models)
    if exp_err:
        return Response(json.dumps({"error": exp_err}),
                        status=503, mimetype="application/json")
    if not models:
        return Response(
            json.dumps({"error": "no MLX models loaded or configured; pass explicit models"}),
            status=503, mimetype="application/json",
        )

    common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p") if data.get(k) is not None}
    swarm_parent_id = str(uuid.uuid4())
    candidates, err = _fanout(
        models, messages, common,
        parent_request_id=swarm_parent_id, route_kind="swarm_vote",
    )
    if err:
        return Response(json.dumps({"error": err}),
                        status=503, mimetype="application/json")

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
        avail = mlx_manager.get_available_aliases()
        judge_alias, jerr = resolve_model_alias(judge_request, avail)

        if jerr or not judge_alias:
            winner = max(successes, key=lambda c: len(c.get("text", "")))
            rationale = f"judge unavailable ({jerr or 'no model'}); picked longest"
        else:
            jresp, jerr = _mlx_chat_completion(judge_alias, judge_messages, max_tokens=200, temperature=0.0)
            if _mlx_dash is not None:
                jlat = 0
                if isinstance(jresp, dict):
                    jlat = int((jresp.get("_meta") or {}).get("latency_ms") or 0)
                pt, ct, tt = _mlx_dash.usage_from_openai_response(jresp if isinstance(jresp, dict) else None)
                _mlx_dash.record_event(
                    request_id=f"{swarm_parent_id}_judge",
                    parent_request_id=swarm_parent_id,
                    agent_slot=None,
                    route_kind="swarm_vote_judge",
                    requested_model=str(judge_request),
                    resolved_model=str(judge_alias),
                    backend="mlx",
                    stream=False,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    latency_ms=jlat,
                    status="ok" if not jerr else "error",
                    error_message=str(jerr) if jerr else None,
                    preview=_mlx_dash.build_preview(judge_messages),
                )
            verdict = _extract_text(jresp)
            picked_idx = _parse_judge_verdict(verdict, labels)
            if picked_idx is None:
                winner = max(successes, key=lambda c: len(c.get("text", "")))
                rationale = (f"judge response unparseable; fell back to longest. "
                             f"Verdict: {(verdict or '')[:140]}")
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
    return Response(json.dumps(out), status=200, mimetype="application/json",
                    headers={"X-Swarm-Strategy": strategy, "X-Swarm-Winner": str(winner["model"])})


@app.route("/swarm/pipeline", methods=["POST"])
def swarm_pipeline():
    """Sequential chain. Each step's `system`/`user` template can reference
    {{previous}} or {{step_name}} from earlier steps.
    """
    if _GRAB is not None:
        return _swarm_disabled_in_grab()

    data = request.get_json(silent=True) or {}
    steps = data.get("steps") or []
    messages = data.get("messages") or []
    if not (isinstance(steps, list) and steps and isinstance(messages, list) and messages):
        return Response(json.dumps({"error": "steps and messages are required"}),
                        status=400, mimetype="application/json")

    pipeline_parent_id = str(uuid.uuid4())
    available = mlx_manager.get_available_aliases()
    history = []
    last_text = ""

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        name = step.get("name") or f"step_{idx}"
        ctx = {h["name"]: h["text"] for h in history}
        ctx["previous"] = last_text

        sys_prompt = _substitute_pipeline_template(step.get("system") or "", ctx)
        user_template = step.get("user")

        agent_messages = []
        if sys_prompt:
            agent_messages.append({"role": "system", "content": sys_prompt})
        if user_template:
            agent_messages.append({
                "role": "user",
                "content": _substitute_pipeline_template(user_template, ctx),
            })
        else:
            agent_messages += [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]

        kwargs = {k: step[k] for k in ("max_tokens", "temperature", "top_p")
                  if step.get(k) is not None}

        label, resp, step_err, latency = _run_one_agent(
            {"model": step.get("model")}, agent_messages, kwargs, available
        )
        if step_err or not resp:
            return Response(
                json.dumps({"error": f"step '{name}' failed: {step_err}", "history": history}),
                status=502, mimetype="application/json",
            )
        if _mlx_dash is not None:
            pt, ct, tt = _mlx_dash.usage_from_openai_response(resp if isinstance(resp, dict) else None)
            backend = "anthropic" if str(label).lower().startswith("anthropic") else "mlx"
            _mlx_dash.record_event(
                request_id=f"{pipeline_parent_id}_step{idx}",
                parent_request_id=pipeline_parent_id,
                agent_slot=idx,
                route_kind="swarm_pipeline",
                requested_model=str(step.get("model")),
                resolved_model=str(label),
                backend=backend,
                stream=False,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                latency_ms=latency,
                status="ok",
                error_message=None,
                preview=_mlx_dash.build_preview(agent_messages),
                extra={"step_name": name},
            )
        text = _extract_text(resp)
        history.append({"name": name, "model": label, "text": text, "latency_ms": latency})
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
        status=200, mimetype="application/json",
        headers={
            "X-Swarm-Strategy": "pipeline",
            "X-Swarm-Steps": ",".join(h["name"] for h in history),
        },
    )


@app.route("/swarm/debate", methods=["POST"])
def swarm_debate():
    """Multi-round debate among N models, then a judge synthesizes.

    Body:
      {"models": ["role:reasoner", "role:fast", "role:coder"],
       "messages": [...], "rounds": 2, "judge": "role:reasoner",
       "max_tokens": 512}
    """
    if _GRAB is not None:
        return _swarm_disabled_in_grab()

    data = request.get_json(silent=True) or {}
    models = data.get("models") or ["role:reasoner", "role:fast", "role:coder"]
    messages = data.get("messages") or []
    rounds = data.get("rounds", 2)
    judge_model = data.get("judge", "role:reasoner")

    if not isinstance(models, (list, str)) or not models:
        return Response(json.dumps({"error": "debate requires a non-empty models list (or 'auto')"}),
                        status=400, mimetype="application/json")
    if not isinstance(messages, list) or not messages:
        return Response(json.dumps({"error": "messages (list) is required"}),
                        status=400, mimetype="application/json")

    models, exp_err = _expand_swarm_models(models)
    if exp_err:
        return Response(json.dumps({"error": exp_err}),
                        status=503, mimetype="application/json")
    if len(models) < 2:
        return Response(
            json.dumps({"error": "debate requires at least 2 models after expansion"}),
            status=400, mimetype="application/json",
        )

    common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p") if data.get(k) is not None}
    available = mlx_manager.get_available_aliases()
    transcript = []
    debate_parent_id = str(uuid.uuid4())

    original_user = _extract_user_intent_text({"messages": messages})

    for r in range(max(1, int(rounds))):
        if r == 0:
            round_specs = models
            round_messages = messages
        else:
            context_str = "\n\n".join(
                f"[{t['model']}]: {t['text']}" for t in transcript
            )
            round_specs = [
                {
                    "model": m,
                    "system": (
                        "You are in round {round} of a debate. Here is the transcript so far:\n"
                        "{transcript}\n\n"
                        "Critique the other arguments, refine your own position, and address any flaws."
                    ).format(round=r + 1, transcript=context_str),
                    "messages": [{"role": "user", "content": original_user}],
                }
                for m in models
            ]
            round_messages = [{"role": "user", "content": original_user}]

        results, err = _fanout(
            round_specs, round_messages, common,
            parent_request_id=debate_parent_id,
            route_kind=f"swarm_debate_round_{r + 1}",
        )
        if err:
            return Response(json.dumps({"error": err, "transcript": transcript}),
                            status=502, mimetype="application/json")

        for res in (results or []):
            if res and res.get("ok") and res.get("text"):
                transcript.append({
                    "round": r + 1,
                    "model": res["model"],
                    "text": res["text"],
                    "latency_ms": res.get("latency_ms", 0),
                })

    final_context = "\n\n".join(
        f"[Round {t['round']} - {t['model']}]: {t['text']}" for t in transcript
    )
    judge_messages = [
        {"role": "system", "content": (
            "You are the Chief Synthesizer. Review the following multi-round debate transcript "
            "and produce a single, unified, balanced answer that resolves all conflicts and "
            "incorporates the strongest arguments from each participant."
        )},
        {"role": "user", "content": (
            f"Original question:\n{original_user}\n\n"
            f"Debate transcript:\n{final_context}"
        )},
    ]

    judge_alias, jerr = resolve_model_alias(judge_model, available)
    if jerr or not judge_alias:
        longest = max(transcript, key=lambda t: len(t.get("text", "")), default=None)
        final_text = longest["text"] if longest else ""
        rationale = f"judge unavailable ({jerr}); returned longest response"
    else:
        jresp, jerr = _mlx_chat_completion(
            judge_alias, judge_messages,
            max_tokens=common.get("max_tokens", 1024),
            temperature=0.3,
        )
        if _mlx_dash is not None:
            jlat = 0
            if isinstance(jresp, dict):
                jlat = int((jresp.get("_meta") or {}).get("latency_ms") or 0)
            pt, ct, tt = _mlx_dash.usage_from_openai_response(jresp if isinstance(jresp, dict) else None)
            _mlx_dash.record_event(
                request_id=f"{debate_parent_id}_judge",
                parent_request_id=debate_parent_id,
                agent_slot=None,
                route_kind="swarm_debate_judge",
                requested_model=str(judge_model),
                resolved_model=str(judge_alias),
                backend="mlx",
                stream=False,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                latency_ms=jlat,
                status="ok" if not jerr else "error",
                error_message=str(jerr) if jerr else None,
                preview=_mlx_dash.build_preview(judge_messages),
            )
        final_text = _extract_text(jresp) if jresp else ""
        rationale = f"synthesized by {judge_alias}" if not jerr else f"judge error: {jerr}"
        if jerr:
            longest = max(transcript, key=lambda t: len(t.get("text", "")), default=None)
            final_text = longest["text"] if longest else ""

    return Response(
        json.dumps({
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"swarm/debate/{judge_alias or '?'}",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }],
            "swarm": {
                "strategy": "debate",
                "rounds": rounds,
                "judge": judge_alias or judge_model,
                "rationale": rationale,
                "transcript": transcript,
            },
        }),
        status=200, mimetype="application/json",
        headers={"X-Swarm-Strategy": "debate", "X-Swarm-Rounds": str(rounds)},
    )


# =============================================================================
# DASHBOARD (optional mlx_dashboard module)
# =============================================================================

_mlx_dashboard_registered = False


def _configure_mlx_dashboard(register_blueprint: bool = False) -> None:
    global _mlx_dashboard_registered
    if _mlx_dash is None:
        return
    _mlx_dash.configure(
        mlx_manager=mlx_manager,
        mlx_available=MLX_AVAILABLE,
        grab_mode=lambda: _GRAB is not None,
        get_roles=lambda: dict(MODEL_ROLES),
        get_default_model_env=lambda: DEFAULT_MODEL,
        max_concurrent_models=MAX_CONCURRENT_MODELS,
        max_parallel_model_calls=MAX_PARALLEL_MODEL_CALLS,
        swarm_fanout_timeout=SWARM_FANOUT_TIMEOUT or None,
        swarm_budget_fn=_swarm_fanout_budget_seconds,
        anthropic_enabled=bool(ANTHROPIC_API_KEY),
        admission_snapshot_fn=_admission_scheduler.snapshot,
    )
    if register_blueprint and not _mlx_dashboard_registered:
        _mlx_dash.register(app)
        _mlx_dashboard_registered = True


_configure_mlx_dashboard(register_blueprint=True)


# =============================================================================
# STARTUP
# =============================================================================

def _preload_and_validate(aliases_to_preload: list[str]):
    """Load each requested model and run a tiny generation to verify it works."""
    if not aliases_to_preload:
        return
    log.info("Preloading %d model(s): %s", len(aliases_to_preload), aliases_to_preload)
    for alias in aliases_to_preload:
        resolved, err = resolve_model_alias(alias)
        if err or not resolved:
            log.warning("Preload '%s' skipped: %s", alias, err)
            continue
        # acquire_inference_handle pins the alias during the self-test
        # so a concurrent dashboard load or LRU pressure can't evict
        # the freshly-loaded model out from under the generate() call.
        with mlx_manager.acquire_inference_handle(resolved) as handle:
            if handle is None:
                load_err = mlx_manager.get_last_load_error(resolved)
                if load_err:
                    log.warning("Preload '%s' failed to load: %s", resolved, load_err)
                else:
                    log.warning("Preload '%s' failed to load", resolved)
                continue
            model, tokenizer, gen_lock = handle
            try:
                with gen_lock:
                    _mlx_generate_text(model, tokenizer, "Hello", 4)
                log.info("Preload '%s' verified OK", resolved)
            except Exception as e:
                log.warning("Preload '%s' loaded but self-test failed: %s", resolved, e)


def _download_model(repo_id: str) -> int:
    """Download a Hugging Face model and exit."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed. pip install huggingface_hub")
        return 1

    cache_root = os.path.expanduser(
        os.environ.get("MLX_GRAB_CACHE", os.path.join("~", ".cache", "mlx-middle-layer-grab"))
    )
    local_name = repo_id.replace("/", "__")
    dest = os.path.join(cache_root, local_name)
    os.makedirs(dest, exist_ok=True)
    log.info("Downloading %r -> %s", repo_id, dest)
    snapshot_download(repo_id=repo_id, local_dir=dest, local_dir_use_symlinks=False)
    if not os.path.isfile(os.path.join(dest, "config.json")):
        log.error("Download finished but config.json missing under %s", dest)
        return 1
    log.info("Download complete: %s", dest)
    log.info("Serve with: python middle_layerMLX.py serve --grab %s", dest)
    return 0


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="middle_layerMLX",
        description="MLX-native OpenAI-compatible gateway with swarm intelligence.",
    )
    sub = p.add_subparsers(dest="command")

    # --- serve ---
    s = sub.add_parser("serve", help="Start the API server (default if no subcommand)")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5001")))
    s.add_argument(
        "--grab",
        metavar="MODEL",
        help="Single-model grab mode: local path or hf:org/model repo id",
    )
    s.add_argument(
        "--display-name",
        default=os.environ.get("MLX_GRAB_DISPLAY_NAME", "mlx"),
        help="Model id returned to clients in grab mode (default: mlx)",
    )
    s.add_argument(
        "--no-grab",
        action="store_true",
        help="Ignore MLX_GRAB_MODEL env for this run (force multi-model mode).",
    )
    s.add_argument(
        "--model-root",
        default=None,
        help="Override MLX_MODEL_ROOT for multi-model discovery",
    )
    s.add_argument(
        "--preload",
        default=None,
        help="Comma-separated model aliases to preload at startup",
    )
    s.add_argument(
        "--no-pick-model",
        action="store_true",
        help="Skip interactive default-model prompt on TTY (non-grab mode)",
    )

    # --- download ---
    d = sub.add_parser("download", help="Download a Hugging Face model and exit")
    d.add_argument("repo", help="Hugging Face repo id, e.g. mlx-community/Qwen3-8B-MLX")

    return p


def _validate_boot_knobs() -> None:
    """Fail fast on invalid knob values before we start touching MLX.

    The historical defaults silently produced runtime crashes (e.g.
    MAX_CONCURRENT_MODELS=0 → KeyError on the first load because
    ``OrderedDict.popitem(last=False)`` was called on an empty dict
    inside ``_ensure_capacity_locked``). Surfacing those at startup
    with a clear, actionable message is strictly better than waiting
    for the first chat request to crash with a confusing traceback.
    """
    if MAX_CONCURRENT_MODELS < 1:
        raise SystemExit(
            f"MAX_CONCURRENT_MODELS={MAX_CONCURRENT_MODELS} is invalid: "
            "must be >= 1 (the LRU eviction loop requires at least one "
            "resident slot). Set MAX_CONCURRENT_MODELS to 1 or higher "
            "and restart."
        )
    if MAX_PARALLEL_MODEL_CALLS < 1:
        raise SystemExit(
            f"MAX_PARALLEL_MODEL_CALLS={MAX_PARALLEL_MODEL_CALLS} is invalid: "
            "must be >= 1 (swarm fanout requires at least one worker). "
            "Set MAX_PARALLEL_MODEL_CALLS to 1 or higher and restart."
        )
    if MLX_PER_MODEL_INFLIGHT_CAP < 0:
        raise SystemExit(
            f"MLX_PER_MODEL_INFLIGHT_CAP={MLX_PER_MODEL_INFLIGHT_CAP} is "
            "invalid: must be >= 0 (0 disables admission, 1+ caps "
            "concurrent inferences per alias). Set to 0 to disable "
            "admission or 1+ to enable and restart."
        )


def main():
    global MLX_MODEL_ROOT, mlx_manager, DEFAULT_MODEL

    _validate_boot_knobs()

    parser = _build_cli()
    args = parser.parse_args()

    # Default to serve if no subcommand given
    if args.command is None:
        args.command = "serve"
        args.host = os.environ.get("HOST", "127.0.0.1")
        args.port = int(os.environ.get("PORT", "5001"))
        args.grab = None
        args.display_name = os.environ.get("MLX_GRAB_DISPLAY_NAME", "mlx")
        args.no_grab = False
        args.model_root = None
        args.preload = None
        args.no_pick_model = False

    log.info("=" * 60)
    log.info("middle_layerMLX - Direct MLX inference + multi-agent swarm")
    log.info("=" * 60)

    # --- download subcommand ---
    if args.command == "download":
        sys.exit(_download_model(args.repo))

    # --- serve ---
    if args.grab:
        os.environ["MLX_GRAB_MODEL"] = args.grab
        os.environ["MLX_GRAB_DISPLAY_NAME"] = args.display_name
    elif args.no_grab:
        os.environ.pop("MLX_GRAB_MODEL", None)
        os.environ.pop("MLX_GRAB_DISPLAY_NAME", None)

    chosen_root = args.model_root
    if not os.environ.get("MLX_GRAB_MODEL", "").strip() and not chosen_root:
        chosen_root = _maybe_interactive_startup_model_root_pick(
            MLX_MODEL_ROOT,
            no_pick=args.no_pick_model,
        )

    if chosen_root:
        MLX_MODEL_ROOT = os.path.abspath(os.path.expanduser(chosen_root))
        os.environ["MLX_MODEL_ROOT"] = MLX_MODEL_ROOT
        mlx_manager = MLXManager(MLX_MODEL_ROOT)

    if not MLX_AVAILABLE:
        log.warning("mlx_lm is not installed. /v1/* and /swarm/* MLX paths will 503.")
        log.warning("Install with: pip install mlx-lm")

    grab_err = init_mlx_grab_model()
    if grab_err:
        log.error("MLX_GRAB_MODEL init failed: %s", grab_err)
        sys.exit(1)

    aliases = mlx_manager.get_available_aliases()
    if _GRAB is not None:
        log.info("Grab mode — serving %r from: %s", _GRAB[4], _GRAB[3])
    else:
        log.info("MLX models discovered: %d", len(aliases))
        for a in aliases[:20]:
            log.info("  - %s", a)
        if len(aliases) > 20:
            log.info("  ... and %d more", len(aliases) - 20)
        if DEFAULT_MODEL and _match_one(DEFAULT_MODEL, aliases) is None:
            log.warning(
                "DEFAULT_MODEL '%s' not found in current model root; clearing stale default.",
                DEFAULT_MODEL,
            )
            DEFAULT_MODEL = ""
            os.environ.pop("DEFAULT_MODEL", None)
            if _mlx_dash is not None:
                try:
                    _mlx_dash.metrics_store.set_preferences("", None)
                except Exception:
                    log.warning("Could not clear runtime dashboard default model", exc_info=True)

    if _GRAB is None:
        _maybe_interactive_startup_model_pick(aliases, no_pick=args.no_pick_model)

    if _GRAB is None and MODEL_ROLES:
        log.info("Roles:")
        for role, prefs in MODEL_ROLES.items():
            log.info("  - %s: %s", role, prefs)
    if _GRAB is None and DEFAULT_MODEL:
        log.info("Default model preference: %s", DEFAULT_MODEL)
    if _GRAB is None:
        log.info("Model miss policy: %s", ON_MODEL_MISS)
        log.info("Max resident MLX models: %d", MAX_CONCURRENT_MODELS)
        log.info("Max parallel swarm calls: %d", MAX_PARALLEL_MODEL_CALLS)
        log.info("Swarm chat routing: %s", "enabled" if SWARM_CHAT_ENABLED else "disabled")
        log.info("Swarm chat strategy: %s", SWARM_CHAT_DEFAULT_STRATEGY)
        log.info("Swarm chat models: %s", SWARM_CHAT_DEFAULT_MODELS)
        log.info("Max tokens ceiling: %d", MAX_TOKENS_CEILING)
    else:
        log.info("Resolver / LRU / swarm: bypassed (grab mode)")

    if MIDDLE_LAYER_API_KEY:
        log.info("Auth: enabled (X-API-Key or Bearer required)")
    else:
        log.info("Auth: disabled (set MIDDLE_LAYER_API_KEY to enable)")

    if _GRAB is not None:
        log.info("Anthropic: disabled (grab mode)")
    elif ANTHROPIC_API_KEY:
        log.info("Anthropic escalation enabled: %s", ANTHROPIC_MODEL)
    else:
        log.info("Anthropic escalation disabled (set ANTHROPIC_API_KEY to enable)")

    if _GRAB is None:
        preload_list = PRELOAD_MODELS
        if args.preload:
            preload_list = [s.strip() for s in args.preload.split(",") if s.strip()]
        _preload_and_validate(preload_list)

    host = args.host
    port = args.port

    try:
        _enforce_safe_bind(host, MIDDLE_LAYER_API_KEY)
    except _PublicBindWithoutAuthError as exc:
        log.error("%s", exc)
        sys.exit(2)

    _configure_mlx_dashboard(register_blueprint=False)
    if _mlx_dash is not None:
        log.info("Dashboard: http://%s:%d/dashboard/ (set MLX_DASHBOARD_ENABLED=0 to disable)", host, port)
    log.info("Listening on %s:%d", host, port)
    log.info("Max request body: %d bytes", app.config["MAX_CONTENT_LENGTH"])
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
