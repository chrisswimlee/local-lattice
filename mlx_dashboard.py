"""
Runtime dashboard for middle_layerMLX: bounded in-memory metrics, JSON API,
and static UI. Does not expose chain-of-thought; only routing and I/O metadata.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any

from flask import Blueprint, Response, abort, redirect, request, send_from_directory

from middle_layer.security import alias_in_allowlist as _alias_in_allowlist
from middle_layer.security import check_api_key as _check_api_key
from middle_layer.security import is_well_formed_alias as _is_well_formed_alias

log = logging.getLogger("mlx_dashboard")

MLX_DASHBOARD_ENABLED = os.environ.get("MLX_DASHBOARD_ENABLED", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
MLX_DASHBOARD_PREVIEW_CHARS = int(os.environ.get("MLX_DASHBOARD_PREVIEW_CHARS", "200"))
MLX_DASHBOARD_MAX_EVENTS = int(os.environ.get("MLX_DASHBOARD_MAX_EVENTS", "200"))
MLX_DASHBOARD_MAX_ERROR_CHARS = int(os.environ.get("MLX_DASHBOARD_MAX_ERROR_CHARS", "500"))
MLX_DASHBOARD_CAPTURE_PROMPTS = os.environ.get("MLX_DASHBOARD_CAPTURE_PROMPTS", "0").strip().lower() in (
    "1", "true", "yes", "on",
)
MLX_DASHBOARD_MAX_PROMPT_CHARS = int(os.environ.get("MLX_DASHBOARD_MAX_PROMPT_CHARS", "8000"))

_CTX: dict[str, Any] = {}
_dash_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
bp = Blueprint("mlx_dashboard", __name__)


def configure(**kwargs: Any) -> None:
    """Called from middle_layerMLX after core globals exist."""
    _CTX.update(kwargs)


def _auth_ok() -> bool:
    key = os.environ.get("MIDDLE_LAYER_API_KEY")
    return _check_api_key(request.headers, key)


def _require_api_auth() -> Response | None:
    if not _auth_ok():
        return Response(json.dumps({"error": "Unauthorized"}), status=401, mimetype="application/json")
    return None


def _truncate(s: str, n: int) -> str:
    if not isinstance(s, str) or n <= 0:
        return ""
    s = s[:n]
    return s


def _sanitize_error(msg: str | None) -> str:
    if not msg:
        return ""
    t = str(msg)
    t = re.sub(r"(?i)(sk-[a-z0-9]{20,})", "[redacted]", t)
    t = re.sub(r"(?i)(Bearer\s+)[\w.-]+", r"\1[redacted]", t)
    t = re.sub(r"(?i)(x-api-key[:\s]+)[^\s]+", r"\1[redacted]", t)
    return _truncate(t, MLX_DASHBOARD_MAX_ERROR_CHARS)


def _last_user_text(messages: list | None) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            return "\n".join(parts)
    return ""


def _role_counts(messages: list | None) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    if not isinstance(messages, list):
        return dict(out)
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("role"), str):
            out[m["role"]] += 1
    return dict(out)


def build_preview(messages: list | None, formatted_prompt: str | None = None) -> dict[str, Any]:
    """Privacy-first preview fields for dashboard events."""
    last_user = _last_user_text(messages)
    base: dict[str, Any] = {
        "role_counts": _role_counts(messages),
        "last_user_chars": len(last_user),
    }
    if MLX_DASHBOARD_CAPTURE_PROMPTS:
        cap = MLX_DASHBOARD_MAX_PROMPT_CHARS
        if formatted_prompt:
            base["formatted_prompt_excerpt"] = _truncate(formatted_prompt, cap)
        base["last_user_excerpt"] = _truncate(last_user, cap)
    else:
        base["last_user_preview"] = _truncate(last_user, MLX_DASHBOARD_PREVIEW_CHARS)
    return base


def usage_from_openai_response(resp: dict | None) -> tuple[int, int, int]:
    if not isinstance(resp, dict):
        return 0, 0, 0
    u = resp.get("usage")
    if not isinstance(u, dict):
        return 0, 0, 0
    pt = int(u.get("prompt_tokens") or 0)
    ct = int(u.get("completion_tokens") or 0)
    tt = int(u.get("total_tokens") or (pt + ct))
    return pt, ct, tt


def _tps(comp_tokens: int, latency_ms: int) -> float | None:
    if latency_ms <= 0 or comp_tokens <= 0:
        return None
    return round(comp_tokens / (latency_ms / 1000.0), 3)


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=MLX_DASHBOARD_MAX_EVENTS)
        self._active_by_alias: dict[str, int] = defaultdict(int)
        self._prefs_lock = threading.Lock()
        self._runtime_default_model = ""
        self._swarm_presets: dict[str, list[Any]] = {}

    def get_runtime_default_model(self) -> str:
        with self._prefs_lock:
            return self._runtime_default_model.strip()

    def set_preferences(self, default_model: str | None, swarm_presets: dict | None) -> dict[str, Any]:
        with self._prefs_lock:
            if default_model is not None:
                self._runtime_default_model = (default_model or "").strip()
            if swarm_presets is not None:
                if not isinstance(swarm_presets, dict):
                    raise ValueError("swarm_presets must be an object")
                cleaned: dict[str, list[Any]] = {}
                for k, v in swarm_presets.items():
                    if not isinstance(k, str) or not k.strip():
                        continue
                    if isinstance(v, list):
                        cleaned[k.strip()] = v
                self._swarm_presets = cleaned
            return {
                "default_model": self._runtime_default_model or None,
                "swarm_presets": dict(self._swarm_presets),
            }

    def get_preferences(self) -> dict[str, Any]:
        with self._prefs_lock:
            return {
                "default_model": self._runtime_default_model or None,
                "swarm_presets": dict(self._swarm_presets),
            }

    def append_event(self, event: dict[str, Any]) -> None:
        if not MLX_DASHBOARD_ENABLED:
            return
        ev = dict(event)
        ev["ts"] = time.time()
        with self._lock:
            self._events.append(ev)

    def active_enter(self, alias: str) -> None:
        if not MLX_DASHBOARD_ENABLED:
            return
        with self._lock:
            self._active_by_alias[alias] += 1

    def active_exit(self, alias: str) -> None:
        if not MLX_DASHBOARD_ENABLED:
            return
        with self._lock:
            self._active_by_alias[alias] = max(0, self._active_by_alias[alias] - 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
            active = dict(self._active_by_alias)
        mgr = _CTX.get("mlx_manager")
        available = mgr.get_available_aliases() if mgr else []
        loaded = mgr.get_loaded_aliases() if mgr else []
        mem = mgr.get_memory_stats() if mgr and hasattr(mgr, "get_memory_stats") else {}
        budget_fn = _CTX.get("swarm_budget_fn")
        budget_example = float(budget_fn(3)) if callable(budget_fn) else None
        grab = bool(_CTX.get("grab_mode")()) if callable(_CTX.get("grab_mode")) else False
        roles = _CTX.get("get_roles")
        roles_v = roles() if callable(roles) else {}
        default_env = _CTX.get("get_default_model_env")
        default_env_v = default_env() if callable(default_env) else ""
        admission_snapshot_fn = _CTX.get("admission_snapshot_fn")
        admission = admission_snapshot_fn() if callable(admission_snapshot_fn) else {}
        return {
            "dashboard_enabled": MLX_DASHBOARD_ENABLED,
            "note": "This dashboard shows routing and I/O metadata only, not internal chain-of-thought.",
            "mlx_available": bool(_CTX.get("mlx_available")),
            "grab_mode": grab,
            "mlx_root": getattr(mgr, "root_path", None) if mgr else None,
            "models_available": available,
            "models_loaded": loaded,
            "memory": mem,
            "active_by_alias": active,
            "events": events[-MLX_DASHBOARD_MAX_EVENTS :],
            "preferences": self.get_preferences(),
            "config": {
                "model_roles": roles_v,
                "default_model_env": default_env_v or None,
                "max_concurrent_models": _CTX.get("max_concurrent_models"),
                "max_parallel_model_calls": _CTX.get("max_parallel_model_calls"),
                "swarm_fanout_timeout_sec": _CTX.get("swarm_fanout_timeout"),
                "swarm_fanout_budget_example_sec": budget_example,
                "anthropic_enabled": bool(_CTX.get("anthropic_enabled")),
                "admission": admission,
            },
        }

    def config_public(self) -> dict[str, Any]:
        return {
            "dashboard_enabled": MLX_DASHBOARD_ENABLED,
            "preview_chars": MLX_DASHBOARD_PREVIEW_CHARS,
            "max_events": MLX_DASHBOARD_MAX_EVENTS,
            "capture_prompts": MLX_DASHBOARD_CAPTURE_PROMPTS,
            "max_prompt_chars": MLX_DASHBOARD_MAX_PROMPT_CHARS,
            "note": "Chain-of-thought is not exposed.",
        }


metrics_store = MetricsStore()


def record_event(
    *,
    request_id: str,
    parent_request_id: str | None,
    agent_slot: int | None,
    route_kind: str,
    requested_model: str | None,
    resolved_model: str | None,
    backend: str,
    stream: bool,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: int,
    status: str,
    error_message: str | None = None,
    preview: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    err = _sanitize_error(error_message) if error_message else None
    ev: dict[str, Any] = {
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "agent_slot": agent_slot,
        "route_kind": route_kind,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "backend": backend,
        "stream": stream,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "tokens_per_sec": _tps(completion_tokens, latency_ms),
        "status": status,
        "error": err,
        "preview": preview or {},
    }
    if extra:
        ev["extra"] = extra
    metrics_store.append_event(ev)


@bp.route("/dashboard/api/snapshot", methods=["GET"])
def api_snapshot():
    if not MLX_DASHBOARD_ENABLED:
        return Response(json.dumps({"error": "dashboard disabled"}), status=404, mimetype="application/json")
    denied = _require_api_auth()
    if denied:
        return denied
    return Response(json.dumps(metrics_store.snapshot(), indent=2), mimetype="application/json")


@bp.route("/dashboard/api/config", methods=["GET"])
def api_config():
    if not MLX_DASHBOARD_ENABLED:
        return Response(json.dumps({"error": "dashboard disabled"}), status=404, mimetype="application/json")
    denied = _require_api_auth()
    if denied:
        return denied
    snap = metrics_store.snapshot()
    out = {**metrics_store.config_public(), **(snap.get("config") or {})}
    return Response(json.dumps(out, indent=2), mimetype="application/json")


@bp.route("/dashboard/api/preferences", methods=["POST"])
def api_preferences():
    if not MLX_DASHBOARD_ENABLED:
        return Response(json.dumps({"error": "dashboard disabled"}), status=404, mimetype="application/json")
    denied = _require_api_auth()
    if denied:
        return denied
    data = request.get_json(silent=True) or {}
    try:
        prefs = metrics_store.set_preferences(
            data.get("default_model"),
            data.get("swarm_presets"),
        )
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype="application/json")
    return Response(json.dumps({"ok": True, "preferences": prefs}), mimetype="application/json")


@bp.route("/dashboard/api/models/load", methods=["POST"])
def api_models_load():
    if not MLX_DASHBOARD_ENABLED:
        return Response(json.dumps({"error": "dashboard disabled"}), status=404, mimetype="application/json")
    denied = _require_api_auth()
    if denied:
        return denied
    mgr = _CTX.get("mlx_manager")
    if mgr is None:
        return Response(json.dumps({"error": "mlx_manager not configured"}), status=503, mimetype="application/json")
    data = request.get_json(silent=True) or {}
    raw_alias = data.get("alias") or data.get("model") or ""
    alias = raw_alias.strip() if isinstance(raw_alias, str) else ""
    if not alias:
        return Response(json.dumps({"error": "alias required"}), status=400, mimetype="application/json")
    if not _is_well_formed_alias(alias):
        return Response(
            json.dumps({"error": "alias contains disallowed characters"}),
            status=400,
            mimetype="application/json",
        )
    try:
        available = list(mgr.get_available_aliases())
    except Exception:
        available = []
    if not _alias_in_allowlist(alias, available):
        return Response(
            json.dumps({
                "error": f"alias '{alias}' is not in the discovered model set",
                "hint": "GET /dashboard/api/snapshot lists available aliases",
            }),
            status=400,
            mimetype="application/json",
        )
    h = mgr.load_model(alias)
    if h is None:
        return Response(json.dumps({"error": f"could not load '{alias}'"}), status=400, mimetype="application/json")
    return Response(
        json.dumps({"ok": True, "alias": alias, "resident": mgr.get_loaded_aliases()}),
        mimetype="application/json",
    )


@bp.route("/dashboard", methods=["GET"])
def dashboard_redirect_slash():
    if not MLX_DASHBOARD_ENABLED:
        abort(404)
    return redirect("/dashboard/", code=302)


@bp.route("/dashboard/", methods=["GET"])
def dashboard_index():
    if not MLX_DASHBOARD_ENABLED:
        abort(404)
    if not os.path.isfile(os.path.join(_dash_dir, "index.html")):
        return Response("Dashboard assets missing (index.html).", status=404)
    return send_from_directory(_dash_dir, "index.html")


@bp.route("/dashboard/<path:name>", methods=["GET"])
def dashboard_static(name: str):
    if not MLX_DASHBOARD_ENABLED:
        abort(404)
    if name.startswith("api/"):
        abort(404)
    safe = os.path.normpath(name).lstrip("/")
    if ".." in safe or safe.startswith("/"):
        abort(404)
    full = os.path.join(_dash_dir, safe)
    root = os.path.abspath(_dash_dir)
    if not (full == root or full.startswith(root + os.sep)):
        abort(404)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(_dash_dir, safe)


def register(app) -> None:
    app.register_blueprint(bp)
