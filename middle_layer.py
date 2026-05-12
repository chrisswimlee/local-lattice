import os
import json
import time
import requests
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

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

# Cache of every loaded LM Studio model id (used by the resolver and swarm).
_cached_model_ids = None
_cached_model_ids_ts = 0
MODEL_LIST_TTL = int(os.environ.get("MODEL_LIST_TTL", "30"))

# Tokens that mean "you pick a model for me". Includes well-known cloud ids
# that often arrive from clients like Continue/Cline regardless of what is
# actually loaded locally.
PLACEHOLDER_MODELS = {
    "", "auto", "default",
    # OpenClaw provider model ids (any of these means "you pick").
    "middlelayer", "middle-layer", "middle_layer",
    "mlxmiddlelayer", "mlx-middle-layer", "mlx_middle_layer", "mlx",
    "lmstudio", "openclaw",
    # Common cloud ids that arrive from clients but aren't loaded locally.
    "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4.1",
    "claude-3-5-sonnet", "claude-3-opus",
}

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
DEFAULT_MODEL_ROLES = {
    "coder":    ["coder", "code"],
    "reasoner": ["72b", "70b", "qwen2.5", "llama-3.3", "deepseek-r1"],
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
            print(f"WARN: MODEL_ROLES_JSON is not valid JSON: {e}")
    path = os.environ.get("MODEL_ROLES_FILE")
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARN: cannot load MODEL_ROLES_FILE={path}: {e}")
    return dict(DEFAULT_MODEL_ROLES)


MODEL_ROLES = _load_model_roles()

# Swarm concurrency knobs. A typical Mac can run two reasonably-sized models
# in parallel; one big + one small is the safe default.
MAX_PARALLEL_MODEL_CALLS = int(os.environ.get("MAX_PARALLEL_MODEL_CALLS", "2"))
SWARM_PER_CALL_TIMEOUT = int(os.environ.get("SWARM_PER_CALL_TIMEOUT", "180"))
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
# requests a swarm meta-model (swarmCouncil / swarm/vote / etc). The swarm itself
# is inherently batch — we run it normally and slice the winner's text back as
# OpenAI chat.completion.chunk frames so streaming-only clients don't see 501.
SWARM_STREAM_CHUNK_CHARS = max(1, int(os.environ.get("SWARM_STREAM_CHUNK_CHARS", "64")))


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
    global _cached_model_ids, _cached_model_ids_ts

    now = time.time()
    if (
        not force_refresh
        and _cached_model_ids is not None
        and (now - _cached_model_ids_ts) < MODEL_LIST_TTL
    ):
        return list(_cached_model_ids), None

    try:
        response = requests.get(LM_STUDIO_MODELS_ENDPOINT, timeout=5)
        if response.status_code != 200:
            return [], f"LM Studio models endpoint returned {response.status_code}"

        data = response.json()
        ids = []

        # Shape A: OpenAI-like {"data":[{"id":"..."}]}
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            for entry in data["data"]:
                if isinstance(entry, dict) and entry.get("id"):
                    ids.append(entry["id"])

        # Shape B: {"models":[{"loaded_instances":[{"id":"..."}], "id": "..."}]}
        if not ids and isinstance(data, dict) and isinstance(data.get("models"), list):
            for entry in data["models"]:
                if not isinstance(entry, dict):
                    continue
                loaded = entry.get("loaded_instances", [])
                if isinstance(loaded, list) and loaded:
                    for inst in loaded:
                        if isinstance(inst, dict) and inst.get("id"):
                            ids.append(inst["id"])
                elif entry.get("id"):
                    ids.append(entry["id"])

        # De-dupe while preserving order.
        seen = set()
        deduped = []
        for mid in ids:
            if mid not in seen:
                seen.add(mid)
                deduped.append(mid)

        _cached_model_ids = deduped
        _cached_model_ids_ts = now
        return list(deduped), None

    except requests.exceptions.ConnectionError:
        return [], "Cannot connect to LM Studio. Is it running?"
    except requests.exceptions.Timeout:
        return [], "Timeout connecting to LM Studio."
    except Exception as e:
        return [], f"Error discovering models: {str(e)}"


def get_current_lmstudio_model():
    """
    Backwards-compatible single-model accessor. Returns (model_id, error)
    where model_id is the first currently loaded LM Studio model.
    Prefer get_lmstudio_model_ids() / resolve_model_id() for new code.
    """
    ids, err = get_lmstudio_model_ids()
    if err:
        return None, err
    if not ids:
        return None, "No model is loaded in LM Studio."
    return ids[0], None


def _is_placeholder(name) -> bool:
    """True when `name` is empty / a generic placeholder / a known cloud id."""
    if name is None:
        return True
    if not isinstance(name, str):
        return True
    return name.strip().lower() in PLACEHOLDER_MODELS


def _match_one(needle: str, haystack):
    """First id in `haystack` matching `needle` (exact then substring, case-insensitive)."""
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
    """First available model id whose name matches any preference for `role`."""
    prefs = MODEL_ROLES.get(role.lower(), [])
    if isinstance(prefs, str):
        prefs = [prefs]
    for p in prefs:
        m = _match_one(p, available)
        if m:
            return m
    return None


def resolve_model_id(requested, available=None):
    """
    Decide which loaded LM Studio model id to use for a request.

    Accepted shapes for `requested`:
      None / "" / "auto" / "default" / "middleLayer"  -> auto-pick
      "exact-model-id"                                -> exact, else substring
      "a,b,c"                                         -> priority list (first match wins)
      "role:coder"                                    -> registry lookup
      "*coder*" / "qwen*"                             -> wildcard substring
      mix any of the above in a comma-separated list, e.g. "role:coder,qwen*"

    Returns (model_id, error_message). On a soft miss (specific name asked but
    not loaded), error is non-None; the caller decides whether to fall back.
    """
    if available is None:
        available, err = get_lmstudio_model_ids()
        if err:
            return None, err
    if not available:
        return None, "No model is loaded in LM Studio."

    if _is_placeholder(requested):
        if DEFAULT_MODEL:
            m = _match_one(DEFAULT_MODEL, available)
            if m:
                return m, None
        # Try the "default" role next, then first available.
        m = _resolve_role("default", available)
        if m:
            return m, None
        return available[0], None

    candidates = [c.strip() for c in str(requested).split(",") if c.strip()]
    for cand in candidates:
        cand_lc = cand.lower()
        if cand_lc.startswith("role:"):
            m = _resolve_role(cand_lc.split(":", 1)[1], available)
            if m:
                return m, None
            continue
        if "*" in cand:
            m = _match_one(cand.replace("*", ""), available)
            if m:
                return m, None
            continue
        m = _match_one(cand, available)
        if m:
            return m, None

    return None, f"No loaded LM Studio model matched '{requested}'. Available: {available}"


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
    status = 200 if ids and not err else 503
    return Response(
        json.dumps(
            {
                "ok": status == 200,
                "lmstudio_model": ids[0] if ids else None,
                "lmstudio_models": ids,
                "lmstudio_error": err,
                "model_roles": MODEL_ROLES,
                "default_model": DEFAULT_MODEL or None,
                "max_parallel": MAX_PARALLEL_MODEL_CALLS,
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
            }
        ),
        status=status,
        mimetype="application/json",
    )

@app.before_request
def _auth_guard():
    if not MIDDLE_LAYER_API_KEY:
        return None
    x_api_key = request.headers.get("X-API-Key")
    authz = request.headers.get("Authorization", "")
    bearer = authz[len("Bearer "):] if authz.startswith("Bearer ") else None

    if x_api_key != MIDDLE_LAYER_API_KEY and bearer != MIDDLE_LAYER_API_KEY:
        return Response(
            json.dumps({"error": "Unauthorized"}),
            status=401,
            mimetype="application/json",
        )
    return None


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

            if (
                endpoint == "chat/completions"
                and SWARM_CHAT_ENABLED
                and _is_swarm_chat_model(requested)
            ):
                wants_stream = json_data.get("stream") is True
                swarm_resp, swarm_err = _run_swarm_chat_completion(requested, json_data)
                if swarm_err or not swarm_resp:
                    return Response(
                        json.dumps({"error": f"Swarm routing failed: {swarm_err}"}),
                        status=502,
                        mimetype="application/json",
                    )
                if wants_stream:
                    return _swarm_body_to_sse_response(swarm_resp)
                return Response(
                    json.dumps(swarm_resp),
                    status=200,
                    mimetype="application/json",
                    headers={"X-Model-Routed-To": str(swarm_resp.get("model", "swarm/unknown"))},
                )
            
            # Resolve the requested model against what is actually loaded.
            model_id, error = resolve_model_id(requested)
            fallback_from = None

            if error or not model_id:
                # If the caller asked for a specific model that is not loaded,
                # either fall back (default) or surface the error.
                if not _is_placeholder(requested) and ON_MODEL_MISS == "fallback":
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
    payload = {"model": model_id, "messages": messages, "stream": False}
    for k in ("max_tokens", "temperature", "top_p", "stop"):
        if kwargs.get(k) is not None:
            payload[k] = kwargs[k]

    try:
        r = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
            timeout=kwargs.get("timeout", SWARM_PER_CALL_TIMEOUT),
        )
        if r.status_code >= 400:
            return None, f"LM Studio {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except requests.exceptions.Timeout:
        return None, "Timeout calling LM Studio"
    except Exception as e:
        return None, f"Error calling LM Studio: {e}"


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


def _extract_text(openai_response) -> str:
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


def _normalize_agent_spec(spec):
    """Accept a string or dict and return a normalized agent dict."""
    if isinstance(spec, str):
        return {"model": spec}
    if isinstance(spec, dict):
        return dict(spec)
    return {"model": str(spec)}


def _run_one_agent(spec, default_messages, default_kwargs, available):
    """Run one agent. Returns (resolved_model_id_or_label, response, error, latency_ms)."""
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

    # Anthropic participant.
    if requested_str.lower().startswith("anthropic"):
        override = None
        if ":" in requested_str:
            override = requested_str.split(":", 1)[1].strip() or None
        label = f"anthropic/{override or ANTHROPIC_MODEL}"
        t0 = time.time()
        resp, err = _call_anthropic_chat(msgs, model_override=override, **kwargs)
        return label, resp, err, int((time.time() - t0) * 1000)

    # LM Studio participant.
    model_id, err = resolve_model_id(requested, available)
    if err:
        return requested or "?", None, err, 0
    t0 = time.time()
    resp, err = _lmstudio_chat_completion(model_id, msgs, **kwargs)
    return model_id, resp, err, int((time.time() - t0) * 1000)


def _fanout(specs, messages, common_kwargs, max_parallel=None):
    """Run each spec in parallel (bounded). Returns (results_list, error)."""
    if not specs:
        return None, "swarm requires at least one model"

    available, err = get_lmstudio_model_ids()
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

    cap = MAX_PARALLEL_MODEL_CALLS
    if isinstance(max_parallel, int) and max_parallel > 0:
        cap = min(cap, max_parallel)
    results = [None] * len(specs)
    workers = max(1, min(cap, len(specs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_run_one_agent, spec, messages, common_kwargs, available): i
            for i, spec in enumerate(specs)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                model_id, resp, e, latency = fut.result()
            except Exception as e:  # noqa: BLE001
                model_id, resp, latency = "?", None, 0
                e = str(e)
            results[i] = {
                "model": model_id,
                "ok": e is None and resp is not None,
                "error": e,
                "latency_ms": latency,
                "response": resp,
                "text": _extract_text(resp) if resp else "",
            }
    return results, None


def _is_swarm_chat_model(requested_model) -> bool:
    if not isinstance(requested_model, str):
        return False
    name = requested_model.strip().lower()
    return name in {
        "swarmcouncil",
        "swarmvote",
        "swarm/vote",
        "swarm/fanout",
        "swarm/pipeline",
    }


def _run_swarm_chat_completion(requested_model: str, json_data: dict):
    """Execute swarm logic and return an OpenAI-shaped chat completion dict."""
    messages = json_data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None, "messages (list) is required for swarm chat"

    common = {k: json_data.get(k) for k in ("max_tokens", "temperature", "top_p")}
    common = {k: v for k, v in common.items() if v is not None}

    swarm_cfg = json_data.get("swarm") if isinstance(json_data.get("swarm"), dict) else {}
    models = swarm_cfg.get("models") or SWARM_CHAT_DEFAULT_MODELS
    strategy = (swarm_cfg.get("strategy") or SWARM_CHAT_DEFAULT_STRATEGY).lower()
    max_parallel = swarm_cfg.get("max_parallel")

    if not isinstance(models, list) or not models:
        return None, "swarm.models must be a non-empty list"

    candidates, err = _fanout(models, messages, common, max_parallel=max_parallel)
    if err:
        return None, err

    successes = [c for c in candidates if c["ok"] and c.get("text")]
    if not successes:
        errs = "; ".join(c.get("error") or "unknown" for c in candidates)
        return None, f"all swarm agents failed: {errs}"

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
        avail, _ = get_lmstudio_model_ids()
        judge_id, jerr = resolve_model_id(judge_request, avail)

        if jerr or not judge_id:
            winner = max(successes, key=lambda c: len(c.get("text", "")))
            rationale = f"judge unavailable ({jerr or 'no model'}); picked longest"
        else:
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
            "requested_model": requested_model,
        },
    }
    return out, None


def _swarm_body_to_sse_response(body: dict, *, chunk_chars: int = SWARM_STREAM_CHUNK_CHARS):
    """Wrap a non-stream swarm chat.completion as OpenAI SSE chunks.

    Swarm vote/fanout/pipeline are inherently batch (every candidate has to
    finish before the judge votes), so streaming clients get the winner's text
    sliced back into ``chat.completion.chunk`` deltas. Trailing ``data: [DONE]``
    is always emitted so well-behaved consumers don't hang.
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
    """
    data = request.get_json(silent=True) or {}
    models = data.get("models") or []
    messages = data.get("messages") or []
    if not isinstance(models, list) or not models:
        return Response(json.dumps({"error": "models (list) is required"}),
                        status=400, mimetype="application/json")
    if not isinstance(messages, list) or not messages:
        return Response(json.dumps({"error": "messages (list) is required"}),
                        status=400, mimetype="application/json")

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

    if not isinstance(models, list) or not models or not isinstance(messages, list) or not messages:
        return Response(json.dumps({"error": "models and messages are required"}),
                        status=400, mimetype="application/json")

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Starting middle_layer on port {port}...")
    print(f"Listening on host: {host}")
    print(f"LM Studio URL: {LM_STUDIO_URL}")
    if MIDDLE_LAYER_API_KEY:
        print("Auth: enabled (X-API-Key required)")
    else:
        print("Auth: disabled (set MIDDLE_LAYER_API_KEY to enable)")

    print(f"Model miss policy: {ON_MODEL_MISS}")
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

    # Test connection to LM Studio on startup
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
