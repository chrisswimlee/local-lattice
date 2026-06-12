"""Flask route layer for the LM Studio gateway (Pass 3 extraction).

Registers ``/healthz``, the OpenAI-compatible ``/v1/*`` proxy, and the
``/swarm/*`` endpoints on a Flask app. Handlers receive dependencies through
:class:`LmStudioRouteContext` so the legacy ``middle_layer.py`` module can
keep thin back-compat wrappers that tests monkey-patch.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import requests
from flask import Flask, Response, request

from middle_layer.lmstudio_client import LMStudioClient
from middle_layer.timing import RequestTimer


def _json_response(
    body: dict | list,
    http_status: int,
    headers: dict[str, str] | None = None,
    *,
    timer: RequestTimer | None = None,
    **log_fields: object,
) -> Response:
    hdrs = dict(headers or {})
    if timer is not None:
        hdrs.update(timer.header_dict())
        timer.maybe_log(**log_fields)
    return Response(json.dumps(body), status=http_status, mimetype="application/json", headers=hdrs)


def _attach_timing(
    response: Response,
    timer: RequestTimer,
    **log_fields: object,
) -> Response:
    for key, value in timer.header_dict().items():
        response.headers[key] = value
    timer.maybe_log(**log_fields)
    return response


def filtered_forward_headers() -> dict[str, str]:
    excluded = {"host", "content-length", "connection", "transfer-encoding"}
    return {k: v for k, v in request.headers if k.lower() not in excluded}


def build_flask_response(upstream_resp: requests.Response) -> Response:
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    proxy_headers = [
        (name, value)
        for (name, value) in upstream_resp.headers.items()
        if name.lower() not in excluded_headers
    ]

    def generate():
        for chunk in upstream_resp.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(generate(), status=upstream_resp.status_code, headers=proxy_headers)


@dataclass
class LmStudioRouteContext:
    """Dependency bundle for LM Studio gateway HTTP handlers."""

    lm_studio_url: str
    lmstudio_client: LMStudioClient
    middle_layer_api_key: str | None
    model_roles: dict
    model_roles_source: str | None
    prefer_loaded_models: bool
    strict_loaded_models: bool
    default_model: str
    on_model_miss: str
    anthropic_model: str
    anthropic_enabled: bool
    enable_litellm_prefix_routing: bool
    litellm_for_anthropic: bool
    litellm_import_error: str | None
    swarm_chat_enabled: bool
    max_parallel: int
    per_model_inflight_cap: int
    swarm_chat_default_models: list
    swarm_chat_default_strategy: str
    swarm_auto_tokens: frozenset[str]
    swarm_chat_canonical: str
    swarm_chat_intents: dict[str, tuple[str, bool]]

    check_api_key: Callable[..., bool]
    apply_security_headers: Callable[..., None]
    get_model_ids: Callable[..., tuple[list[str], str | None]]
    get_loaded_model_ids: Callable[..., tuple[list[str], str | None]]
    litellm_available: Callable[[], bool]
    should_route_to_anthropic: Callable[[str, dict], bool]
    call_anthropic_chat: Callable[..., tuple[dict | None, str | None]]
    call_litellm_chat: Callable[..., tuple[dict | None, str | None]]
    swarm_chat_intent: Callable[[object], tuple[str | None, str | None]]
    run_swarm_chat_completion: Callable[..., tuple[Any, str | None, dict | None]]
    swarm_body_to_sse_response: Callable[..., Response]
    resolve_model_id: Callable[..., tuple[str | None, str | None]]
    is_placeholder: Callable[[object], bool]
    expand_swarm_models: Callable[..., tuple[list | None, str | None]]
    fanout: Callable[..., tuple[list, str | None]]
    run_one_agent: Callable[..., tuple[str, dict | None, str | None, int]]
    extract_text: Callable[[dict | None], str]
    extract_user_intent: Callable[[dict], str]
    lmstudio_chat_completion: Callable[..., tuple[dict | None, str | None]]
    per_model_semaphore: Callable[[str], Any]


def register_lmstudio_routes(app: Flask, ctx: LmStudioRouteContext) -> None:
    """Attach LM Studio gateway routes and request hooks to ``app``."""

    @app.before_request
    def _auth_guard():
        if not ctx.middle_layer_api_key:
            return None
        if ctx.check_api_key(request.headers, ctx.middle_layer_api_key):
            return None
        return Response(
            json.dumps({"error": "Unauthorized"}),
            status=401,
            mimetype="application/json",
        )

    @app.after_request
    def _security_headers(response):
        ctx.apply_security_headers(response, path=request.path or "")
        return response

    @app.route("/healthz", methods=["GET"])
    def healthz():
        ids, err = ctx.get_model_ids(force_refresh=True)
        loaded, lerr = ctx.get_loaded_model_ids(force_refresh=True)
        status = 200 if ids and not err else 503
        return Response(
            json.dumps(
                {
                    "ok": status == 200,
                    "lmstudio_model": ids[0] if ids else None,
                    "lmstudio_models": ids,
                    "lmstudio_loaded_models": loaded,
                    "lmstudio_loaded_endpoint_supported": ctx.lmstudio_client.loaded_endpoint_supported,
                    "lmstudio_loaded_error": lerr,
                    "lmstudio_error": err,
                    "model_roles": ctx.model_roles,
                    "model_roles_source": ctx.model_roles_source,
                    "prefer_loaded_models": bool(ctx.prefer_loaded_models),
                    "strict_loaded_models": bool(ctx.strict_loaded_models),
                    "default_model": ctx.default_model or None,
                    "max_parallel": ctx.max_parallel,
                    "per_model_inflight_cap": ctx.per_model_inflight_cap,
                    "on_model_miss": ctx.on_model_miss,
                    "anthropic_enabled": bool(ctx.anthropic_enabled),
                    "anthropic_model": ctx.anthropic_model,
                    "litellm_available": ctx.litellm_available(),
                    "litellm_for_anthropic": bool(ctx.litellm_for_anthropic),
                    "litellm_prefix_routing": bool(ctx.enable_litellm_prefix_routing),
                    "litellm_import_error": None if ctx.litellm_available() else ctx.litellm_import_error,
                    "swarm_chat_enabled": bool(ctx.swarm_chat_enabled),
                    "swarm_chat_default_models": ctx.swarm_chat_default_models,
                    "swarm_chat_default_strategy": ctx.swarm_chat_default_strategy,
                    "swarm_chat_auto_tokens": sorted(ctx.swarm_auto_tokens),
                    "swarm_chat_canonical": ctx.swarm_chat_canonical,
                    "swarm_chat_aliases": {
                        name: {"intent": intent, "deprecated": deprecated}
                        for name, (intent, deprecated) in sorted(ctx.swarm_chat_intents.items())
                    },
                }
            ),
            status=status,
            mimetype="application/json",
        )

    @app.route("/v1/<path:endpoint>", methods=["POST", "GET"])
    def proxy(endpoint):
        if request.method == "GET":
            resp = requests.request(
                method=request.method,
                url=f"{ctx.lm_studio_url}/v1/{endpoint}",
                headers=filtered_forward_headers(),
                data=None,
                cookies=request.cookies,
                allow_redirects=False,
                stream=True,
            )
            return build_flask_response(resp)

        headers = filtered_forward_headers()
        data = request.get_data()

        if request.is_json:
            try:
                json_data = json.loads(data)
                timer: RequestTimer | None = None
                if endpoint == "chat/completions":
                    timer = RequestTimer.start()

                if ctx.should_route_to_anthropic(endpoint, json_data):
                    if json_data.get("stream") is True:
                        return Response(
                            json.dumps(
                                {
                                    "error": (
                                        "Streaming via Anthropic routing is not enabled "
                                        "in middle_layer.py yet. Set stream=false or route locally."
                                    )
                                }
                            ),
                            status=501,
                            mimetype="application/json",
                        )

                    llm_resp, llm_err = ctx.call_anthropic_chat(
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

                    return Response(
                        json.dumps(llm_resp),
                        status=200,
                        mimetype="application/json",
                        headers={"X-Model-Routed-To": f"anthropic/{ctx.anthropic_model}"},
                    )

                requested = json_data.get("model")
                if (
                    endpoint == "chat/completions"
                    and ctx.enable_litellm_prefix_routing
                    and isinstance(requested, str)
                    and requested.lower().startswith("litellm/")
                ):
                    if json_data.get("stream") is True:
                        return Response(
                            json.dumps(
                                {
                                    "error": (
                                        "Streaming via litellm/ routing is not enabled "
                                        "in middle_layer.py yet. Set stream=false."
                                    )
                                }
                            ),
                            status=501,
                            mimetype="application/json",
                        )
                    routed_model = requested.split("/", 1)[1].strip()
                    llm_resp, llm_err = ctx.call_litellm_chat(
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
                if endpoint == "chat/completions" and ctx.swarm_chat_enabled:
                    swarm_intent, swarm_canonical = ctx.swarm_chat_intent(requested)

                if swarm_intent is not None:
                    if swarm_intent == "pipeline":
                        _body, err, _ = ctx.run_swarm_chat_completion(
                            requested, json_data, intent="pipeline"
                        )
                        return Response(
                            json.dumps({"error": err, "redirect": "POST /swarm/pipeline"}),
                            status=400,
                            mimetype="application/json",
                        )

                    wants_stream = json_data.get("stream") is True
                    assert timer is not None
                    with timer.measure("upstream_ms"):
                        swarm_resp, swarm_err, swarm_err_details = ctx.run_swarm_chat_completion(
                            requested, json_data, intent=swarm_intent
                        )
                    if swarm_err or not swarm_resp:
                        body: dict = {"error": f"Swarm routing failed: {swarm_err}"}
                        if swarm_err_details:
                            body["error_details"] = swarm_err_details
                        resp_headers: dict = {}
                        if (
                            swarm_canonical
                            and isinstance(requested, str)
                            and swarm_canonical.lower() != requested.strip().lower()
                        ):
                            resp_headers["X-Swarm-Canonical-Name"] = swarm_canonical
                        if isinstance(swarm_err_details, dict):
                            kinds = swarm_err_details.get("kinds") or {}
                            if kinds:
                                resp_headers["X-Swarm-Error-Kinds"] = ",".join(
                                    f"{k}={v}" for k, v in sorted(kinds.items())
                                )
                        return Response(
                            json.dumps(body),
                            status=502,
                            mimetype="application/json",
                            headers=resp_headers or None,
                        )
                    if wants_stream:
                        stream_resp = ctx.swarm_body_to_sse_response(swarm_resp)
                        if timer is not None:
                            _attach_timing(
                                stream_resp,
                                timer,
                                method="POST",
                                path=f"/v1/{endpoint}",
                                status=200,
                                route_kind="swarm_chat",
                            )
                        return stream_resp
                    resp_headers = {
                        "X-Model-Routed-To": str(swarm_resp.get("model", "swarm/unknown")),
                        "X-Swarm-Intent": swarm_intent,
                    }
                    if (
                        swarm_canonical
                        and isinstance(requested, str)
                        and swarm_canonical.lower() != requested.strip().lower()
                    ):
                        resp_headers["X-Swarm-Canonical-Name"] = swarm_canonical
                    return _json_response(
                        swarm_resp,
                        200,
                        resp_headers,
                        timer=timer,
                        method="POST",
                        path=f"/v1/{endpoint}",
                        status=200,
                        route_kind="swarm_chat",
                    )

                with timer.measure("resolve_ms") if timer else nullcontext():
                    model_id, error = ctx.resolve_model_id(requested)
                    fallback_from = None

                    if error or not model_id:
                        if not ctx.is_placeholder(requested) and ctx.on_model_miss == "fallback":
                            if ctx.strict_loaded_models:
                                fb_ids, fb_err = ctx.get_loaded_model_ids()
                            else:
                                fb_ids, fb_err = ctx.get_model_ids()
                            if not fb_err and fb_ids:
                                model_id = fb_ids[0]
                                fallback_from = requested
                                error = None
                        if error or not model_id:
                            return Response(
                                json.dumps({"error": f"503 Service Unavailable - {error}"}),
                                status=503,
                                mimetype="application/json",
                            )

                json_data["model"] = model_id

                with timer.measure("upstream_ms") if timer else nullcontext():
                    resp = requests.request(
                        method=request.method,
                        url=f"{ctx.lm_studio_url}/v1/{endpoint}",
                        headers=headers,
                        data=json.dumps(json_data).encode("utf-8"),
                        cookies=request.cookies,
                        allow_redirects=False,
                        stream=True,
                        timeout=300,
                    )

                flask_resp = build_flask_response(resp)
                flask_resp.headers["X-Model-Routed-To"] = f"local/{model_id}"
                if fallback_from:
                    flask_resp.headers["X-Model-Resolution"] = (
                        f"fallback (requested '{fallback_from}', not loaded)"
                    )
                if timer is not None:
                    _attach_timing(
                        flask_resp,
                        timer,
                        method="POST",
                        path=f"/v1/{endpoint}",
                        status=resp.status_code,
                        routed_to=model_id,
                        model_spec=requested,
                    )
                return flask_resp

            except Exception:
                resp = requests.request(
                    method=request.method,
                    url=f"{ctx.lm_studio_url}/v1/{endpoint}",
                    headers=headers,
                    data=data,
                    cookies=request.cookies,
                    allow_redirects=False,
                    stream=True,
                    timeout=300,
                )
                return build_flask_response(resp)

        resp = requests.request(
            method=request.method,
            url=f"{ctx.lm_studio_url}/v1/{endpoint}",
            headers=headers,
            data=data,
            cookies=request.cookies,
            allow_redirects=False,
            stream=True,
            timeout=300,
        )
        return build_flask_response(resp)

    @app.route("/swarm/models", methods=["GET"])
    def swarm_models():
        ids, err = ctx.get_model_ids(force_refresh=True)
        return Response(
            json.dumps(
                {
                    "models": ids,
                    "roles": ctx.model_roles,
                    "default_model": ctx.default_model or None,
                    "max_parallel": ctx.max_parallel,
                    "anthropic_available": bool(ctx.anthropic_enabled),
                    "anthropic_model": ctx.anthropic_model if ctx.anthropic_enabled else None,
                    "litellm_available": ctx.litellm_available(),
                    "litellm_for_anthropic": bool(ctx.litellm_for_anthropic),
                    "swarm_chat_auto_tokens": sorted(ctx.swarm_auto_tokens),
                    "error": err,
                }
            ),
            status=200 if not err else 503,
            mimetype="application/json",
        )

    @app.route("/swarm/fanout", methods=["POST"])
    def swarm_fanout():
        timer = RequestTimer.start()
        data = request.get_json(silent=True) or {}
        models = data.get("models") or []
        messages = data.get("messages") or []
        if not isinstance(models, (list, str)) or not models:
            return Response(
                json.dumps({"error": "models (list, or 'auto') is required"}),
                status=400,
                mimetype="application/json",
            )
        if not isinstance(messages, list) or not messages:
            return Response(
                json.dumps({"error": "messages (list) is required"}),
                status=400,
                mimetype="application/json",
            )

        with timer.measure("resolve_ms"):
            models, exp_err = ctx.expand_swarm_models(models)
        if exp_err:
            return Response(json.dumps({"error": exp_err}), status=503, mimetype="application/json")
        if not models:
            return Response(
                json.dumps(
                    {
                        "error": (
                            "no LM Studio models loaded; load at least one or pass explicit models"
                        )
                    }
                ),
                status=503,
                mimetype="application/json",
            )

        common = {k: data.get(k) for k in ("max_tokens", "temperature", "top_p")}
        common = {k: v for k, v in common.items() if v is not None}

        with timer.measure("upstream_ms"):
            results, err = ctx.fanout(models, messages, common, max_parallel=data.get("max_parallel"))
        if err:
            return Response(json.dumps({"error": err}), status=503, mimetype="application/json")

        return _json_response(
            {
                "id": f"swarm_{uuid.uuid4().hex}",
                "object": "swarm.fanout",
                "created": int(time.time()),
                "responses": results,
            },
            200,
            {"X-Swarm-Models": ",".join((r or {}).get("model", "?") for r in results)},
            timer=timer,
            method="POST",
            path="/swarm/fanout",
            status=200,
        )

    @app.route("/swarm/vote", methods=["POST"])
    def swarm_vote():
        timer = RequestTimer.start()
        data = request.get_json(silent=True) or {}
        models = data.get("models") or []
        messages = data.get("messages") or []
        strategy = (data.get("strategy") or "best-of-n").lower()

        models_ok = isinstance(models, (list, str)) and models
        if not models_ok or not isinstance(messages, list) or not messages:
            return Response(
                json.dumps({"error": "models and messages are required"}),
                status=400,
                mimetype="application/json",
            )

        with timer.measure("resolve_ms"):
            models, exp_err = ctx.expand_swarm_models(models)
        if exp_err:
            return Response(json.dumps({"error": exp_err}), status=503, mimetype="application/json")
        if not models:
            return Response(
                json.dumps(
                    {
                        "error": (
                            "no LM Studio models loaded; load at least one or pass explicit models"
                        )
                    }
                ),
                status=503,
                mimetype="application/json",
            )

        common = {
            k: data.get(k) for k in ("max_tokens", "temperature", "top_p") if data.get(k) is not None
        }

        with timer.measure("upstream_ms"):
            candidates, err = ctx.fanout(models, messages, common)
            if err:
                return Response(json.dumps({"error": err}), status=503, mimetype="application/json")

            successes = [c for c in candidates if c["ok"] and c.get("text")]
            if not successes:
                errs = "; ".join(c.get("error") or "unknown" for c in candidates)
                return Response(
                    json.dumps({"error": f"all agents failed: {errs}", "candidates": candidates}),
                    status=502,
                    mimetype="application/json",
                )

            rationale = ""
            if strategy == "first-success":
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
                original_user = ctx.extract_user_intent({"messages": messages})
                judge_system = data.get("judge_system") or (
                    "You are a strict judge. Below are candidate responses to a user request "
                    "from different models, labeled [A], [B], etc. Pick the single best one. "
                    "Reply with ONLY the letter on its own line, then a one-sentence reason."
                )
                judge_messages = [
                    {"role": "system", "content": judge_system},
                    {
                        "role": "user",
                        "content": (
                            f"Original request:\n{original_user}\n\n"
                            f"Candidate responses:\n{rendered}\n\n"
                            "Pick the best one (A, B, ...) and explain briefly."
                        ),
                    },
                ]
                judge_request = data.get("judge") or "role:reasoner"
                avail, _ = ctx.get_model_ids()
                judge_id, jerr = ctx.resolve_model_id(judge_request, avail)

                if jerr or not judge_id:
                    winner = max(successes, key=lambda c: len(c.get("text", "")))
                    rationale = f"judge unavailable ({jerr or 'no model'}); picked longest"
                else:
                    with ctx.per_model_semaphore(judge_id):
                        jresp, jerr = ctx.lmstudio_chat_completion(
                            judge_id, judge_messages, max_tokens=200, temperature=0.0
                        )
                    verdict = ctx.extract_text(jresp)
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
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": winner["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "swarm": {
                    "strategy": strategy,
                    "winner": winner["model"],
                    "rationale": rationale,
                    "candidates": candidates,
                },
            }

        return _json_response(
            out,
            200,
            {"X-Swarm-Strategy": strategy, "X-Swarm-Winner": str(winner["model"])},
            timer=timer,
            method="POST",
            path="/swarm/vote",
            status=200,
        )

    @app.route("/swarm/pipeline", methods=["POST"])
    def swarm_pipeline():
        timer = RequestTimer.start()
        data = request.get_json(silent=True) or {}
        steps = data.get("steps") or []
        messages = data.get("messages") or []
        if not isinstance(steps, list) or not steps or not isinstance(messages, list) or not messages:
            return Response(
                json.dumps({"error": "steps and messages are required"}),
                status=400,
                mimetype="application/json",
            )

        with timer.measure("resolve_ms"):
            available, err = ctx.get_model_ids()
        if err:
            return Response(json.dumps({"error": err}), status=503, mimetype="application/json")

        history: list[dict] = []
        last_text = ""

        with timer.measure("upstream_ms"):
            for idx, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                name = step.get("name") or f"step_{idx}"
                step_ctx = {h["name"]: h["text"] for h in history}
                step_ctx["previous"] = last_text

                def _fmt(template, _ctx=step_ctx):
                    if not isinstance(template, str):
                        return template
                    t = re.sub(r"\{\{(\w+)\}\}", r"{\1}", template)
                    try:
                        return t.format(**_ctx)
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
                        m for m in messages if isinstance(m, dict) and m.get("role") != "system"
                    ]

                kwargs = {
                    k: step[k]
                    for k in ("max_tokens", "temperature", "top_p")
                    if step.get(k) is not None
                }

                model_id, resp, e, latency = ctx.run_one_agent(
                    {"model": step.get("model")}, agent_messages, kwargs, available
                )
                if e or not resp:
                    return Response(
                        json.dumps({"error": f"step '{name}' failed: {e}", "history": history}),
                        status=502,
                        mimetype="application/json",
                    )

                text = ctx.extract_text(resp)
                history.append(
                    {"name": name, "model": model_id, "text": text, "latency_ms": latency}
                )
                last_text = text

            final = history[-1] if history else {"text": "", "model": "?"}

        return _json_response(
            {
                "id": f"chatcmpl_{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"swarm/pipeline/{final['model']}",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final.get("text", "")},
                        "finish_reason": "stop",
                    }
                ],
                "swarm": {"strategy": "pipeline", "history": history},
            },
            200,
            {
                "X-Swarm-Strategy": "pipeline",
                "X-Swarm-Steps": ",".join(str(h["name"]) for h in history),
            },
            timer=timer,
            method="POST",
            path="/swarm/pipeline",
            status=200,
        )


__all__ = ["LmStudioRouteContext", "build_flask_response", "filtered_forward_headers", "register_lmstudio_routes"]
