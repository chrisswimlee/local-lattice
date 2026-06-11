"""Optional cloud escalation for the LM Studio gateway (Pass 3 extraction).

Handles Anthropic direct calls, LiteLLM-backed Anthropic routing, explicit
``litellm/`` prefix routing, and the local-first "big task" auto-escalation
heuristic. No Flask imports — the route layer calls into this module.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import requests


@dataclass(frozen=True)
class BigTaskThresholds:
    """Knobs for when a chat request is treated as a multi-step "big task"."""

    min_words: int = 80
    min_chars: int = 500
    min_bullets: int = 4
    min_step_markers: int = 3


def extract_user_intent_text(json_data: dict) -> str:
    """Best-effort extraction of user intent from OpenAI-shaped payloads."""
    parts: list[str] = []

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
                    for p in content:
                        if (
                            isinstance(p, dict)
                            and p.get("type") == "text"
                            and isinstance(p.get("text"), str)
                        ):
                            parts.append(p["text"])

    prompt = json_data.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)

    return "\n".join(parts).strip()


def looks_like_code(text_lower: str) -> bool:
    return bool(
        re.search(r"[{};()\[\]]", text_lower)
        or re.search(
            r"\b(def|class|function|var|let|const|import|from|#include|traceback|stack trace)\b",
            text_lower,
        )
        or "```" in text_lower
    )


def is_big_task(text: str, *, thresholds: BigTaskThresholds) -> bool:
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

    if word_count >= thresholds.min_words or char_count >= thresholds.min_chars:
        return True
    if bullet_count >= thresholds.min_bullets:
        return True
    if step_score >= thresholds.min_step_markers:
        return True

    return False


def litellm_model_for_anthropic(model_name: str) -> str:
    name = (model_name or "").strip()
    if "/" in name:
        return name
    return f"anthropic/{name}"


def litellm_response_to_dict(resp: object) -> dict:
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    return json.loads(json.dumps(resp, default=str))


def openai_messages_to_anthropic(
    json_data: dict,
    *,
    default_model: str,
) -> dict:
    messages_in = json_data.get("messages", [])
    system_chunks: list[str] = []
    out_messages: list[dict] = []

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
                texts = []
                for p in content:
                    if (
                        isinstance(p, dict)
                        and p.get("type") == "text"
                        and isinstance(p.get("text"), str)
                    ):
                        texts.append(p["text"])
                text = "\n".join(texts).strip()

            if not text:
                continue

            if role == "system":
                system_chunks.append(text)
                continue
            if role in ("user", "assistant"):
                out_messages.append(
                    {"role": role, "content": [{"type": "text", "text": text}]}
                )

    system_text = "\n".join(system_chunks).strip() if system_chunks else None

    max_tokens = json_data.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = 1024

    temperature = json_data.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        temperature = None

    payload: dict = {
        "model": default_model,
        "max_tokens": max_tokens,
        "messages": out_messages,
    }
    if system_text:
        payload["system"] = system_text
    if temperature is not None:
        payload["temperature"] = temperature

    return payload


def anthropic_to_openai_chat_completion(
    anthropic_json: dict,
    *,
    anthropic_model: str,
) -> dict:
    text_parts: list[str] = []
    content = anthropic_json.get("content")
    if isinstance(content, list):
        for p in content:
            if (
                isinstance(p, dict)
                and p.get("type") == "text"
                and isinstance(p.get("text"), str)
            ):
                text_parts.append(p["text"])
    assistant_text = "".join(text_parts)

    now = int(time.time())
    resp = {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": f"anthropic/{anthropic_model}",
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
        it = usage.get("input_tokens")
        ot = usage.get("output_tokens")
        if isinstance(it, int) and isinstance(ot, int):
            resp["usage"] = {
                "prompt_tokens": it,
                "completion_tokens": ot,
                "total_tokens": it + ot,
            }

    return resp


@dataclass
class CloudEscalationClient:
    """Anthropic + LiteLLM escalation with local-first big-task routing."""

    anthropic_api_key: str | None
    anthropic_base_url: str
    anthropic_model: str
    anthropic_version: str
    use_litellm_for_anthropic: bool
    litellm_timeout_seconds: int
    big_task: BigTaskThresholds = field(default_factory=BigTaskThresholds)
    litellm_completion: Callable | None = None
    litellm_import_error: str | None = None
    default_chat_timeout: float = 180.0

    def litellm_available(self) -> bool:
        return self.litellm_completion is not None

    def should_route_to_anthropic(self, endpoint: str, json_data: dict) -> bool:
        if endpoint not in ("chat/completions",):
            return False
        if not self.anthropic_api_key:
            return False

        text = extract_user_intent_text(json_data)
        if not text:
            return False

        if looks_like_code(text.lower()):
            return False

        return is_big_task(text, thresholds=self.big_task)

    def call_litellm_chat(
        self, messages: list, model_override: str | None = None, **kwargs: object
    ) -> tuple[dict | None, str | None]:
        if not self.litellm_available():
            return None, f"LiteLLM not available: {self.litellm_import_error or 'import failed'}"

        payload: dict = {
            "model": model_override,
            "messages": messages or [],
            "stream": False,
        }
        for k in ("max_tokens", "temperature", "top_p", "stop"):
            if kwargs.get(k) is not None:
                payload[k] = kwargs[k]

        try:
            resp = self.litellm_completion(  # type: ignore[misc]
                **payload,
                timeout=kwargs.get("timeout", self.litellm_timeout_seconds),
            )
            return litellm_response_to_dict(resp), None
        except Exception as e:  # noqa: BLE001
            return None, f"LiteLLM error: {e}"

    def call_anthropic_chat(
        self, messages: list, model_override: str | None = None, **kwargs: object
    ) -> tuple[dict | None, str | None]:
        if self.use_litellm_for_anthropic and self.litellm_available():
            model_name = litellm_model_for_anthropic(model_override or self.anthropic_model)
            return self.call_litellm_chat(messages, model_override=model_name, **kwargs)

        if not self.anthropic_api_key:
            return None, "ANTHROPIC_API_KEY not set"

        pseudo: dict = {"messages": messages}
        if kwargs.get("max_tokens") is not None:
            pseudo["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("temperature") is not None:
            pseudo["temperature"] = kwargs["temperature"]

        payload = openai_messages_to_anthropic(pseudo, default_model=self.anthropic_model)
        if model_override:
            payload["model"] = model_override

        try:
            r = requests.post(
                f"{self.anthropic_base_url.rstrip('/')}/v1/messages",
                headers={
                    "content-type": "application/json",
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": self.anthropic_version,
                },
                data=json.dumps(payload).encode("utf-8"),
                timeout=kwargs.get("timeout", self.default_chat_timeout),  # type: ignore[arg-type]
            )
            if r.status_code >= 400:
                return None, f"Anthropic {r.status_code}: {r.text[:300]}"
            return (
                anthropic_to_openai_chat_completion(
                    r.json(), anthropic_model=self.anthropic_model
                ),
                None,
            )
        except Exception as e:  # noqa: BLE001
            return None, f"Anthropic error: {e}"


__all__ = [
    "BigTaskThresholds",
    "CloudEscalationClient",
    "anthropic_to_openai_chat_completion",
    "extract_user_intent_text",
    "is_big_task",
    "litellm_model_for_anthropic",
    "litellm_response_to_dict",
    "looks_like_code",
    "openai_messages_to_anthropic",
]
