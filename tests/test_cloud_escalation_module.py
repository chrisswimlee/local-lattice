"""Direct tests for ``middle_layer.cloud_escalation`` (Pass 3 extraction)."""

from __future__ import annotations

from middle_layer.cloud_escalation import (
    BigTaskThresholds,
    CloudEscalationClient,
    anthropic_to_openai_chat_completion,
    extract_user_intent_text,
    is_big_task,
    looks_like_code,
    openai_messages_to_anthropic,
)


def test_extract_user_intent_from_messages_and_prompt() -> None:
    text = extract_user_intent_text(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "prompt": "legacy",
        }
    )
    assert "sys" in text and "hello" in text and "legacy" in text


def test_looks_like_code_detects_braces_and_keywords() -> None:
    assert looks_like_code("def foo():")
    assert looks_like_code("traceback here")
    assert not looks_like_code("plain english question")


def test_is_big_task_thresholds() -> None:
    th = BigTaskThresholds(min_words=10, min_chars=100, min_bullets=2, min_step_markers=2)
    assert is_big_task("one two three four five six seven eight nine ten eleven", thresholds=th)
    assert is_big_task("- a\n- b\n- c", thresholds=th)
    assert not is_big_task("short", thresholds=th)


def test_should_route_to_anthropic_skips_code_and_small_prompts() -> None:
    client = CloudEscalationClient(
        anthropic_api_key="sk-test",
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-test",
        anthropic_version="2025-04-14",
        use_litellm_for_anthropic=False,
        litellm_timeout_seconds=30,
    )
    assert not client.should_route_to_anthropic(
        "chat/completions", {"messages": [{"role": "user", "content": "fix ```python x```"}]}
    )
    assert not client.should_route_to_anthropic(
        "chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
    )
    big = " ".join(["word"] * 100)
    assert client.should_route_to_anthropic(
        "chat/completions", {"messages": [{"role": "user", "content": big}]}
    )
    assert not client.should_route_to_anthropic("models", {"messages": [{"role": "user", "content": big}]})


def test_openai_to_anthropic_and_back() -> None:
    payload = openai_messages_to_anthropic(
        {
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "ping"},
            ],
            "max_tokens": 128,
        },
        default_model="claude-test",
    )
    assert payload["model"] == "claude-test"
    assert payload["system"] == "be helpful"
    assert payload["messages"][0]["role"] == "user"

    openai = anthropic_to_openai_chat_completion(
        {
            "content": [{"type": "text", "text": "pong"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        anthropic_model="claude-test",
    )
    assert openai["choices"][0]["message"]["content"] == "pong"
    assert openai["usage"]["total_tokens"] == 5


def test_call_litellm_chat_delegates_to_completion_fn() -> None:
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    client = CloudEscalationClient(
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-test",
        anthropic_version="2025-04-14",
        use_litellm_for_anthropic=True,
        litellm_timeout_seconds=30,
        litellm_completion=fake_completion,
    )
    resp, err = client.call_litellm_chat(
        [{"role": "user", "content": "x"}], model_override="anthropic/claude-test", max_tokens=7
    )
    assert err is None and resp is not None
    assert seen["model"] == "anthropic/claude-test"
    assert seen["max_tokens"] == 7
