"""Structured swarm error metadata: classifier matrix, summarizer shape.

Pinned regressions:
- Every per-agent error string the gateway can emit must classify to a
  stable ``error_kind`` so callers (notably the openclaw embedded-agent
  runtime) can dispatch on the kind without parsing prose.
- The ``error_detail`` field must strip the ``LM Studio NNN: `` /
  ``Anthropic NNN: `` prefix so callers can show the upstream payload
  directly.
- The all-failed structured body shape (``summary``, ``agent_count``,
  ``kinds``, ``upstream_statuses``, ``agents[]``) is pinned by
  contract — openclaw and other clients build dispatch logic on it.
"""

from __future__ import annotations

from tests._helpers import _load_middle_layer


def test_swarm_error_classifier_matrix() -> None:
    """Pin the classifier mapping. Adding a new error_kind should also add
    a case here so the openclaw runtime never sees a silently-renamed
    enum.
    """
    mod = _load_middle_layer()

    cases = [
        # The two failure modes that bit us in production:
        (
            'LM Studio 400: {"error":{"message":"Failed to load model \\"x\\". '
            "Error: Model loading was stopped due to insufficient system "
            'resources."}}',
            "oom",
        ),
        (
            'LM Studio 400: {"error":"The model has crashed without additional '
            'information. (Exit code: null)"}',
            "model_crashed",
        ),
        # Other LM Studio buckets:
        ("LM Studio 500: <!DOCTYPE html><pre>Internal Server Error</pre>", "upstream_5xx"),
        ("LM Studio 404: not found", "upstream_4xx"),
        ("LM Studio 503: service unavailable", "upstream_5xx"),
        # Resolver-level:
        ("No model is loaded in LM Studio.", "no_models_loaded"),
        ("No loaded LM Studio model matched 'role:reasoner'. Available: []", "model_not_resolved"),
        (
            "swarm.models expanded to an empty set: no models are loaded in "
            "LM Studio. Load at least one model.",
            "no_models_loaded",
        ),
        # Transport:
        ("Timeout calling LM Studio", "timeout"),
        ("Cannot connect to LM Studio. Is it running?", "connection_error"),
        # Cloud adapters:
        ("Anthropic 401: invalid api key", "upstream_4xx"),
        ("Anthropic 500: server error", "upstream_5xx"),
        ("Anthropic error: ConnectionResetError", "anthropic_error"),
        ("LiteLLM error: SomeUpstreamFailure", "litellm_error"),
        # Config:
        ("ANTHROPIC_API_KEY not set", "config_error"),
        ("LiteLLM not available: import failed", "config_error"),
        # Fallback:
        ("Some weird untagged failure", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ]
    for err, expected in cases:
        got = mod._classify_swarm_error(err)
        assert got == expected, (
            f"_classify_swarm_error({err!r}) = {got!r}, expected {expected!r}"
        )
        # Every classification must stay inside the documented enum, even
        # on fallback. Caller contract is "treat unknown values as unknown".
        assert got in mod._SWARM_ERROR_KINDS


def test_swarm_error_classifier_extracts_upstream_status() -> None:
    """The HTTP status hidden in 'LM Studio NNN: ...' / 'Anthropic NNN: ...'
    error strings is what callers want to dispatch on. Make sure the
    helper pulls it out of every prefix shape we emit.
    """
    mod = _load_middle_layer()
    assert mod._extract_upstream_status("LM Studio 400: oom") == 400
    assert mod._extract_upstream_status("LM Studio 502: bad gateway") == 502
    assert mod._extract_upstream_status("Anthropic 429: rate limited") == 429
    assert mod._extract_upstream_status("LM Studio models endpoint returned 503") == 503
    assert mod._extract_upstream_status("Timeout calling LM Studio") is None
    assert mod._extract_upstream_status("") is None
    assert mod._extract_upstream_status(None) is None


def test_swarm_error_strip_upstream_prefix() -> None:
    """error_detail must be the upstream payload (no LM Studio NNN: prefix)
    so callers can show it directly without re-parsing.
    """
    mod = _load_middle_layer()
    assert (
        mod._strip_upstream_prefix("LM Studio 400: model could not load")
        == "model could not load"
    )
    assert (
        mod._strip_upstream_prefix("Anthropic 429:    rate limited\n")
        == "rate limited"
    )
    # No prefix → return as-is.
    assert mod._strip_upstream_prefix("Timeout calling LM Studio") == "Timeout calling LM Studio"


def test_summarize_failed_candidates_shape() -> None:
    """The structured all-failed body must include per-agent rows, an
    aggregate kind histogram, and an aggregate upstream-status histogram.
    Openclaw pins against this shape.
    """
    mod = _load_middle_layer()
    candidates = [
        {
            "agent_id": "role:reasoner",
            "model": "qwen3.5-122b-a10b",
            "ok": False,
            "error": "LM Studio 400: insufficient system resources",
            "error_kind": "oom",
            "http_status": 400,
            "error_detail": "insufficient system resources",
            "latency_ms": 120,
        },
        {
            "agent_id": "role:coder",
            "model": "qwen/qwen3-coder-next",
            "ok": False,
            "error": "LM Studio 400: insufficient system resources",
            "error_kind": "oom",
            "http_status": 400,
            "error_detail": "insufficient system resources",
            "latency_ms": 90,
        },
        {
            "agent_id": "role:fast",
            "model": "?",
            "ok": False,
            "error": "Timeout calling LM Studio",
            "error_kind": "timeout",
            "http_status": None,
            "error_detail": "Timeout calling LM Studio",
            "latency_ms": 180000,
        },
    ]
    body = mod._summarize_failed_candidates(candidates)
    assert body["summary"] == "all swarm agents failed"
    assert body["agent_count"] == 3
    assert body["kinds"] == {"oom": 2, "timeout": 1}
    assert body["upstream_statuses"] == {400: 2}
    assert len(body["agents"]) == 3
    # Per-agent rows must NOT include the heavy `response` payload (PII /
    # token-leak risk), only the metadata callers need.
    for row in body["agents"]:
        assert "response" not in row
        assert row["ok"] is False
        assert row["error_kind"] in mod._SWARM_ERROR_KINDS
        assert row["agent_id"] is not None


def test_summarize_uniform_empty_response_emits_actionable_summary() -> None:
    """When every candidate is an empty_response (reasoning models eating
    the whole token budget), the summary should tell the caller to raise
    max_tokens / inspect reasoning_content rather than blame the upstream.
    Otherwise openclaw would surface ``all swarm agents failed`` for what
    is actually a config issue.
    """
    mod = _load_middle_layer()
    candidates = [
        {"agent_id": f"role:r{i}", "model": "qwen3.5-122b-a10b",
         "ok": False, "error": "empty assistant content",
         "error_kind": "empty_response", "http_status": None,
         "error_detail": "upstream returned 200 with empty assistant content",
         "latency_ms": 1200}
        for i in range(3)
    ]
    body = mod._summarize_failed_candidates(candidates)
    assert body["kinds"] == {"empty_response": 3}
    assert "max_tokens" in body["summary"], body["summary"]
    assert "reasoning_content" in body["summary"], body["summary"]
