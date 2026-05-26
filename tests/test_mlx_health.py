"""Tests for the MLX gateway's /healthz field renames and the dead
TimeoutError handler removal (PR 5 of the audit hardening plan).
"""

from __future__ import annotations

from tests._helpers import _run_mlx_subprocess


def test_healthz_advertises_generation_advisory_timeout_sec() -> None:
    """The /healthz response must include the new field name
    ``generation_advisory_timeout_sec`` reflecting the actual
    semantics (advisory only, not enforced).
    """
    snippet = """
    # Pretend MLX is available so /healthz doesn't 503 on us.
    mod.MLX_AVAILABLE = True

    client = mod.app.test_client()
    rv = client.get("/healthz")
    body = rv.get_json() or {}
    import json as _j
    print("RESULT=" + _j.dumps({
        "status": rv.status_code,
        "has_advisory": "generation_advisory_timeout_sec" in body,
        "advisory_value": body.get("generation_advisory_timeout_sec"),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["has_advisory"] is True, result
    assert isinstance(result["advisory_value"], int)


def test_healthz_keeps_legacy_field_as_deprecated_alias() -> None:
    """The legacy ``generation_timeout_sec`` field is kept for one
    minor as an alias, with a sibling ``_deprecated`` field carrying
    the migration message. Per AGENTS.md rule 1 (deprecation paths).
    """
    snippet = """
    mod.MLX_AVAILABLE = True
    client = mod.app.test_client()
    rv = client.get("/healthz")
    body = rv.get_json() or {}
    import json as _j
    print("RESULT=" + _j.dumps({
        "legacy_value": body.get("generation_timeout_sec"),
        "deprecated_note": body.get("generation_timeout_sec_deprecated"),
        "advisory_value": body.get("generation_advisory_timeout_sec"),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["legacy_value"] == result["advisory_value"], (
        "legacy field must mirror new field while the alias is active"
    )
    assert result["deprecated_note"], "deprecation note field missing"
    assert "renamed" in result["deprecated_note"]
    assert "advisory" in result["deprecated_note"].lower()


def test_generation_helper_does_not_raise_timeout_error() -> None:
    """Sanity test for the removal of dead TimeoutError handlers:
    ``_mlx_generate_text_timed`` must never raise TimeoutError, even
    when ``timeout_sec`` is set to 0 or negative. The audit found three
    dead ``except TimeoutError`` handlers across the codebase; PR 5
    removed them and consolidated the catch into a single
    ``except Exception`` per generation site. This test pins the
    invariant so a future refactor can't reintroduce the dead handlers.
    """
    snippet = """
    import types

    captured = {"out": None}

    # Replace _mlx_generate_text with a fake that just returns a tuple.
    # We're testing _mlx_generate_text_timed's wrapping behavior, not
    # the real MLX call.
    def fake_generate(model, tokenizer, prompt, max_tokens, temperature=None, top_p=None):
        captured["out"] = "fake"
        return ("hello", 1, 1)

    mod._mlx_generate_text = fake_generate

    # Even with timeout_sec=0 the helper must NOT raise — it returns
    # the result and logs a soft-budget warning at most.
    out_zero = mod._mlx_generate_text_timed(None, None, "p", 4, timeout_sec=0)
    out_neg = mod._mlx_generate_text_timed(None, None, "p", 4, timeout_sec=-1)
    out_normal = mod._mlx_generate_text_timed(None, None, "p", 4, timeout_sec=300)

    import json as _j
    print("RESULT=" + _j.dumps({
        "zero": list(out_zero),
        "neg": list(out_neg),
        "normal": list(out_normal),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["zero"] == ["hello", 1, 1]
    assert result["neg"] == ["hello", 1, 1]
    assert result["normal"] == ["hello", 1, 1]
