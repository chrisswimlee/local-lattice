"""Tests for the MLX gateway's /healthz field renames and the dead
TimeoutError handler removal (PR 5 of the audit hardening plan).
"""

from __future__ import annotations

import pytest

from tests._helpers import _run_mlx_subprocess

# Subprocess-based MLX tests; skipped by default. See test_mlx_boot.py
# header for the rationale.
pytestmark = pytest.mark.mlx


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


def test_healthz_includes_recent_load_errors() -> None:
    """After a load failure is recorded in MLXManager._last_load_errors,
    /healthz must surface it under recent_load_errors so operators can
    diagnose without grep-ing logs.
    """
    snippet = """
    mod.MLX_AVAILABLE = True
    # Manually seed a load error as if the load had failed.
    mod.mlx_manager._last_load_errors["fake-alias"] = (
        "Failed to load MLX model 'fake-alias': out of memory. Hint: lower MAX_CONCURRENT_MODELS."
    )
    mod.mlx_manager._last_load_error_ts["fake-alias"] = 1700000000

    client = mod.app.test_client()
    rv = client.get("/healthz")
    body = rv.get_json() or {}
    rle = body.get("recent_load_errors") or {}
    import json as _j
    print("RESULT=" + _j.dumps({
        "rle": rle,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert "fake-alias" in result["rle"], result
    assert result["rle"]["fake-alias"]["ts"] == 1700000000
    assert "out of memory" in result["rle"]["fake-alias"]["error"]


def test_manager_recent_load_errors_helper_returns_snapshot() -> None:
    """Sanity test for the new MLXManager.get_recent_load_errors()."""
    snippet = """
    mgr = mod.mlx_manager
    mgr._last_load_errors["a"] = "err_a"
    mgr._last_load_error_ts["a"] = 100
    mgr._last_load_errors["b"] = "err_b"
    mgr._last_load_error_ts["b"] = 200

    result = mgr.get_recent_load_errors()
    import json as _j
    print("RESULT=" + _j.dumps({"result": result}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["result"] == {
        "a": {"error": "err_a", "ts": 100},
        "b": {"error": "err_b", "ts": 200},
    }


def test_memory_stats_includes_load_error_count() -> None:
    """get_memory_stats now exposes recent_load_errors_count for the
    dashboard snapshot's quick triage view.
    """
    snippet = """
    mgr = mod.mlx_manager
    mgr._last_load_errors.clear()
    mgr._last_load_errors["x"] = "err_x"
    mgr._last_load_errors["y"] = "err_y"

    stats = mgr.get_memory_stats()
    import json as _j
    print("RESULT=" + _j.dumps({"stats": stats}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["stats"]["recent_load_errors_count"] == 2


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
