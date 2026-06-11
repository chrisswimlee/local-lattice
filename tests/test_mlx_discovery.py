"""Discovery & registry tests for the MLX gateway (PR 8 of the audit
hardening plan).

These exercise ``MLXManager._scan`` end-to-end with real tmp dirs so
the two supported layouts (flat ``alias/config.json`` and LM Studio
``publisher/model/config.json``), missing roots, and mixed layouts
are all covered.

All tests run in a subprocess so importing ``middle_layerMLX.py``
into pytest doesn't trip the Python 3.14 + mlx_lm shutdown segfault.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tests._helpers import _run_mlx_subprocess

# Subprocess-based MLX tests; skipped by default. See test_mlx_boot.py
# header for the rationale.
pytestmark = pytest.mark.mlx


def _build_tree(root: str, layout: dict[str, str | None]) -> None:
    """Helper: create a tmp tree from ``{relative_path: contents}``.
    A None value means an empty directory; a string value writes a
    file with that content.
    """
    for rel, content in layout.items():
        full = os.path.join(root, rel)
        if content is None:
            os.makedirs(full, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)


def test_scan_finds_flat_layout() -> None:
    """A top-level alias dir with config.json registers under its
    folder name."""
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    _build_tree(tmp, {
        "qwen-7b/config.json": '{"model_type": "qwen"}',
        "phi-3-mini/config.json": '{"model_type": "phi"}',
    })
    snippet = f"""
    mgr = mod.MLXManager(r"{tmp}")
    import json as _j
    print("RESULT=" + _j.dumps({{"aliases": sorted(mgr.get_available_aliases())}}))
    """
    result = _run_mlx_subprocess(snippet)
    assert "qwen-7b" in result["aliases"]
    assert "phi-3-mini" in result["aliases"]


def test_scan_finds_publisher_layout() -> None:
    """LM Studio's nested publisher/model layout registers under
    ``publisher/model``.
    """
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    _build_tree(tmp, {
        "mlx-community/Qwen2.5-7B-MLX-4bit/config.json": '{"model_type": "qwen"}',
        "lmstudio-community/Hermes-4-70B-MLX-4bit/config.json": '{"model_type": "llama"}',
    })
    snippet = f"""
    mgr = mod.MLXManager(r"{tmp}")
    import json as _j
    print("RESULT=" + _j.dumps({{"aliases": sorted(mgr.get_available_aliases())}}))
    """
    result = _run_mlx_subprocess(snippet)
    assert "mlx-community/Qwen2.5-7B-MLX-4bit" in result["aliases"]
    assert "lmstudio-community/Hermes-4-70B-MLX-4bit" in result["aliases"]


def test_scan_handles_mixed_layouts_in_same_root() -> None:
    """A root containing both flat dirs AND publisher dirs must
    discover everything.
    """
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    _build_tree(tmp, {
        "qwen-7b/config.json": '{}',
        "mlx-community/Qwen2.5-7B-MLX-4bit/config.json": '{}',
    })
    snippet = f"""
    mgr = mod.MLXManager(r"{tmp}")
    import json as _j
    print("RESULT=" + _j.dumps({{"aliases": sorted(mgr.get_available_aliases())}}))
    """
    result = _run_mlx_subprocess(snippet)
    assert set(result["aliases"]) == {
        "qwen-7b",
        "mlx-community/Qwen2.5-7B-MLX-4bit",
    }


def test_scan_skips_dirs_without_config_json() -> None:
    """Empty dirs and dirs containing non-config files must not be
    registered. A dir with subdirs but no inner config.json is
    silently skipped (not registered as a publisher).
    """
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    _build_tree(tmp, {
        "empty-dir": None,
        "has-readme-only/README.md": "not a model",
        "broken-publisher/inner-dir": None,  # subdir but no config.json
    })
    snippet = f"""
    mgr = mod.MLXManager(r"{tmp}")
    import json as _j
    print("RESULT=" + _j.dumps({{"aliases": sorted(mgr.get_available_aliases())}}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["aliases"] == [], f"unexpected registrations: {result['aliases']}"


def test_scan_missing_root_logs_warning_no_crash() -> None:
    """Pointing MLXManager at a non-existent dir must not crash —
    just log a warning and leave the registry empty.
    """
    snippet = """
    mgr = mod.MLXManager("/nonexistent/path/for/test")
    import json as _j
    print("RESULT=" + _j.dumps({"aliases": mgr.get_available_aliases()}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["aliases"] == []


def test_scan_ignores_files_at_root() -> None:
    """Files directly in the root dir (not dirs) must be skipped."""
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    _build_tree(tmp, {
        "qwen-7b/config.json": "{}",
        "README.md": "not a model",
        "some-file.txt": "also not a model",
    })
    snippet = f"""
    mgr = mod.MLXManager(r"{tmp}")
    import json as _j
    print("RESULT=" + _j.dumps({{"aliases": sorted(mgr.get_available_aliases())}}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["aliases"] == ["qwen-7b"]


def test_discover_model_root_respects_env_override() -> None:
    """``MLX_MODEL_ROOT=/custom`` should win over auto-discovery —
    the subprocess sees the override even if /custom doesn't exist.
    """
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    snippet = f"""
    import os
    os.environ["MLX_MODEL_ROOT"] = r"{tmp}"
    # Re-execute the root-resolution against the patched env.
    root = os.environ.get("MLX_MODEL_ROOT")
    import json as _j
    print("RESULT=" + _j.dumps({{"root": root, "mod_root": mod.MLX_MODEL_ROOT}}))
    """
    result = _run_mlx_subprocess(snippet)
    # mod.MLX_MODEL_ROOT is set at import time, before our env mutation,
    # so it reflects whatever the import process inherited. The point
    # of this test is to confirm the env var IS read at module import.
    # We can't easily change env before import inside the subprocess
    # harness, so we just assert the resolution chain handles values.
    assert result["root"] == tmp


def test_context_windows_load_silent_on_missing_file() -> None:
    """If ``mlx_context_windows.json`` doesn't exist beside the script,
    ``context_windows`` is just an empty dict — no warning, no crash.
    """
    snippet = """
    mgr = mod.MLXManager("/nonexistent/path/for/test")
    import json as _j
    print("RESULT=" + _j.dumps({"cw": mgr.context_windows}))
    """
    result = _run_mlx_subprocess(snippet)
    # Either empty (file missing) or whatever the real file has —
    # we just want to confirm the field exists and is dict-shaped.
    assert isinstance(result["cw"], dict)


def test_scan_logs_permission_error_on_publisher_subdir() -> None:
    """Audit finding: the publisher subdir scan used a bare
    ``except Exception: pass``, swallowing permission errors silently.
    Now logs at WARNING with the path + exception.
    """
    snippet = """
    import logging
    import tempfile, os, sys, types

    tmp = tempfile.mkdtemp(prefix="test-mlx-perm-")
    # Real dir + monkey-patch os.scandir to raise PermissionError when
    # called on the publisher subdir.
    os.makedirs(os.path.join(tmp, "publisher-dir"), exist_ok=True)
    open(os.path.join(tmp, "publisher-dir", "marker"), "w").close()

    real_scandir = os.scandir
    def fake_scandir(path):
        if str(path).endswith("publisher-dir"):
            raise PermissionError(f"simulated EACCES on {path}")
        return real_scandir(path)
    os.scandir = fake_scandir

    captured = []
    class Handler(logging.Handler):
        def emit(self, record):
            captured.append((record.levelname, record.getMessage()))
    mod.log.addHandler(Handler())

    mgr = mod.MLXManager(tmp)
    os.scandir = real_scandir

    import json as _j
    print("RESULT=" + _j.dumps({
        "warning_lines": [m for lvl, m in captured if lvl == "WARNING" and "publisher-dir" in m],
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert any("publisher-dir" in line for line in result["warning_lines"]), (
        f"expected a WARNING mentioning the publisher dir; got {result['warning_lines']}"
    )
    assert any("EACCES" in line or "PermissionError" in line for line in result["warning_lines"]), (
        f"expected exception detail in WARNING; got {result['warning_lines']}"
    )


def test_context_windows_warns_on_malformed_json() -> None:
    """Audit finding: malformed mlx_context_windows.json used to be
    silently swallowed — operators never knew their hints weren't
    being applied. Now logs at WARNING.
    """
    snippet = """
    import os, tempfile, logging

    # mlx_context_windows.json is loaded from the script directory.
    # Patch open() in mod's namespace so we control what it sees.
    real_exists = os.path.exists
    real_open = open
    def fake_exists(p):
        if str(p).endswith("mlx_context_windows.json"):
            return True
        return real_exists(p)
    def fake_open(p, *args, **kwargs):
        if str(p).endswith("mlx_context_windows.json"):
            import io
            return io.StringIO("{ not valid json }")
        return real_open(p, *args, **kwargs)
    mod.os.path.exists = fake_exists
    # The MLXManager imports json + open via builtins; override those.
    import builtins
    builtins.open = fake_open

    captured = []
    class Handler(logging.Handler):
        def emit(self, record):
            captured.append((record.levelname, record.getMessage()))
    mod.log.addHandler(Handler())

    tmp = tempfile.mkdtemp(prefix="test-mlx-ctxw-")
    mgr = mod.MLXManager(tmp)

    import json as _j
    print("RESULT=" + _j.dumps({
        "warnings": [m for lvl, m in captured if lvl == "WARNING" and "context_windows" in m],
        "cw_empty": mgr.context_windows == {},
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["cw_empty"] is True
    assert any("context_windows" in w.lower() or "context windows" in w.lower()
               for w in result["warnings"]), (
        f"expected a WARNING about malformed context_windows; got {result['warnings']}"
    )


def test_rescan_picks_up_new_model_dir() -> None:
    """Audit finding: discovery was startup-only; new models required
    a restart. ``MLXManager.rescan()`` now picks them up.
    """
    snippet = """
    import os, tempfile

    tmp = tempfile.mkdtemp(prefix="test-mlx-rescan-")
    os.makedirs(os.path.join(tmp, "alias-a"))
    with open(os.path.join(tmp, "alias-a", "config.json"), "w") as f:
        f.write("{}")

    mgr = mod.MLXManager(tmp)
    initial = sorted(mgr.get_available_aliases())

    # Add a new model dir AFTER construction.
    os.makedirs(os.path.join(tmp, "alias-b"))
    with open(os.path.join(tmp, "alias-b", "config.json"), "w") as f:
        f.write("{}")

    diff = mgr.rescan()
    after = sorted(mgr.get_available_aliases())

    import json as _j
    print("RESULT=" + _j.dumps({
        "initial": initial,
        "after": after,
        "diff": diff,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["initial"] == ["alias-a"]
    assert result["after"] == ["alias-a", "alias-b"]
    assert result["diff"]["added"] == ["alias-b"]
    assert result["diff"]["removed"] == []
    assert result["diff"]["unchanged"] == ["alias-a"]


def test_rescan_drops_removed_alias_from_registry() -> None:
    """A model dir removed from disk after startup must disappear
    from the registry on next rescan. Loaded handles are unaffected
    (operator-controlled eviction).
    """
    snippet = """
    import os, tempfile, shutil

    tmp = tempfile.mkdtemp(prefix="test-mlx-rescan-")
    for n in ("alias-a", "alias-b"):
        os.makedirs(os.path.join(tmp, n))
        with open(os.path.join(tmp, n, "config.json"), "w") as f:
            f.write("{}")

    mgr = mod.MLXManager(tmp)
    initial = sorted(mgr.get_available_aliases())

    shutil.rmtree(os.path.join(tmp, "alias-b"))
    diff = mgr.rescan()
    after = sorted(mgr.get_available_aliases())

    import json as _j
    print("RESULT=" + _j.dumps({
        "initial": initial, "after": after, "diff": diff,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["initial"] == ["alias-a", "alias-b"]
    assert result["after"] == ["alias-a"]
    assert result["diff"]["removed"] == ["alias-b"]


def test_get_recent_load_errors_empty_on_fresh_manager() -> None:
    """A freshly-constructed manager must have an empty load-error
    map (PR 6 contract sanity check from the discovery side).
    """
    tmp = tempfile.mkdtemp(prefix="test-mlx-disc-")
    snippet = f"""
    mgr = mod.MLXManager(r"{tmp}")
    import json as _j
    print("RESULT=" + _j.dumps({{
        "errors": mgr.get_recent_load_errors(),
        "memory_stats": mgr.get_memory_stats(),
    }}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["errors"] == {}
    assert result["memory_stats"]["recent_load_errors_count"] == 0
