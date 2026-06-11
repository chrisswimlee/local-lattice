"""Tests for MLX grab mode + Hugging Face download error handling
(PR 4 of the MLX audit hardening plan).

Two audit findings closed here:
1. ``init_mlx_grab_model()`` and the ``download`` CLI subcommand used
   to let ``huggingface_hub.snapshot_download`` exceptions surface as
   raw tracebacks.
2. The dashboard ``/dashboard/api/models/load`` endpoint worked even
   in grab mode, bloating the LRU with models the API can't serve.

All tests run in a subprocess so MLX init doesn't leak.
"""

from __future__ import annotations

import pytest

from tests._helpers import _run_mlx_subprocess

# Subprocess-based MLX tests; skipped by default. See test_mlx_boot.py
# header for the rationale.
pytestmark = pytest.mark.mlx


def test_grab_init_handles_hf_download_failure_cleanly() -> None:
    """If ``snapshot_download`` raises (network error, auth, etc.),
    ``init_mlx_grab_model`` must return a clean error string —
    not let the exception propagate as a traceback.
    """
    snippet = """
    import os, sys

    # Inject a fake huggingface_hub.snapshot_download that always raises.
    import types
    fake_hub = types.ModuleType("huggingface_hub")
    def _fake_snapshot_download(**kwargs):
        raise ConnectionError("simulated network failure")
    fake_hub.snapshot_download = _fake_snapshot_download
    sys.modules["huggingface_hub"] = fake_hub

    # Point grab mode at an HF repo id that doesn't exist locally.
    os.environ["MLX_GRAB_MODEL"] = "fake-org/does-not-exist"
    os.environ["MLX_GRAB_CACHE"] = "/tmp/test-mlx-grab-cache"

    mod.MLX_AVAILABLE = True
    err = mod.init_mlx_grab_model()
    import json as _j
    print("RESULT=" + _j.dumps({"err": err}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["err"] is not None, "init should have returned an error string"
    assert "Hugging Face download failed" in result["err"]
    assert "simulated network failure" in result["err"]
    assert "delete that" in result["err"].lower() or "retry" in result["err"].lower()


def test_grab_init_handles_missing_config_after_download() -> None:
    """If the download finishes but config.json is missing (HF repo
    isn't a self-contained MLX model), grab init must return a
    descriptive error, not a bare ``Download finished but...`` string.
    """
    snippet = """
    import os, sys, types, tempfile

    fake_hub = types.ModuleType("huggingface_hub")
    def _fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        # Pretend the download "succeeded" but produced no config.json.
        os.makedirs(local_dir, exist_ok=True)
    fake_hub.snapshot_download = _fake_snapshot_download
    sys.modules["huggingface_hub"] = fake_hub

    cache = tempfile.mkdtemp(prefix="test-mlx-grab-")
    os.environ["MLX_GRAB_MODEL"] = "fake-org/no-config-json"
    os.environ["MLX_GRAB_CACHE"] = cache

    mod.MLX_AVAILABLE = True
    err = mod.init_mlx_grab_model()
    import json as _j
    print("RESULT=" + _j.dumps({"err": err}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["err"] is not None
    assert "config.json is missing" in result["err"]
    assert "self-contained MLX model layout" in result["err"]


def test_download_cli_subcommand_handles_failure_with_nonzero_exit() -> None:
    """``python middle_layerMLX.py download <repo>`` must return a
    non-zero exit code on download failure, with a clean log message.
    """
    snippet = """
    import sys, types

    fake_hub = types.ModuleType("huggingface_hub")
    def _fake_snapshot_download(**kwargs):
        raise OSError("simulated disk full")
    fake_hub.snapshot_download = _fake_snapshot_download
    sys.modules["huggingface_hub"] = fake_hub

    rc = mod._download_model("fake-org/whatever")
    import json as _j
    print("RESULT=" + _j.dumps({"rc": rc}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["rc"] == 1


_DASHBOARD_SETUP = """
    import os
    # Auth uses MIDDLE_LAYER_API_KEY; unset means _check_api_key returns True.
    os.environ.pop("MIDDLE_LAYER_API_KEY", None)

    import mlx_dashboard as _dash
    from types import SimpleNamespace

    fake_mgr = SimpleNamespace(
        get_available_aliases=lambda: ["fake-a"],
        load_model=lambda alias: None,
        get_loaded_aliases=lambda: [],
        get_last_load_error=lambda alias: None,
    )
"""


def test_dashboard_load_blocked_in_grab_mode() -> None:
    """In grab mode, the dashboard's /dashboard/api/models/load must
    return 400 with a clear message — not silently fan out RAM into
    extra models the chat API won't even serve.
    """
    snippet = _DASHBOARD_SETUP + """
    _dash.configure(
        mlx_manager=fake_mgr,
        mlx_available=True,
        grab_mode=lambda: True,
        get_roles=lambda: {},
        get_default_model_env=lambda: "",
        max_concurrent_models=2,
        max_parallel_model_calls=2,
        swarm_fanout_timeout=None,
        swarm_budget_fn=lambda: 60,
        anthropic_enabled=False,
        admission_snapshot_fn=lambda: {},
    )
    _dash.MLX_DASHBOARD_ENABLED = True

    from flask import Flask
    app = Flask(__name__)
    _dash.register(app)
    client = app.test_client()
    rv = client.post(
        "/dashboard/api/models/load",
        json={"alias": "fake-a"},
    )
    import json as _j
    body = rv.get_json()
    print("RESULT=" + _j.dumps({"status": rv.status_code, "body": body}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["status"] == 400, result
    assert "grab mode" in result["body"]["error"]


def test_dashboard_load_surfaces_guided_load_error() -> None:
    """Audit finding: the dashboard's load endpoint used to return a
    generic 'could not load' message even when MLXManager had a
    rich, OOM-decorated error in ``_last_load_errors``. Now the
    detail is surfaced as a 503 with the guided error string.
    """
    snippet = _DASHBOARD_SETUP + """
    fake_mgr.get_last_load_error = lambda alias: (
        "Failed to load MLX model 'fake-a': out of memory. "
        "Detected probable memory pressure. Try the stable launcher."
    )

    _dash.configure(
        mlx_manager=fake_mgr,
        mlx_available=True,
        grab_mode=lambda: False,
        get_roles=lambda: {},
        get_default_model_env=lambda: "",
        max_concurrent_models=2,
        max_parallel_model_calls=2,
        swarm_fanout_timeout=None,
        swarm_budget_fn=lambda: 60,
        anthropic_enabled=False,
        admission_snapshot_fn=lambda: {},
    )
    _dash.MLX_DASHBOARD_ENABLED = True

    from flask import Flask
    app = Flask(__name__)
    _dash.register(app)
    client = app.test_client()
    rv = client.post(
        "/dashboard/api/models/load",
        json={"alias": "fake-a"},
    )
    import json as _j
    body = rv.get_json()
    print("RESULT=" + _j.dumps({"status": rv.status_code, "body": body}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["status"] == 503, result
    assert "out of memory" in result["body"]["error"]
    assert "stable launcher" in result["body"]["error"]
