"""Boot-time validation and deprecation-path tests for the MLX gateway.

These tests must not import ``middle_layerMLX.py`` into the pytest
process — that triggers MLX init which segfaults on shutdown in
Python 3.14 (see ``tests/_helpers.py:run_mlx_subprocess``). Instead we
spawn a tiny subprocess per case that imports the gateway with a
controlled environment and prints a JSON result line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from tests._helpers import REPO_ROOT


def _spawn(env_overrides: dict[str, str | None], snippet: str) -> subprocess.CompletedProcess:
    """Run ``snippet`` in a fresh interpreter with ``env_overrides``
    applied on top of the current env. ``None`` value means *unset*.

    The snippet runs *after* the module is loaded (or fails to load),
    so import-time DeprecationWarnings show up in stderr.
    """
    env = os.environ.copy()
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    # Make sure warnings actually fire — pyproject's filterwarnings only
    # applies in pytest, not in our subprocess.
    env["PYTHONWARNINGS"] = "default::DeprecationWarning"

    bootstrap = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        import warnings

        warnings.simplefilter("default", DeprecationWarning)

        spec = importlib.util.spec_from_file_location(
            "middle_layer_mlx_boot_test", r"{REPO_ROOT / 'middle_layerMLX.py'}"
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit as e:
            print("RESULT=" + json.dumps({{"systemexit": str(e)}}))
            sys.exit(0)
        """
    )
    program = bootstrap + "\n" + textwrap.dedent(snippet)
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
        env=env,
    )


def _parse_result(proc: subprocess.CompletedProcess) -> dict:
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT="):
            return json.loads(line[len("RESULT="):])
    raise AssertionError(
        f"no RESULT= line in stdout (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_max_concurrent_models_unset_uses_safe_default() -> None:
    """Sanity: with the knob unset, MAX_CONCURRENT_MODELS resolves to
    2 and boot validation passes.
    """
    proc = _spawn(
        {"MAX_CONCURRENT_MODELS": None},
        snippet="""
        result = {"value": mod.MAX_CONCURRENT_MODELS}
        print("RESULT=" + json.dumps(result))
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert _parse_result(proc) == {"value": 2}


_VALIDATE_SNIPPET = """
try:
    mod._validate_boot_knobs()
    print("RESULT=" + json.dumps({"systemexit": None}))
except SystemExit as e:
    print("RESULT=" + json.dumps({"systemexit": str(e)}))
"""


def test_validate_boot_knobs_rejects_zero_max_concurrent_models() -> None:
    """The audit's #2 P0 finding: MAX_CONCURRENT_MODELS=0 used to
    KeyError on the first load. _validate_boot_knobs must now exit
    cleanly with an actionable message.
    """
    proc = _spawn({"MAX_CONCURRENT_MODELS": "0"}, snippet=_VALIDATE_SNIPPET)
    result = _parse_result(proc)
    assert result.get("systemexit") is not None, result
    assert "MAX_CONCURRENT_MODELS=0" in result["systemexit"]
    assert "must be >= 1" in result["systemexit"]


def test_validate_boot_knobs_rejects_zero_max_parallel_model_calls() -> None:
    """MAX_PARALLEL_MODEL_CALLS=0 would create a 0-worker
    ThreadPoolExecutor and ValueError. Catch at boot.
    """
    proc = _spawn({"MAX_PARALLEL_MODEL_CALLS": "0"}, snippet=_VALIDATE_SNIPPET)
    result = _parse_result(proc)
    assert result.get("systemexit") is not None
    assert "MAX_PARALLEL_MODEL_CALLS=0" in result["systemexit"]


def test_validate_boot_knobs_rejects_negative_inflight_cap() -> None:
    """MLX_PER_MODEL_INFLIGHT_CAP=-1 is meaningless; reject."""
    proc = _spawn({"MLX_PER_MODEL_INFLIGHT_CAP": "-1"}, snippet=_VALIDATE_SNIPPET)
    result = _parse_result(proc)
    assert result.get("systemexit") is not None
    assert "MLX_PER_MODEL_INFLIGHT_CAP=-1" in result["systemexit"]


def test_validate_boot_knobs_accepts_valid_values() -> None:
    """Sanity: with reasonable values, validation passes silently."""
    proc = _spawn(
        {
            "MAX_CONCURRENT_MODELS": "2",
            "MAX_PARALLEL_MODEL_CALLS": "2",
            "MLX_PER_MODEL_INFLIGHT_CAP": "1",
        },
        snippet=_VALIDATE_SNIPPET,
    )
    result = _parse_result(proc)
    assert result == {"systemexit": None}


def test_mlx_per_model_inflight_cap_defaults_to_one_with_deprecation() -> None:
    """Default flipped from 0 to 1: direct python middle_layerMLX.py now
    gets admission back-pressure by default. AGENTS.md rule 1 requires
    a one-shot DeprecationWarning explaining how to pin the legacy.
    """
    proc = _spawn(
        {
            "MLX_PER_MODEL_INFLIGHT_CAP": None,
            "MLX_PER_MODEL_ADMISSION_CAP": None,
        },
        snippet="""
        print("RESULT=" + json.dumps({"cap": mod.MLX_PER_MODEL_INFLIGHT_CAP}))
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert _parse_result(proc) == {"cap": 1}
    # DeprecationWarning is emitted on stderr by warnings module.
    assert "MLX_PER_MODEL_INFLIGHT_CAP is unset" in proc.stderr
    assert "set MLX_PER_MODEL_INFLIGHT_CAP=0 explicitly" in proc.stderr.lower() or \
           "MLX_PER_MODEL_INFLIGHT_CAP=0" in proc.stderr


def test_mlx_per_model_inflight_cap_explicit_zero_no_warning() -> None:
    """Operators who explicitly pin =0 to keep legacy behavior must
    NOT see the DeprecationWarning (it's only for unset).
    """
    proc = _spawn(
        {"MLX_PER_MODEL_INFLIGHT_CAP": "0"},
        snippet="""
        print("RESULT=" + json.dumps({"cap": mod.MLX_PER_MODEL_INFLIGHT_CAP}))
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert _parse_result(proc) == {"cap": 0}
    assert "MLX_PER_MODEL_INFLIGHT_CAP is unset" not in proc.stderr


def test_mlx_per_model_admission_cap_legacy_fallback_warns() -> None:
    """Historical name MLX_PER_MODEL_ADMISSION_CAP is honored as a
    fallback for one minor with its own DeprecationWarning.
    """
    proc = _spawn(
        {
            "MLX_PER_MODEL_INFLIGHT_CAP": None,
            "MLX_PER_MODEL_ADMISSION_CAP": "2",
        },
        snippet="""
        print("RESULT=" + json.dumps({"cap": mod.MLX_PER_MODEL_INFLIGHT_CAP}))
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert _parse_result(proc) == {"cap": 2}
    assert "MLX_PER_MODEL_ADMISSION_CAP is deprecated" in proc.stderr


def test_max_workers_unset_does_not_warn() -> None:
    """MAX_WORKERS deprecation only fires on explicit use; unset is silent."""
    proc = _spawn(
        {"MAX_WORKERS": None},
        snippet="""
        print("RESULT=" + json.dumps({"mw": mod.MAX_WORKERS}))
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert _parse_result(proc) == {"mw": 0}
    assert "MAX_WORKERS is no longer honored" not in proc.stderr


def test_max_workers_explicit_value_warns_and_is_ignored() -> None:
    """Setting MAX_WORKERS used to silently do nothing. Now warns
    explicitly so operators stop trying to tune it.
    """
    proc = _spawn(
        {"MAX_WORKERS": "8"},
        snippet="""
        print("RESULT=" + json.dumps({"mw": mod.MAX_WORKERS}))
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert _parse_result(proc) == {"mw": 0}
    assert "MAX_WORKERS is no longer honored" in proc.stderr
    assert "upstream WSGI server" in proc.stderr


# Subprocess-based MLX tests are marked ``mlx`` so the default
# ``-m "not mlx and not network"`` invocation (Makefile default,
# pyproject.toml default) skips them — they each spawn a fresh Python
# interpreter to import middle_layerMLX, which dominates wall time
# (~5-10s per file vs <1s for the rest of the suite). Run with
# ``make test-mlx`` or ``pytest -m mlx`` when you've touched MLX code.
#
# They still don't *require* mlx_lm to be importable (the gateway
# wraps that import in try/except), so they pass even on Linux CI.
pytestmark = pytest.mark.mlx
