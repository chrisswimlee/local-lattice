"""Shared test helpers extracted out of the original ``test_smoke.py`` so
each focused test file can import them without re-defining the import-from-
path / MLX-subprocess plumbing.

Kept out of ``conftest.py`` on purpose: these are plain functions, not
pytest fixtures, and a regular helper module is easier to import explicitly
than relying on pytest's auto-conftest behavior.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_middle_layer():
    """Import ``middle_layer.py`` (the LM Studio gateway script, not the
    package) under a unique module name so each test gets a fresh, isolated
    module — important because the swarm intent map keeps a process-level
    ``_swarm_alias_warned`` set that would leak across tests otherwise.
    """
    path = REPO_ROOT / "middle_layer.py"
    spec = importlib.util.spec_from_file_location("middle_layer_root_smoke", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Back-compat alias — historical tests imported the underscore-prefixed name.
_load_middle_layer = load_middle_layer


def run_mlx_subprocess(snippet: str) -> dict:
    """Run a tiny script in a fresh interpreter that loads
    ``middle_layerMLX.py``.

    Isolates MLX init and teardown from the pytest process (Python 3.14 +
    the MLX library segfault during interpreter shutdown when both Flask
    and MLX threads are torn down in-process). The snippet must end by
    printing one JSON line to stdout starting with ``RESULT=``.
    """
    bootstrap = textwrap.dedent(
        f"""
        import importlib.util
        import json
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location(
            "middle_layer_mlx_subproc", r"{REPO_ROOT / 'middle_layerMLX.py'}"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        """
    )
    program = bootstrap + "\n" + textwrap.dedent(snippet)
    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"mlx subprocess failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT="):
            return json.loads(line[len("RESULT=") :])
    raise AssertionError(f"no RESULT= line in stdout:\n{proc.stdout}")


_run_mlx_subprocess = run_mlx_subprocess  # back-compat alias


__all__ = [
    "REPO_ROOT",
    "load_middle_layer",
    "_load_middle_layer",
    "run_mlx_subprocess",
    "_run_mlx_subprocess",
]
