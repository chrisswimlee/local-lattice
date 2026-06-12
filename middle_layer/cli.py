"""Console-script entry points for ``local-lattice-mlx`` / ``middle-layer-mlx``.

Pass 1 transitional shim. The real CLI definition still lives in
``middle_layerMLX._build_cli`` at the repository root; this module
forwards into it without modification, so

::

    local-lattice-mlx serve --help

(and ``middle-layer-mlx serve --help``) prints exactly the same help as

::

    python middle_layerMLX.py serve --help

once Pass 3 extracts the canonical implementation into this package,
this module will own the argparse definition outright and the legacy
top-level scripts will be replaced by one-line shims.

Two sub-entry-points are also provided so callers can be explicit about
which backend they want:

- ``main`` — auto-pick (currently always the MLX backend).
- ``main_mlx`` — MLX backend (Apple Silicon, ``mlx_lm``).
- ``main_lmstudio`` — legacy LM Studio proxy.

All three call the legacy ``main()`` of the corresponding module and
return whatever it returns (typically ``None``; SystemExit propagates).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable


def _package_parent() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def _ensure_legacy_on_path() -> None:
    """Make the legacy top-level modules importable.

    When this package is installed with ``pip install -e .`` from the
    repository root, the repo root is added to ``sys.path`` by the
    editable install and the legacy modules are importable
    automatically. When installed from a wheel, Hatch's
    ``force-include`` in ``pyproject.toml`` copies the legacy files
    next to this package, so the resolution still works.

    For defence-in-depth, we also add the parent of this package's
    directory to ``sys.path[0]`` if the legacy file is co-located
    there. This is a no-op in correctly-installed environments.
    """
    parent = _package_parent()
    candidate = os.path.join(parent, "middle_layerMLX.py")
    if os.path.isfile(candidate) and parent not in sys.path:
        sys.path.insert(0, parent)


def _legacy_main(module_name: str) -> Callable[[], None]:
    _ensure_legacy_on_path()
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - import-time wiring
        raise SystemExit(
            f"local-lattice-mlx: could not import legacy module {module_name!r}: "
            f"{exc}. Reinstall the package or run `python {module_name}.py serve` "
            "directly from a source checkout."
        ) from exc
    fn = getattr(module, "main", None)
    if not callable(fn):
        raise SystemExit(
            f"local-lattice-mlx: legacy module {module_name!r} has no callable "
            "main(); reinstall the package."
        )
    return fn


def _legacy_lmstudio_main() -> Callable[[], None]:
    """Load the LM Studio proxy module without colliding with package ``middle_layer``."""
    parent = _package_parent()
    candidates = [
        os.path.join(parent, "middle_layer_lmstudio.py"),
        os.path.join(parent, "middle_layer.py"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:  # pragma: no cover
        raise SystemExit(
            "local-lattice-lmstudio: could not find middle_layer.py (or "
            "middle_layer_lmstudio.py) next to the installed package."
        )
    mod_name = "middle_layer_legacy_lmstudio"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SystemExit(f"local-lattice-lmstudio: could not load spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "main", None)
    if not callable(fn):
        raise SystemExit(
            "local-lattice-lmstudio: legacy LM Studio module has no callable main()."
        )
    return fn


def main_mlx() -> None:
    """Entry point: ``local-lattice-mlx`` / ``middle-layer-mlx`` — MLX backend."""
    _legacy_main("middle_layerMLX")()


def main_lmstudio() -> None:
    """Entry point: ``local-lattice-lmstudio`` / ``middle-layer-lmstudio`` — legacy LM Studio proxy."""
    _legacy_lmstudio_main()()


def main() -> None:
    """Default entry point; identical to :func:`main_mlx` in this release."""
    main_mlx()


def main_init() -> None:
    """Entry point: ``local-lattice-init`` — probe backends and write roles JSON."""
    from middle_layer.bootstrap import run_init

    raise SystemExit(run_init())


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    main()
