"""Minimal smoke tests so ``make test`` is meaningful before Pass 5."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_placeholder_models_include_core_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "middle_layer.py"
    spec = importlib.util.spec_from_file_location("middle_layer_root_smoke", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "auto" in mod.PLACEHOLDER_MODELS
    assert "" in mod.PLACEHOLDER_MODELS


def test_cli_lmstudio_loader_finds_root_module() -> None:
    from middle_layer import cli

    fn = cli._legacy_lmstudio_main()
    assert callable(fn)
