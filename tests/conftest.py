"""Pytest fixtures (Pass 1 minimal)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _quiet_placeholder_deprecation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid failing the suite on ``filterwarnings = error`` when legacy code warns."""
    monkeypatch.setenv("EXTRA_PLACEHOLDER_MODELS", "")
