"""Pytest fixtures (Pass 1 minimal)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _quiet_default_deprecations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid failing the suite on ``filterwarnings = error`` when legacy code
    warns about unset env vars. Each of these knobs has a deprecation-path
    default flip in flight; explicitly setting them to the legacy value
    silences the one-shot DeprecationWarning while preserving the legacy
    behavior the existing tests are written against. Individual tests can
    still override any of these via ``monkeypatch.setenv``.
    """
    monkeypatch.setenv("EXTRA_PLACEHOLDER_MODELS", "")
    monkeypatch.setenv("PREFER_LOADED_MODELS", "1")
    monkeypatch.setenv(
        "SWARM_CHAT_DEFAULT_MODELS", "role:reasoner,role:coder,role:fast"
    )
