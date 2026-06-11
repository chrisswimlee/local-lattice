"""True/false positive matrix for the shared OOM classifier
(``middle_layer.swarm.is_probable_oom_error``) and its integration
with ``classify_swarm_error`` (PR 7 of the MLX audit hardening pass).

The previous implementation used a naive ``any(marker in text)``
substring check that false-positived on "zoom", "room", and any other
word containing the literal "oom" substring. PR 7 switched to a
word-boundary regex and consolidated MLX + LM Studio + swarm into a
single classifier.
"""

from __future__ import annotations

import os
import pytest

# Set the legacy default env vars before the swarm module is imported,
# so the import-time DeprecationWarnings (which pyproject's
# ``filterwarnings = ["error"]`` would otherwise promote to a hard
# failure) never fire during test collection. The autouse fixture in
# tests/conftest.py sets these too but only at test-call time.
os.environ.setdefault("EXTRA_PLACEHOLDER_MODELS", "")
os.environ.setdefault("PREFER_LOADED_MODELS", "1")
os.environ.setdefault(
    "SWARM_CHAT_DEFAULT_MODELS", "role:reasoner,role:coder,role:fast"
)

from middle_layer.swarm import (  # noqa: E402
    classify_swarm_error,
    is_probable_oom_error,
)


# ---------------------------------------------------------------------------
# is_probable_oom_error: word-boundary semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        # MLX / Metal
        "out of memory",
        "MPS backend out of memory",
        "Out Of Memory while loading",
        "std::bad_alloc thrown by Metal allocator",
        # CUDA / generic GPU
        "CUDA out of memory",
        "RESOURCE_EXHAUSTED on device 0",
        # OS-level
        "process killed by OOM killer",
        "Killed",
        # Bare OOM token
        "OOM during inference",
        "oom: try a smaller model",
        # Allocation
        "Allocation failed (12 GB)",
        # LM Studio specific
        "Model loading was stopped",
        "Would likely overload your system",
        "Insufficient system resources",
        # Exception object
        Exception("CUDA out of memory while loading weights"),
        RuntimeError("std::bad_alloc"),
    ],
)
def test_is_probable_oom_error_true_positives(msg) -> None:
    assert is_probable_oom_error(msg) is True, f"expected OOM for: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        # The previously-false-positive cases.
        "zooming in too fast",
        "room temperature reading",
        "broomstick failure",
        "vroom vroom",
        # Unrelated errors.
        "Timeout waiting for response",
        "connection refused",
        "404 not found",
        "unauthorized",
        "validation error",
        # Empty / None / non-string.
        "",
        None,
        0,
        42,
        # The literal word "kill" without word-boundary "killed".
        "killing time",  # "kill" is a substring but not the word "killed"
    ],
)
def test_is_probable_oom_error_false_positives(msg) -> None:
    assert is_probable_oom_error(msg) is False, f"expected NOT-OOM for: {msg!r}"


# ---------------------------------------------------------------------------
# classify_swarm_error: MLX OOMs now classify as "oom"
# ---------------------------------------------------------------------------


def test_classify_swarm_error_picks_up_mlx_oom_phrasing() -> None:
    """Audit finding: MLX-native OOM exception strings used to
    classify as ``"unknown"`` in structured swarm error_details
    because the LM-Studio-targeted ``_OOM_PHRASES`` list didn't
    include MLX wording. PR 7 consolidates the classifier.
    """
    mlx_errors = [
        "Failed to load MLX model 'qwen-122b': out of memory",
        "MPS backend out of memory while generating",
        "std::bad_alloc",
        "process killed",
    ]
    for err in mlx_errors:
        assert classify_swarm_error(err) == "oom", (
            f"MLX-shape error should classify as 'oom': {err!r}"
        )


def test_classify_swarm_error_still_picks_up_lmstudio_oom_phrasing() -> None:
    """The legacy LM-Studio-targeted phrases must still classify as
    ``"oom"`` after consolidation (no regression).
    """
    lmstudio_errors = [
        "LM Studio 400: insufficient system resources",
        "model loading was stopped because it would likely overload your system",
    ]
    for err in lmstudio_errors:
        assert classify_swarm_error(err) == "oom", (
            f"LM Studio OOM-shape error should classify as 'oom': {err!r}"
        )


def test_classify_swarm_error_does_not_misclassify_unrelated_4xx() -> None:
    """A generic 4xx that doesn't carry OOM markers should still
    bucket as ``"upstream_4xx"`` — the OOM check shouldn't false-
    positive on the word "memory" alone for example.
    """
    err = "LM Studio 422: invalid request body"
    assert classify_swarm_error(err) == "upstream_4xx"


def test_classify_swarm_error_handles_word_zoom_correctly() -> None:
    """Direct regression test for the substring false-positive: a
    message containing 'zoom' or 'room' must not classify as OOM.
    """
    assert classify_swarm_error("zoom rate too low") != "oom"
    assert classify_swarm_error("room for improvement") != "oom"


def test_is_probable_oom_error_is_exported_as_underscore_alias() -> None:
    """The MLX gateway imports as the underscore-prefixed name for
    back-compat; this confirms the alias is exposed.
    """
    from middle_layer.swarm import _is_probable_oom_error

    assert _is_probable_oom_error is is_probable_oom_error
