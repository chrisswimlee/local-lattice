"""Direct tests for ``middle_layer.resolver`` (the Pass-3 extraction).

The behavioral contract is already pinned through the gateway surface in
``tests/test_resolver.py``; these tests exercise the pure module directly
so future callers (the MLX gateway, Pass-3 app factory) can rely on it
without loading the monolith.
"""

from __future__ import annotations

from middle_layer.resolver import (
    ResolverPolicy,
    is_placeholder,
    match_one,
    resolve_model_id,
    resolve_role,
)

AVAILABLE = ["qwen3.5-122b-a10b", "qwen/qwen3-coder-next", "gemma-4-31b-it"]


def _policy(**kw) -> ResolverPolicy:
    base = dict(
        roles={
            "coder": ["qwen3-coder-next"],
            "reasoner": ["qwen3.5-122b-a10b"],
            "default": [],
        },
        prefer_loaded=True,
        strict_loaded=False,
        default_model="",
        placeholder_ids=frozenset({"", "auto", "default"}),
    )
    base.update(kw)
    return ResolverPolicy(**base)


def test_match_one_exact_beats_substring() -> None:
    ids = ["qwen3.5-9b", "qwen3.5"]
    assert match_one("qwen3.5", ids) == "qwen3.5"
    assert match_one("9b", ids) == "qwen3.5-9b"
    assert match_one("nope", ids) is None
    assert match_one("", ids) is None


def test_is_placeholder_handles_non_strings() -> None:
    ph = frozenset({"", "auto"})
    assert is_placeholder(None, ph)
    assert is_placeholder(123, ph)
    assert is_placeholder("AUTO", ph)
    assert not is_placeholder("qwen3.5", ph)


def test_resolve_role_prefers_loaded_subset() -> None:
    policy = _policy(roles={"reasoner": ["qwen/qwen3-coder-next", "gemma-4-31b-it"]})
    got = resolve_role("reasoner", AVAILABLE, ["gemma-4-31b-it"], policy=policy)
    assert got == "gemma-4-31b-it"


def test_resolve_role_strict_suppresses_fallthrough() -> None:
    policy = _policy(strict_loaded=True)
    assert resolve_role("coder", AVAILABLE, ["gemma-4-31b-it"], policy=policy) is None
    # Without a loaded view, strict mode is inert and we may fall through.
    assert (
        resolve_role("coder", AVAILABLE, [], policy=policy) == "qwen/qwen3-coder-next"
    )


def test_resolve_model_id_priority_list_and_wildcard() -> None:
    policy = _policy()
    rid, err = resolve_model_id("nope,role:coder", AVAILABLE, None, policy=policy)
    assert err is None and rid == "qwen/qwen3-coder-next"
    rid, err = resolve_model_id("*gemma*", AVAILABLE, None, policy=policy)
    assert err is None and rid == "gemma-4-31b-it"


def test_resolve_model_id_placeholder_default_model_strict() -> None:
    # DEFAULT_MODEL pointing at a not-loaded id must not be JIT-loaded in
    # strict mode; the first loaded id wins instead.
    policy = _policy(strict_loaded=True, default_model="qwen3.5-122b-a10b")
    rid, err = resolve_model_id("auto", AVAILABLE, ["gemma-4-31b-it"], policy=policy)
    assert err is None and rid == "gemma-4-31b-it"


def test_resolve_model_id_strict_miss_errors() -> None:
    policy = _policy(strict_loaded=True)
    rid, err = resolve_model_id(
        "qwen3.5-122b-a10b", AVAILABLE, ["gemma-4-31b-it"], policy=policy
    )
    assert rid is None
    assert err is not None and "STRICT_LOADED_MODELS" in err
