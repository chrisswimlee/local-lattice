"""Swarm chat meta-model intent map: canonical names + deprecated aliases.

Pinned regressions:
- The openclaw runtime sends ``swarmIntelligence``, which used to fall
  through ``_is_swarm_chat_model`` and 404 on the unknown model id.
  swarmIntelligence must resolve to council intent + emit a one-shot
  DeprecationWarning.
- ``swarm/fanout`` used to silently run best-of-n. With the intent map,
  a missing ``swarm.strategy`` defaults to ``"fanout"`` (no judge).
- ``swarm/pipeline`` used to silently run best-of-n. The OpenAI chat
  shape can't carry ``stages[]``, so it must return 400 with a helpful
  redirect to ``POST /swarm/pipeline``.
- The MLX gateway must recognize the same alias set so callers see one
  contract regardless of which gateway they hit.
"""

from __future__ import annotations

import json

from tests._helpers import _load_middle_layer, _run_mlx_subprocess


def test_swarm_chat_intent_council_aliases() -> None:
    """Council aliases all resolve to intent='council' with the canonical
    name pointing at swarmCouncil. Regression: the openclaw runtime sends
    'swarmIntelligence' which used to be silently ignored.

    The deprecation warning for swarmIntelligence is asserted in
    ``test_swarm_chat_intent_deprecation_warning_once``; suppressed here so
    pytest's default warning-as-error doesn't shadow the actual assertion.
    """
    import warnings as _w
    mod = _load_middle_layer()
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        for alias in ("swarmCouncil", "swarmVote", "swarm/vote", "swarmIntelligence",
                      "SWARMCOUNCIL", "  swarmcouncil  "):
            intent, canonical = mod._swarm_chat_intent(alias)
            assert intent == "council", f"{alias!r} -> {intent}"
            assert canonical == "swarmCouncil"


def test_swarm_chat_intent_fanout_and_pipeline() -> None:
    mod = _load_middle_layer()
    assert mod._swarm_chat_intent("swarm/fanout") == ("fanout", "swarm/fanout")
    assert mod._swarm_chat_intent("swarm/pipeline") == ("pipeline", "swarm/pipeline")


def test_swarm_chat_intent_rejects_non_swarm_models() -> None:
    """No substring/wildcard interception — a model literally named
    'my-swarm-vote' must not get routed through the swarm path.
    """
    mod = _load_middle_layer()
    for not_swarm in ("granite-4.1-8b", "qwen3.5-122b-a10b", "my-swarm-vote",
                      "swarmcouncilxyz", None, 42, ""):
        intent, canonical = mod._swarm_chat_intent(not_swarm)
        assert intent is None, f"{not_swarm!r} should not match"
        assert canonical is None


def test_swarm_chat_intent_deprecation_warning_once() -> None:
    """Deprecated aliases must emit DeprecationWarning per AGENTS.md non-
    negotiable #1. To keep the warning useful, only fire once per alias
    per process (otherwise busy callers spam stderr).
    """
    import warnings as _w
    mod = _load_middle_layer()
    # Reset warned-set so this test is self-contained even if other tests ran first.
    mod._swarm_alias_warned.clear()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        mod._swarm_chat_intent("swarmIntelligence")
        mod._swarm_chat_intent("swarmIntelligence")
        mod._swarm_chat_intent("swarmIntelligence")
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1, f"expected exactly 1 DeprecationWarning, got {len(dep)}"
    assert "swarmIntelligence" in str(dep[0].message)
    assert "swarmCouncil" in str(dep[0].message)


def test_swarm_chat_intent_canonical_uses_distinct_name_for_fanout() -> None:
    """The X-Swarm-Canonical-Name header is set only when the request used
    a non-canonical alias. swarm/fanout is itself canonical for the fanout
    intent, so its canonical_name must be 'swarm/fanout' (not
    'swarmCouncil'). Otherwise the route handler would falsely flag it as
    deprecated.
    """
    mod = _load_middle_layer()
    _, canonical = mod._swarm_chat_intent("swarm/fanout")
    assert canonical == "swarm/fanout"


def test_run_swarm_chat_completion_pipeline_intent_returns_400_redirect() -> None:
    """swarm/pipeline on the chat path is rejected (not silently fallback to
    best-of-n) because the OpenAI shape can't carry stages[]. The error
    message must mention POST /swarm/pipeline so the caller can self-correct.
    """
    mod = _load_middle_layer()
    body, err, details = mod._run_swarm_chat_completion(
        "swarm/pipeline",
        {"messages": [{"role": "user", "content": "hi"}]},
        intent="pipeline",
    )
    assert body is None and details is None
    assert err is not None
    assert "POST /swarm/pipeline" in err
    assert "stages" in err.lower()


def test_run_swarm_chat_completion_fanout_intent_defaults_strategy() -> None:
    """Without an explicit swarm.strategy, swarm/fanout must use
    strategy=fanout (no judge). Confirmed by patching the shared
    ``swarm.fanout`` runner to capture the strategy that actually drives
    the pick logic.
    """
    mod = _load_middle_layer()

    def fake_fanout(specs, messages, common, max_parallel=None):
        return [
            {
                "agent_id": specs[0],
                "model": "fake-a",
                "ok": True,
                "error": None,
                "error_kind": None,
                "http_status": None,
                "error_detail": None,
                "latency_ms": 1,
                "response": {"choices": [{"message": {"content": "first"}}]},
                "text": "first",
            },
            {
                "agent_id": specs[1] if len(specs) > 1 else "?",
                "model": "fake-b",
                "ok": True,
                "error": None,
                "error_kind": None,
                "http_status": None,
                "error_detail": None,
                "latency_ms": 2,
                "response": {"choices": [{"message": {"content": "second-longer"}}]},
                "text": "second-longer",
            },
        ], None

    runner = mod._swarm_runner
    saved_fanout = runner.fanout
    saved_swarm_default = runner.SWARM_CHAT_DEFAULT_MODELS
    saved_mod_default = mod.SWARM_CHAT_DEFAULT_MODELS

    def fake_fanout_with_deps(specs, messages, common, deps, max_parallel=None):
        # Adapter: swarm.run_swarm_chat_completion calls swarm.fanout with a
        # ``deps`` positional, but the legacy fake_fanout signature doesn't
        # take it. Drop and forward.
        return fake_fanout(specs, messages, common, max_parallel=max_parallel)

    try:
        runner.fanout = fake_fanout_with_deps
        runner.SWARM_CHAT_DEFAULT_MODELS = ["fake-a", "fake-b"]
        mod.SWARM_CHAT_DEFAULT_MODELS = ["fake-a", "fake-b"]
        body, err, details = mod._run_swarm_chat_completion(
            "swarm/fanout",
            {"messages": [{"role": "user", "content": "hi"}]},
            intent="fanout",
        )
        assert err is None and details is None and body is not None
        assert body["swarm"]["strategy"] == "fanout"
        assert body["swarm"]["winner"] == "fake-a"
        assert body["swarm"]["rationale"].startswith("fanout")
    finally:
        runner.fanout = saved_fanout
        runner.SWARM_CHAT_DEFAULT_MODELS = saved_swarm_default
        mod.SWARM_CHAT_DEFAULT_MODELS = saved_mod_default


def test_mlx_swarm_intent_map_is_shared_with_lmstudio_gateway() -> None:
    """Pass-3 wired the MLX gateway through ``middle_layer.swarm`` for the
    intent map. Pin the alias contract from inside the MLX process so a
    regression can't desync the two gateways: callers sending
    ``swarmIntelligence`` to either gateway must get the same intent +
    canonical name + deprecation warning behavior.
    """
    snippet = """
    import warnings as _w

    out = {}
    out["council"] = mod._swarm_chat_intent("swarmCouncil")
    out["fanout"] = mod._swarm_chat_intent("swarm/fanout")
    out["pipeline"] = mod._swarm_chat_intent("swarm/pipeline")

    # Deprecation warning fires once for swarmIntelligence.
    mod._SWARM_CHAT_INTENTS  # ensure attr exists
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out["alias"] = mod._swarm_chat_intent("swarmIntelligence")
    out["dep_warn"] = any(
        issubclass(w.category, DeprecationWarning) for w in caught
    )
    out["dep_msg_mentions_canonical"] = any(
        "swarmCouncil" in str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
    )

    out["non_swarm"] = mod._swarm_chat_intent("granite-4.1-8b")
    out["alias_count"] = len(mod._SWARM_CHAT_INTENTS)
    out["canonical"] = mod._SWARM_CHAT_CANONICAL
    print("RESULT=" + json.dumps(out))
    """
    result = _run_mlx_subprocess(snippet)
    # Tuples become lists across the JSON boundary.
    assert result["council"] == ["council", "swarmCouncil"]
    assert result["fanout"] == ["fanout", "swarm/fanout"]
    assert result["pipeline"] == ["pipeline", "swarm/pipeline"]
    assert result["alias"] == ["council", "swarmCouncil"]
    assert result["dep_warn"] is True
    assert result["dep_msg_mentions_canonical"] is True
    assert result["non_swarm"] == [None, None]
    assert result["alias_count"] >= 6
    assert result["canonical"] == "swarmCouncil"
