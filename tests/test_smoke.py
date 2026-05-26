"""Module-level / cross-cutting smoke tests for both gateways.

Focused tests live in sibling files:

  - ``tests/test_resolver.py``       — LM Studio resolver behavior
  - ``tests/test_swarm_intents.py``  — chat meta-model intent map
  - ``tests/test_swarm_errors.py``   — structured per-agent errors
  - ``tests/test_concurrency.py``    — per-model semaphore

This file keeps tests that don't fit cleanly under any single focus:
placeholder model registry, CLI loader entry point, and the
``swarm.models`` sentinel expansion behavior on both gateways.

Shared helpers (``_load_middle_layer``, ``_run_mlx_subprocess``) live in
``tests/_helpers.py``.
"""

from __future__ import annotations

import json

from tests._helpers import _load_middle_layer, _run_mlx_subprocess


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------


def test_placeholder_models_include_core_ids() -> None:
    mod = _load_middle_layer()
    assert "auto" in mod.PLACEHOLDER_MODELS
    assert "" in mod.PLACEHOLDER_MODELS


def test_cli_lmstudio_loader_finds_root_module() -> None:
    from middle_layer import cli

    fn = cli._legacy_lmstudio_main()
    assert callable(fn)


# ---------------------------------------------------------------------------
# Swarm "auto" sentinel expansion (LM Studio backend)
# ---------------------------------------------------------------------------


def test_lmstudio_swarm_auto_expands_to_loaded_models() -> None:
    mod = _load_middle_layer()

    loaded = ["qwen2.5-7b-instruct", "qwen2.5-coder-7b"]
    expanded, err = mod._expand_swarm_models("auto", available=loaded)
    assert err is None
    assert expanded == loaded


def test_lmstudio_swarm_auto_dedupes_and_preserves_order() -> None:
    mod = _load_middle_layer()

    loaded = ["model-a", "model-b", "model-c"]
    spec = ["model-b", "auto", "anthropic", "model-b"]
    expanded, err = mod._expand_swarm_models(spec, available=loaded)
    assert err is None
    # model-b first (explicit), then auto-expansion skips model-b, then anthropic.
    assert expanded == ["model-b", "model-a", "model-c", "anthropic"]


def test_lmstudio_swarm_passthrough_when_no_sentinel() -> None:
    mod = _load_middle_layer()

    spec = ["role:coder", "role:reasoner"]
    expanded, err = mod._expand_swarm_models(spec, available=[])
    assert err is None
    assert expanded == spec


def test_lmstudio_swarm_recognized_tokens() -> None:
    mod = _load_middle_layer()

    for tok in ("auto", "loaded", "*", "all", "all-loaded", "AUTO", " Loaded "):
        assert mod._is_auto_swarm_token(tok), tok
    for tok in ("role:coder", "anthropic", "qwen2.5", "", None, 42):
        assert not mod._is_auto_swarm_token(tok), tok


def test_lmstudio_swarm_invalid_spec_returns_error() -> None:
    mod = _load_middle_layer()

    expanded, err = mod._expand_swarm_models(123, available=[])
    assert expanded is None
    assert err and "list" in err.lower()


# ---------------------------------------------------------------------------
# Swarm-intelligence-effectiveness pass: ``auto`` filters out embedding
# models, ``SWARM_CHAT_AUTO_MAX`` caps expansion, and the chat-capable
# helper passes through chat ids unchanged.
# ---------------------------------------------------------------------------


def test_is_chat_capable_filters_embedding_id_substrings() -> None:
    """Embedding model ids that LM Studio exposes on this box (and the
    common naming conventions across HF) must be rejected. Chat / VLM ids
    must pass through unchanged.
    """
    mod = _load_middle_layer()

    # Reject — known embedding naming patterns.
    for mid in (
        "text-embedding-nomic-embed-text-v1.5",
        "nomic-embed-text-v1.5",
        "bge-base-en-v1.5",
        "e5-large-v2",
        "intfloat/e5-mistral-7b-instruct",
        "embedding-3-large",
    ):
        assert not mod.is_chat_capable_model_id(mid), mid

    # Accept — chat / coder / reasoning / vision ids must not be filtered.
    for mid in (
        "qwen3.5-122b-a10b",
        "qwen/qwen3-coder-next",
        "nousresearch/hermes-4-70b",
        "gemma-4-26b-a4b-it",
        "qwen3.6-27b",  # VL model in role config
        "granite-4.1-8b",
    ):
        assert mod.is_chat_capable_model_id(mid), mid


def test_expand_swarm_auto_caps_at_max_auto_entries() -> None:
    """``SWARM_CHAT_AUTO_MAX`` (default 3) prevents an N-loaded-model box
    from fanning every default-shaped swarm out to all N. Pin the cap
    semantics here so the LM Studio gateway's wrapper can rely on it.
    """
    from middle_layer.swarm import expand_swarm_models

    loaded = [f"model-{i}" for i in range(7)]
    expanded, err = expand_swarm_models(
        "auto", available=loaded, max_auto_entries=3
    )
    assert err is None
    assert expanded == loaded[:3]

    # Explicit ids alongside ``auto`` are always kept, even past the cap.
    expanded, err = expand_swarm_models(
        ["model-5", "auto"], available=loaded, max_auto_entries=2
    )
    assert err is None
    # ``model-5`` first (explicit), then the cap=2 auto entries skipping
    # already-seen model-5 → model-0, model-1.
    assert expanded == ["model-5", "model-0", "model-1"]

    # ``max_auto_entries=0`` / ``None`` disables the cap.
    expanded, err = expand_swarm_models(
        "auto", available=loaded, max_auto_entries=0
    )
    assert err is None
    assert expanded == loaded


def test_swarm_auto_pool_prefers_loaded_filters_embeddings(monkeypatch) -> None:
    """The LM Studio gateway's ``_swarm_auto_pool`` must prefer the
    truly-loaded probe (``/api/v0/models``), strip embedding ids, and only
    fall back to the installed probe when the loaded probe is empty.
    """
    mod = _load_middle_layer()

    # Loaded probe returns one chat + one embedding model. The pool must
    # drop the embedding.
    monkeypatch.setattr(
        mod,
        "get_loaded_lmstudio_model_ids",
        lambda force_refresh=False: (
            ["qwen3.5-122b-a10b", "text-embedding-nomic-embed-text-v1.5"],
            None,
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_lmstudio_model_ids",
        lambda force_refresh=False: (["should-not-be-used"], None),
    )
    pool, err = mod._swarm_auto_pool()
    assert err is None
    assert pool == ["qwen3.5-122b-a10b"]

    # When the loaded probe is empty (older LM Studio without /api/v0),
    # fall back to the installed probe — also filtered to chat ids.
    monkeypatch.setattr(
        mod,
        "get_loaded_lmstudio_model_ids",
        lambda force_refresh=False: ([], None),
    )
    monkeypatch.setattr(
        mod,
        "get_lmstudio_model_ids",
        lambda force_refresh=False: (
            ["qwen3.5-122b-a10b", "bge-base-en-v1.5", "granite-4.1-8b"],
            None,
        ),
    )
    pool, err = mod._swarm_auto_pool()
    assert err is None
    assert pool == ["qwen3.5-122b-a10b", "granite-4.1-8b"]


# ---------------------------------------------------------------------------
# Swarm-intelligence-effectiveness pass: ``first-success`` strategy must
# actually exit early on the first temporal success and cancel pending
# peers (not the previous "wait for all then pick input-order-first").
# ---------------------------------------------------------------------------


def test_fanout_early_exit_does_not_wait_for_slow_in_flight_peers() -> None:
    """The whole point of early-exit is wall-clock latency. With
    ``max_parallel`` ≥ swarm size, every agent starts at once; if the
    fast one finishes first, ``fanout`` must return immediately —
    *not* block on the slow peers (which would defeat the purpose,
    since the per-call timeout is 180s by default).

    We measure wall time: a fast agent that returns in ~10ms plus a slow
    agent that would take ~2s, with both running concurrently. Early
    exit must return in well under 2s.
    """
    import threading
    import time as _time
    from types import SimpleNamespace

    from middle_layer.swarm import fanout, SwarmDeps

    SLOW_SECONDS = 2.0

    def fake_chat(model_id, messages, **kwargs):
        if model_id == "fast":
            _time.sleep(0.01)
            return ({"choices": [{"message": {"content": "first!"}}]}, None)
        _time.sleep(SLOW_SECONDS)
        return ({"choices": [{"message": {"content": f"from {model_id}"}}]}, None)

    deps = SwarmDeps(
        chat_completion=fake_chat,
        anthropic_chat=None,
        resolve_model_id=lambda req, avail, loaded=None: (req, None),
        get_available_models=lambda: (["fast", "slow"], None),
        anthropic_default_model="x",
        extract_user_intent=lambda d: "",
        prefer_loaded_models=False,
    )
    # Both run concurrently — max_parallel=2 starts them at the same time
    # so the only thing that can save us from waiting ~SLOW_SECONDS is
    # actually exiting without joining the slow in-flight peer.
    specs = [{"model": "fast"}, {"model": "slow"}]
    t0 = _time.monotonic()
    results, err = fanout(
        specs,
        [{"role": "user", "content": "hi"}],
        {},
        deps,
        max_parallel=2,
        early_exit_on_first_success=True,
    )
    elapsed = _time.monotonic() - t0
    assert err is None
    successes = [r for r in results if r is not None and r["ok"]]
    assert len(successes) == 1
    assert successes[0]["model"] == "fast"
    # Generous bound: anything under half the slow agent's runtime means
    # we did not join it. Tight bound would be ~0.05s but CI jitter etc.
    assert elapsed < SLOW_SECONDS * 0.5, (
        f"fanout waited for slow in-flight peer: elapsed={elapsed:.2f}s "
        f"(should be <<{SLOW_SECONDS}s)"
    )


def test_fanout_early_exit_returns_on_first_success_and_skips_pending() -> None:
    """When ``early_exit_on_first_success=True`` and one agent returns
    ok+text, ``fanout`` must stop collecting and try to cancel pending
    peers. The returned list reflects only completed agents.
    """
    import threading
    import time as _time
    from types import SimpleNamespace

    from middle_layer.swarm import fanout, SwarmDeps

    started = []
    completion_gate = threading.Event()

    def fake_chat(model_id, messages, **kwargs):
        started.append(model_id)
        if model_id == "fast":
            return ({"choices": [{"message": {"content": "first!"}}]}, None)
        # Slow agents block until released; if early-exit cancels them
        # before they start, this gate is never even waited on.
        completion_gate.wait(timeout=2)
        return ({"choices": [{"message": {"content": f"from {model_id}"}}]}, None)

    deps = SwarmDeps(
        chat_completion=fake_chat,
        anthropic_chat=None,
        resolve_model_id=lambda req, avail, loaded=None: (req, None),
        get_available_models=lambda: (["fast", "slow1", "slow2"], None),
        anthropic_default_model="anthropic-default",
        extract_user_intent=lambda d: "",
        prefer_loaded_models=False,
    )

    # Force serial execution so ``fast`` completes first and the others
    # are still pending when we cancel. With max_parallel=1 the fast spec
    # listed first runs first; if it succeeds the executor never picks up
    # slow1/slow2 because the loop breaks before the next ``as_completed``.
    specs = [{"model": "fast"}, {"model": "slow1"}, {"model": "slow2"}]
    started.clear()
    results, err = fanout(
        specs,
        [{"role": "user", "content": "hi"}],
        {},
        deps,
        max_parallel=1,
        early_exit_on_first_success=True,
    )
    completion_gate.set()  # release any slow agents that did start
    assert err is None
    # At least the fast one ran; pending peers were cancelled before start.
    assert "fast" in started
    successes = [r for r in results if r is not None and r["ok"]]
    assert len(successes) == 1
    assert successes[0]["model"] == "fast"


def test_run_swarm_chat_completion_first_success_passes_early_exit(monkeypatch) -> None:
    """``strategy: first-success`` (and the ``fanout`` intent) must flip
    ``early_exit_on_first_success=True`` when calling ``swarm.fanout``.
    Pinned so a future refactor can't silently regress to wait-for-all.
    """
    from middle_layer import swarm as runner

    captured = {}

    def spy_fanout(specs, messages, common, deps, max_parallel=None, **kwargs):
        captured["early_exit"] = kwargs.get("early_exit_on_first_success", False)
        return [
            {
                "agent_id": "x",
                "model": "fake",
                "ok": True,
                "error": None,
                "error_kind": None,
                "http_status": None,
                "error_detail": None,
                "latency_ms": 1,
                "response": {"choices": [{"message": {"content": "ok"}}]},
                "text": "ok",
            }
        ], None

    monkeypatch.setattr(runner, "fanout", spy_fanout)

    fake_deps = runner.SwarmDeps(
        chat_completion=lambda *a, **k: ({}, None),
        anthropic_chat=None,
        resolve_model_id=lambda req, avail, loaded=None: (req, None),
        get_available_models=lambda: (["fake"], None),
        anthropic_default_model="x",
        extract_user_intent=lambda d: "",
        prefer_loaded_models=False,
    )

    for strategy in ("first-success", "fanout"):
        captured.clear()
        body, err, _ = runner.run_swarm_chat_completion(
            "swarm/fanout" if strategy == "fanout" else "swarmCouncil",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "swarm": {"models": ["fake"], "strategy": strategy},
            },
            fake_deps,
            intent="fanout" if strategy == "fanout" else "council",
        )
        assert err is None and body is not None
        assert captured["early_exit"] is True, strategy

    # ``best-of-n`` must NOT early-exit: the judge needs every candidate.
    captured.clear()
    body, err, _ = runner.run_swarm_chat_completion(
        "swarmCouncil",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "swarm": {"models": ["fake"], "strategy": "best-of-n"},
        },
        fake_deps,
        intent="council",
    )
    assert err is None and body is not None
    assert captured["early_exit"] is False


# ---------------------------------------------------------------------------
# Swarm "auto" sentinel expansion (MLX backend) + judge-verdict / pipeline
# template helpers — bundled into a single MLX subprocess to amortize the
# slow MLX bootstrap cost.
# ---------------------------------------------------------------------------


def test_mlx_swarm_helpers_matrix() -> None:
    """Verify the two helpers that silently degraded swarm quality before.

    Bundled into one MLX subprocess for the same reason as the expansion test.
    """
    snippet = """
    parse = mod._parse_judge_verdict
    subst = mod._substitute_pipeline_template

    verdict_cases = {
        "bare_letter":          parse("A", ["A", "B"]),
        "bare_letter_newline":  parse("A\\n", ["A", "B"]),
        "with_period":          parse("B.", ["A", "B"]),
        "letter_with_reason":   parse("B - because it is clearer.", ["A", "B"]),
        "bold_letter":          parse("**A** is best", ["A", "B"]),
        "bracketed":            parse("[B] wins", ["A", "B"]),
        "parenthesized":        parse("(C) is correct", ["A", "B", "C"]),
        "answer_is":            parse("The answer is A.", ["A", "B"]),
        "i_pick":               parse("I pick B because it cites sources.", ["A", "B"]),
        "winner_colon":         parse("Winner: C", ["A", "B", "C"]),
        "leading_prose":        parse("Looking at these,\\nB is best.", ["A", "B"]),
        "lowercase_input":      parse("answer is a", ["A", "B"]),
        "label_not_in_set":     parse("D is best", ["A", "B", "C"]),
        "empty":                parse("", ["A", "B"]),
        "none_verdict":         parse(None, ["A", "B"]),
        "no_labels":            parse("A", []),
        "verbose_pick_last":    parse(
            "Both A and B are good but I'll go with B for clarity.",
            ["A", "B"],
        ),
    }

    subst_cases = {
        "no_placeholders":     subst("hello world", {"previous": "x"}),
        "basic":               subst("see: {{previous}}", {"previous": "DRAFT"}),
        "missing_key_kept":    subst("see: {{ghost}}", {"previous": "x"}),
        "value_with_braces":   subst(
            "review:\\n{{previous}}",
            {"previous": "def f(): return {1: 2}"},
        ),
        "value_with_fstring":  subst(
            "{{previous}}",
            {"previous": 'print(f"x={value}")'},
        ),
        "named_step":          subst(
            "draft was: {{draft}}, review: {{review}}",
            {"draft": "D{x}", "review": "R}{"},
        ),
        "none_value":          subst("x={{previous}}", {"previous": None}),
        "non_string_input":    subst(None, {"previous": "x"}),
        "non_dict_ctx":        subst("{{previous}}", "not-a-dict"),
    }

    out = {"verdict": verdict_cases, "subst": subst_cases}
    print("RESULT=" + json.dumps(out))
    """
    result = _run_mlx_subprocess(snippet)

    verdict = result["verdict"]
    # Strict matches that must resolve to the right index.
    assert verdict["bare_letter"] == 0
    assert verdict["bare_letter_newline"] == 0
    assert verdict["with_period"] == 1
    assert verdict["letter_with_reason"] == 1
    assert verdict["bold_letter"] == 0
    assert verdict["bracketed"] == 1
    assert verdict["parenthesized"] == 2
    assert verdict["answer_is"] == 0
    assert verdict["i_pick"] == 1
    assert verdict["winner_colon"] == 2
    assert verdict["leading_prose"] == 1
    assert verdict["lowercase_input"] == 0
    # Cases where no decision should be made.
    assert verdict["label_not_in_set"] is None
    assert verdict["empty"] is None
    assert verdict["none_verdict"] is None
    assert verdict["no_labels"] is None
    # "Both A and B" picks the first occurrence (A). This is a known
    # limitation of last-resort \b matching; document it via the test.
    assert verdict["verbose_pick_last"] == 0

    subst = result["subst"]
    assert subst["no_placeholders"] == "hello world"
    assert subst["basic"] == "see: DRAFT"
    # Missing keys are preserved as literal placeholders (caller can detect).
    assert subst["missing_key_kept"] == "see: {{ghost}}"
    # The critical regression: braces in values must not break substitution.
    assert subst["value_with_braces"] == "review:\ndef f(): return {1: 2}"
    assert subst["value_with_fstring"] == 'print(f"x={value}")'
    assert subst["named_step"] == "draft was: D{x}, review: R}{"
    assert subst["none_value"] == "x="
    assert subst["non_string_input"] is None
    assert subst["non_dict_ctx"] == "{{previous}}"


def test_mlx_swarm_expansion_matrix() -> None:
    """One subprocess covers the four MLX expansion behaviors we care about.

    Bundling them keeps the slow MLX bootstrap to a single subprocess instead
    of paying that cost per case.
    """
    snippet = """
    fake_loaded_two = SimpleNamespace(
        get_loaded_aliases=lambda: ["alpha", "beta"],
        get_available_aliases=lambda: ["alpha", "beta", "gamma"],
    )
    fake_loaded_zero = SimpleNamespace(
        get_loaded_aliases=lambda: [],
        get_available_aliases=lambda: ["alpha", "beta"],
    )

    out = {}

    mod.mlx_manager = fake_loaded_two
    out["auto"] = mod._expand_swarm_models("auto")[0]
    out["mixed"] = mod._expand_swarm_models(["beta", "auto", "anthropic"])[0]
    out["available"] = mod._expand_swarm_models(["available"])[0]

    mod.mlx_manager = fake_loaded_zero
    out["loaded_falls_back"] = mod._expand_swarm_models(["loaded"])[0]

    print("RESULT=" + json.dumps(out))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["auto"] == ["alpha", "beta"]
    assert result["mixed"] == ["beta", "alpha", "anthropic"]
    assert result["available"] == ["alpha", "beta", "gamma"]
    assert result["loaded_falls_back"] == ["alpha", "beta"]
