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
