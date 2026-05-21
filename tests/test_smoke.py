"""Minimal smoke tests so ``make test`` is meaningful before Pass 5."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_middle_layer():
    path = REPO_ROOT / "middle_layer.py"
    spec = importlib.util.spec_from_file_location("middle_layer_root_smoke", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_mlx_subprocess(snippet: str) -> dict:
    """Run a tiny script in a fresh interpreter that loads ``middle_layerMLX.py``.

    Isolates MLX init and teardown from the pytest process (Python 3.14 + the
    MLX library segfault during interpreter shutdown when both Flask and MLX
    threads are torn down in-process). The snippet must end by printing one
    JSON line to stdout starting with ``RESULT=``.
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


def test_placeholder_models_include_core_ids() -> None:
    mod = _load_middle_layer()
    assert "auto" in mod.PLACEHOLDER_MODELS
    assert "" in mod.PLACEHOLDER_MODELS


# ---------------------------------------------------------------------------
# Resolver: PREFER_LOADED_MODELS keeps swarm fanouts off cold/JIT giants
# ---------------------------------------------------------------------------


def test_resolve_role_prefers_loaded_id_over_not_loaded() -> None:
    """Regression: when LM Studio reports both a loaded small model and a
    not-loaded giant that also matches a role preference, the resolver must
    pick the loaded one. This is the exact failure mode that produced the
    'all swarm agents failed: ... insufficient system resources' cascade.
    """
    mod = _load_middle_layer()
    # Mimic the user's actual /api/v0/models snapshot: 122B is loaded, 70B is
    # downloaded but not loaded. Both match the reasoner role pattern "70b"
    # / "qwen3.5-122b" via substring; without prefer-loaded we'd pick whichever
    # appears first in the role pref list.
    available = [
        "qwen3.5-122b-a10b",
        "nousresearch/hermes-4-70b",
        "qwen/qwen3-coder-next",
    ]
    loaded = ["qwen3.5-122b-a10b"]
    saved_roles, saved_pref = mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS
    try:
        # Reasoner pref puts the not-loaded giant FIRST so a naive resolver
        # would pick it; prefer-loaded must override that ordering.
        mod.MODEL_ROLES = {
            "reasoner": [
                "nousresearch/hermes-4-70b",
                "qwen3.5-122b-a10b",
            ],
            "coder": ["qwen/qwen3-coder-next"],
            "fast": ["qwen3.5-122b-a10b"],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = True
        mid = mod._resolve_role("reasoner", available, loaded=loaded)
        assert mid == "qwen3.5-122b-a10b", (
            f"prefer-loaded should keep us on the loaded id, got {mid!r}"
        )
        rid, err = mod.resolve_model_id("role:reasoner", available, loaded=loaded)
        assert err is None and rid == "qwen3.5-122b-a10b"
    finally:
        mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS = saved_roles, saved_pref


def test_resolve_role_falls_back_to_not_loaded_when_no_loaded_match() -> None:
    """If nothing in the loaded subset matches the role list, resolver may
    still return a not-loaded id (LM Studio will JIT it). We deliberately
    keep this fallback so callers asking for a specific 'role:coder' on a
    machine with no coder loaded still work.
    """
    mod = _load_middle_layer()
    available = ["qwen3.5-122b-a10b", "qwen/qwen3-coder-next"]
    loaded = ["qwen3.5-122b-a10b"]
    saved_roles, saved_pref = mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS
    try:
        mod.MODEL_ROLES = {
            "coder": ["qwen3-coder-next"],
            "reasoner": ["qwen3.5-122b-a10b"],
            "fast": [],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = True
        # Loaded set has no coder, so we should fall back to the not-loaded one.
        mid = mod._resolve_role("coder", available, loaded=loaded)
        assert mid == "qwen/qwen3-coder-next"
    finally:
        mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS = saved_roles, saved_pref


def test_prefer_loaded_disabled_uses_first_match() -> None:
    """With PREFER_LOADED_MODELS off, we keep the legacy 'first match in pref
    list wins' behavior. Important for users who explicitly want JIT.
    """
    mod = _load_middle_layer()
    available = [
        "qwen3.5-122b-a10b",
        "nousresearch/hermes-4-70b",
    ]
    loaded = ["qwen3.5-122b-a10b"]
    saved_roles, saved_pref = mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS
    try:
        mod.MODEL_ROLES = {
            "reasoner": ["nousresearch/hermes-4-70b", "qwen3.5-122b-a10b"],
            "coder": [],
            "fast": [],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = False
        mid = mod._resolve_role("reasoner", available, loaded=loaded)
        assert mid == "nousresearch/hermes-4-70b"
    finally:
        mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS = saved_roles, saved_pref


# ---------------------------------------------------------------------------
# Role-file auto-discovery (regression: the running server kept using
# in-code defaults because MODEL_ROLES_FILE wasn't exported)
# ---------------------------------------------------------------------------


def test_autodiscover_finds_lmstudio_roles_next_to_script() -> None:
    """If lmstudio_roles.json exists next to middle_layer.py, _load_model_roles
    must pick it up even when neither MODEL_ROLES_JSON nor MODEL_ROLES_FILE
    is set. Prevents the 'process never picked up the new file' regression.
    """
    mod = _load_middle_layer()
    found = mod._autodiscover_roles_file()
    # In this repo lmstudio_roles.json sits at the root, so discovery must
    # succeed (not None) and prefer it over mlx_roles.json when both exist.
    assert found is not None and found.endswith("lmstudio_roles.json"), found


# ---------------------------------------------------------------------------
# Per-model serialization (LM Studio crashes 128k-ctx MoE under 2 concurrent
# inference jobs; same-model swarm agents must serialize)
# ---------------------------------------------------------------------------


def test_per_model_semaphore_serializes_same_model_calls() -> None:
    """When two ``_run_one_agent`` calls resolve to the same LM Studio id,
    they must hold the per-model semaphore one-at-a-time. Regression: under
    the previous code a 3-way swarm with all specs resolving to a single
    loaded MoE crashed LM Studio because 2 inference jobs hit the same model
    in parallel.
    """
    import threading
    import time as _time

    mod = _load_middle_layer()

    inflight = {"now": 0, "max": 0}
    inflight_lock = threading.Lock()
    bumps = []

    def fake_chat(model_id, messages, **kwargs):
        # Record concurrency the moment we "enter" the LM Studio call. The
        # semaphore is held outside this function, so if anything > 1 ever
        # appears here we know serialization is broken.
        with inflight_lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
            bumps.append(inflight["now"])
        _time.sleep(0.05)
        with inflight_lock:
            inflight["now"] -= 1
        return ({"choices": [{"message": {"content": f"hi from {model_id}"}}]}, None)

    saved_chat = mod._lmstudio_chat_completion
    saved_resolve = mod.resolve_model_id
    saved_sems = dict(mod._per_model_semaphores)
    saved_cap = mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP
    try:
        mod._lmstudio_chat_completion = fake_chat
        # Force every spec to the same model so the per-model semaphore is the
        # only thing that can prevent overlap.
        mod.resolve_model_id = lambda req, avail, loaded=None: ("loaded-122b", None)
        mod._per_model_semaphores = {}
        mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP = 1
        specs = [{"model": "role:reasoner"}, {"model": "role:coder"}, {"model": "role:fast"}]
        threads = []

        def run(spec):
            mod._run_one_agent(spec, [{"role": "user", "content": "hi"}], {}, ["loaded-122b"], loaded=["loaded-122b"])

        for s in specs:
            t = threading.Thread(target=run, args=(s,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), "agent thread hung; serialization deadlocked?"
        # Critical assertion: never more than 1 concurrent same-model inference.
        assert inflight["max"] == 1, (
            f"serialization broken: inflight peaked at {inflight['max']} (samples={bumps})"
        )
        # Sanity: all three actually ran.
        assert len(bumps) == 3
    finally:
        mod._lmstudio_chat_completion = saved_chat
        mod.resolve_model_id = saved_resolve
        mod._per_model_semaphores = saved_sems
        mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP = saved_cap


# ---------------------------------------------------------------------------
# Swarm chat meta-model intent map (canonical names + deprecated aliases)
# ---------------------------------------------------------------------------


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
    negotiable #1. To keep the warning useful, only fire once per alias per
    process (otherwise busy callers spam stderr).
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
    """The X-Swarm-Canonical-Name header is set only when the request used a
    non-canonical alias. swarm/fanout is itself canonical for the fanout
    intent, so its canonical_name must be 'swarm/fanout' (not 'swarmCouncil').
    Otherwise the route handler would falsely flag it as deprecated.
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
    """Without an explicit swarm.strategy, swarm/fanout must use strategy=fanout
    (no judge). Confirmed by patching _fanout to capture the strategy that
    actually drives the pick logic.
    """
    mod = _load_middle_layer()
    captured = {}

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

    # The IO runners now live in middle_layer.swarm; patch there so the
    # gateway's thin _fanout wrapper still hits our fake. This is a deliberate
    # ergonomic shift — Pass-3 split swarm IO into a shared module so MLX can
    # reuse it. ``_swarm_runner`` is the in-tree alias for that module.
    runner = mod._swarm_runner
    saved_fanout = runner.fanout
    saved_swarm_default = runner.SWARM_CHAT_DEFAULT_MODELS
    saved_mod_default = mod.SWARM_CHAT_DEFAULT_MODELS

    def fake_fanout_with_deps(specs, messages, common, deps, max_parallel=None):
        # Adapter: swarm.run_swarm_chat_completion calls swarm.fanout with a
        # ``deps`` positional, but the legacy fake_fanout signature doesn't
        # take it. Drop it and forward.
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
        # strategy=fanout means winner is FIRST success (not longest).
        assert body["swarm"]["strategy"] == "fanout"
        assert body["swarm"]["winner"] == "fake-a"
        assert body["swarm"]["rationale"].startswith("fanout")
    finally:
        runner.fanout = saved_fanout
        runner.SWARM_CHAT_DEFAULT_MODELS = saved_swarm_default
        mod.SWARM_CHAT_DEFAULT_MODELS = saved_mod_default


# ---------------------------------------------------------------------------
# Structured swarm error classification
# ---------------------------------------------------------------------------


def test_swarm_error_classifier_matrix() -> None:
    """Pin the classifier mapping. Adding a new error_kind should also add a
    case here so the openclaw runtime never sees a silently-renamed enum.
    """
    mod = _load_middle_layer()

    # (input error string, expected kind)
    cases = [
        # The two failure modes that bit us in production:
        (
            'LM Studio 400: {"error":{"message":"Failed to load model \\"x\\". '
            "Error: Model loading was stopped due to insufficient system "
            'resources."}}',
            "oom",
        ),
        (
            'LM Studio 400: {"error":"The model has crashed without additional '
            'information. (Exit code: null)"}',
            "model_crashed",
        ),
        # Other LM Studio buckets:
        ("LM Studio 500: <!DOCTYPE html><pre>Internal Server Error</pre>", "upstream_5xx"),
        ("LM Studio 404: not found", "upstream_4xx"),
        ("LM Studio 503: service unavailable", "upstream_5xx"),
        # Resolver-level:
        ("No model is loaded in LM Studio.", "no_models_loaded"),
        ("No loaded LM Studio model matched 'role:reasoner'. Available: []", "model_not_resolved"),
        (
            "swarm.models expanded to an empty set: no models are loaded in "
            "LM Studio. Load at least one model.",
            "no_models_loaded",
        ),
        # Transport:
        ("Timeout calling LM Studio", "timeout"),
        ("Cannot connect to LM Studio. Is it running?", "connection_error"),
        # Cloud adapters:
        ("Anthropic 401: invalid api key", "upstream_4xx"),
        ("Anthropic 500: server error", "upstream_5xx"),
        ("Anthropic error: ConnectionResetError", "anthropic_error"),
        ("LiteLLM error: SomeUpstreamFailure", "litellm_error"),
        # Config:
        ("ANTHROPIC_API_KEY not set", "config_error"),
        ("LiteLLM not available: import failed", "config_error"),
        # Fallback:
        ("Some weird untagged failure", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ]
    for err, expected in cases:
        got = mod._classify_swarm_error(err)
        assert got == expected, (
            f"_classify_swarm_error({err!r}) = {got!r}, expected {expected!r}"
        )
        # Every classification must stay inside the documented enum, even on
        # fallback. Caller contract is "treat unknown values as unknown".
        assert got in mod._SWARM_ERROR_KINDS


def test_swarm_error_classifier_extracts_upstream_status() -> None:
    """The HTTP status hidden in 'LM Studio NNN: ...' / 'Anthropic NNN: ...'
    error strings is what callers want to dispatch on. Make sure the helper
    pulls it out of every prefix shape we emit.
    """
    mod = _load_middle_layer()
    assert mod._extract_upstream_status("LM Studio 400: oom") == 400
    assert mod._extract_upstream_status("LM Studio 502: bad gateway") == 502
    assert mod._extract_upstream_status("Anthropic 429: rate limited") == 429
    assert mod._extract_upstream_status("LM Studio models endpoint returned 503") == 503
    assert mod._extract_upstream_status("Timeout calling LM Studio") is None
    assert mod._extract_upstream_status("") is None
    assert mod._extract_upstream_status(None) is None


def test_swarm_error_strip_upstream_prefix() -> None:
    """error_detail must be the upstream payload (no LM Studio NNN: prefix)
    so callers can show it directly without re-parsing.
    """
    mod = _load_middle_layer()
    assert (
        mod._strip_upstream_prefix("LM Studio 400: model could not load")
        == "model could not load"
    )
    assert (
        mod._strip_upstream_prefix("Anthropic 429:    rate limited\n")
        == "rate limited"
    )
    # No prefix → return as-is.
    assert mod._strip_upstream_prefix("Timeout calling LM Studio") == "Timeout calling LM Studio"


def test_summarize_failed_candidates_shape() -> None:
    """The structured all-failed body must include per-agent rows, an
    aggregate kind histogram, and an aggregate upstream-status histogram.
    Openclaw pins against this shape.
    """
    mod = _load_middle_layer()
    candidates = [
        {
            "agent_id": "role:reasoner",
            "model": "qwen3.5-122b-a10b",
            "ok": False,
            "error": "LM Studio 400: insufficient system resources",
            "error_kind": "oom",
            "http_status": 400,
            "error_detail": "insufficient system resources",
            "latency_ms": 120,
        },
        {
            "agent_id": "role:coder",
            "model": "qwen/qwen3-coder-next",
            "ok": False,
            "error": "LM Studio 400: insufficient system resources",
            "error_kind": "oom",
            "http_status": 400,
            "error_detail": "insufficient system resources",
            "latency_ms": 90,
        },
        {
            "agent_id": "role:fast",
            "model": "?",
            "ok": False,
            "error": "Timeout calling LM Studio",
            "error_kind": "timeout",
            "http_status": None,
            "error_detail": "Timeout calling LM Studio",
            "latency_ms": 180000,
        },
    ]
    body = mod._summarize_failed_candidates(candidates)
    assert body["summary"] == "all swarm agents failed"
    assert body["agent_count"] == 3
    assert body["kinds"] == {"oom": 2, "timeout": 1}
    assert body["upstream_statuses"] == {400: 2}
    assert len(body["agents"]) == 3
    # Per-agent rows must NOT include the heavy `response` payload (PII /
    # token-leak risk), only the metadata callers need.
    for row in body["agents"]:
        assert "response" not in row
        assert row["ok"] is False
        assert row["error_kind"] in mod._SWARM_ERROR_KINDS
        assert row["agent_id"] is not None


def test_summarize_uniform_empty_response_emits_actionable_summary() -> None:
    """When every candidate is an empty_response (reasoning models eating the
    whole token budget), the summary should tell the caller to raise
    max_tokens / inspect reasoning_content rather than blame the upstream.
    Otherwise openclaw would surface ``all swarm agents failed`` for what is
    actually a config issue.
    """
    mod = _load_middle_layer()
    candidates = [
        {"agent_id": f"role:r{i}", "model": "qwen3.5-122b-a10b",
         "ok": False, "error": "empty assistant content",
         "error_kind": "empty_response", "http_status": None,
         "error_detail": "upstream returned 200 with empty assistant content",
         "latency_ms": 1200}
        for i in range(3)
    ]
    body = mod._summarize_failed_candidates(candidates)
    assert body["kinds"] == {"empty_response": 3}
    assert "max_tokens" in body["summary"], body["summary"]
    assert "reasoning_content" in body["summary"], body["summary"]


def test_per_model_semaphore_lets_distinct_models_run_in_parallel() -> None:
    """The per-model serializer only affects same-model concurrency. Two
    agents resolving to different ids must still be able to overlap.
    """
    import threading
    import time as _time

    mod = _load_middle_layer()

    inflight = {"now": 0, "max": 0}
    inflight_lock = threading.Lock()

    def fake_chat(model_id, messages, **kwargs):
        with inflight_lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        _time.sleep(0.10)
        with inflight_lock:
            inflight["now"] -= 1
        return ({"choices": [{"message": {"content": "ok"}}]}, None)

    saved_chat = mod._lmstudio_chat_completion
    saved_resolve = mod.resolve_model_id
    saved_sems = dict(mod._per_model_semaphores)
    saved_cap = mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP
    try:
        mod._lmstudio_chat_completion = fake_chat
        # Each spec resolves to a DIFFERENT id, so the semaphores are independent.
        ids = iter(["model-a", "model-b"])
        mod.resolve_model_id = lambda req, avail, loaded=None: (next(ids), None)
        mod._per_model_semaphores = {}
        mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP = 1
        threads = []

        def run(spec):
            mod._run_one_agent(spec, [{"role": "user", "content": "hi"}], {}, ["model-a", "model-b"])

        for s in [{"model": "role:a"}, {"model": "role:b"}]:
            t = threading.Thread(target=run, args=(s,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()
        # Distinct models => parallelism allowed.
        assert inflight["max"] >= 2, (
            f"distinct models should run in parallel, max={inflight['max']}"
        )
    finally:
        mod._lmstudio_chat_completion = saved_chat
        mod.resolve_model_id = saved_resolve
        mod._per_model_semaphores = saved_sems
        mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP = saved_cap


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
# Swarm "auto" sentinel expansion (MLX backend)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Swarm helper unit tests (judge-verdict parser + pipeline templating)
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
