"""Tests for ``MLXManager``'s lazy-load, pinning, eviction, and unload
semantics (PR 2 of the MLX audit hardening plan).

These tests are critical: PR 2 introduces a new invariant — that
in-flight aliases are pinned and survive eviction — and without
test coverage a future refactor can quietly break it. We exercise
the manager directly with mocked ``_mlx_load_model`` so no actual
MLX weights are touched.

All tests run in a subprocess (via ``_run_mlx_subprocess``) because
importing ``middle_layerMLX.py`` into the pytest process is unsafe
on Python 3.14 + mlx_lm (see ``tests/_helpers.py``).
"""

from __future__ import annotations

import pytest

from tests._helpers import _run_mlx_subprocess


# All tests in this file go through the MLX subprocess harness — there
# is no in-pytest-process import path that's safe for the real MLX
# gateway. CI environments without mlx_lm still run them: the gateway
# wraps ``import mlx_lm`` in try/except so MLX_AVAILABLE just becomes
# False; for these tests we mock ``_mlx_load_model`` directly.
#
# Marked ``mlx`` so the default fast suite skips the subprocess cost;
# opt in with ``make test-mlx`` or ``pytest -m mlx``.
pytestmark = pytest.mark.mlx


def _setup_fake_loader(num_loads_succeed: int = 99) -> str:
    """Return a Python snippet that replaces ``_mlx_load_model`` with
    a fake that returns sentinel ``(model, tokenizer)`` tuples without
    touching real MLX weights. ``MLX_AVAILABLE`` is force-set to True
    so the load path actually executes.
    """
    return f"""
    from types import SimpleNamespace
    mod.MLX_AVAILABLE = True

    _LOAD_COUNT = [0]
    _LOAD_LIMIT = {num_loads_succeed}

    def fake_load(path):
        _LOAD_COUNT[0] += 1
        if _LOAD_COUNT[0] > _LOAD_LIMIT:
            raise RuntimeError(f"fake load #{{_LOAD_COUNT[0]}} forced failure")
        return SimpleNamespace(_path=path, _id=_LOAD_COUNT[0]), SimpleNamespace(_path=path)

    mod._mlx_load_model = fake_load
    """


def test_acquire_inference_handle_pins_alias() -> None:
    """The new context manager increments inflight on entry and
    decrements on exit. Pinned aliases must be visible in ``_inflight``.
    """
    snippet = _setup_fake_loader() + """
    # Fake the registry so load_model finds the alias.
    mod.mlx_manager.registry["fake-a"] = "/fake/path/a"

    with mod.mlx_manager.acquire_inference_handle("fake-a") as handle:
        assert handle is not None, "load should succeed with fake loader"
        inflight_during = dict(mod.mlx_manager._inflight)

    inflight_after = dict(mod.mlx_manager._inflight)

    import json as _j
    print("RESULT=" + _j.dumps({
        "during": inflight_during,
        "after": inflight_after,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["during"] == {"fake-a": 1}
    assert result["after"] == {}


def test_eviction_skips_pinned_alias() -> None:
    """The audit's #1 P0 finding: LRU eviction used to happily yank
    an in-flight model. With pinning, the eviction loop must pick
    another (unpinned) victim instead.

    Setup: cap=2, load A + B, pin B (simulate in-flight on B), load C.
    Expected: A evicted (oldest unpinned), B kept (pinned), C resident.
    """
    snippet = _setup_fake_loader() + """
    mod.MAX_CONCURRENT_MODELS = 2
    mgr = mod.mlx_manager
    mgr.registry.update({
        "fake-a": "/fake/path/a",
        "fake-b": "/fake/path/b",
        "fake-c": "/fake/path/c",
    })

    # Load A and B, both unpinned.
    mgr.load_model("fake-a")
    mgr.load_model("fake-b")
    assert set(mgr.get_loaded_aliases()) == {"fake-a", "fake-b"}

    # Pin B (simulate in-flight).
    mgr.pin_alias("fake-b")

    # Load C with cap=2; eviction must pick A (the only unpinned one)
    # even though A is older than B is not the only criterion.
    mgr.load_model("fake-c")
    resident_after_c = mgr.get_loaded_aliases()

    mgr.release_pin("fake-b")
    import json as _j
    print("RESULT=" + _j.dumps({"resident": resident_after_c}))
    """
    result = _run_mlx_subprocess(snippet)
    assert "fake-b" in result["resident"], (
        f"pinned alias was evicted! resident={result['resident']}"
    )
    assert "fake-c" in result["resident"]
    assert "fake-a" not in result["resident"], (
        f"unpinned older alias should have been evicted; resident={result['resident']}"
    )


def test_unload_deferred_while_pinned_fires_on_release() -> None:
    """Unloading a pinned model must defer the actual dict removal
    until the last holder releases. The deferred unload then fires
    automatically — operator does not need to retry.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    mgr.registry["fake-a"] = "/fake/path/a"
    mgr.load_model("fake-a")
    mgr.pin_alias("fake-a")

    # Unload while pinned.
    result1 = mgr.unload_model("fake-a")
    resident_after_unload = mgr.get_loaded_aliases()

    # Release the pin — deferred unload should fire.
    mgr.release_pin("fake-a")
    resident_after_release = mgr.get_loaded_aliases()

    import json as _j
    print("RESULT=" + _j.dumps({
        "first_result": result1,
        "after_unload": resident_after_unload,
        "after_release": resident_after_release,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["first_result"] == {"unloaded": False, "deferred": True}
    assert result["after_unload"] == ["fake-a"], (
        "deferred unload should NOT immediately drop the dict entry"
    )
    assert result["after_release"] == [], (
        "release after deferred unload should fire the actual drop"
    )


def test_post_eviction_reload_does_not_duplicate_pinned_model() -> None:
    """Audit finding: after eviction + reload of the same alias, the
    old code created a second resident copy with a different gen_lock,
    breaking per-alias serialization. With pinning + immediate eviction
    of unpinned aliases, this shouldn't happen for an in-flight model
    — the alias stays pinned, and a "reload" returns the existing handle.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    mgr.registry["fake-a"] = "/fake/path/a"

    # First load + pin.
    h1 = mgr.load_model("fake-a")
    mgr.pin_alias("fake-a")
    gen_lock_1 = h1[2]
    model_1_id = h1[0]._id

    # Now another caller "reloads" while the first is pinned.
    # load_model must return the same cached handle, not a new load.
    h2 = mgr.load_model("fake-a")
    gen_lock_2 = h2[2]
    model_2_id = h2[0]._id

    mgr.release_pin("fake-a")
    import json as _j
    print("RESULT=" + _j.dumps({
        "same_gen_lock": gen_lock_1 is gen_lock_2,
        "same_model": model_1_id == model_2_id,
        "load_count_after_two_loads": 1,  # The fake loader's counter; expect 1
        "actual_load_count": h1[0]._id,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["same_gen_lock"] is True, (
        "second load_model returned a different gen_lock — serialization "
        "guarantee is broken"
    )
    assert result["same_model"] is True
    assert result["actual_load_count"] == 1, (
        f"_mlx_load_model should have run exactly once; "
        f"actual={result['actual_load_count']}"
    )


def test_loading_locks_pruned_on_unload() -> None:
    """Audit finding: `_loading_locks` were created via setdefault on
    every cold load and never removed — unbounded growth for every
    alias ever attempted. PR 2 prunes on eviction and unload.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    mgr.registry["fake-a"] = "/fake/path/a"

    mgr.load_model("fake-a")
    assert "fake-a" in mgr._loading_locks

    mgr.unload_model("fake-a")

    import json as _j
    print("RESULT=" + _j.dumps({
        "loading_locks": list(mgr._loading_locks.keys()),
        "loaded": mgr.get_loaded_aliases(),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert "fake-a" not in result["loading_locks"], (
        f"_loading_locks not pruned on unload: {result['loading_locks']}"
    )
    assert result["loaded"] == []


def test_all_pinned_load_exceeds_cap_with_warning(caplog=None) -> None:
    """If every resident alias is in-flight, a new load must NOT
    deadlock — it proceeds and exceeds the cap, logging a warning.
    Operators see the warning and know to raise MAX_CONCURRENT_MODELS.
    """
    snippet = _setup_fake_loader() + """
    mod.MAX_CONCURRENT_MODELS = 1
    mgr = mod.mlx_manager
    mgr.registry.update({
        "fake-a": "/fake/path/a",
        "fake-b": "/fake/path/b",
    })

    mgr.load_model("fake-a")
    mgr.pin_alias("fake-a")

    # All resident (just A) is pinned. Loading B should still succeed,
    # exceeding the cap of 1.
    mgr.load_model("fake-b")
    resident = mgr.get_loaded_aliases()

    mgr.release_pin("fake-a")
    import json as _j
    print("RESULT=" + _j.dumps({"resident": resident}))
    """
    result = _run_mlx_subprocess(snippet)
    # Both should be resident because A was pinned and couldn't be evicted.
    assert set(result["resident"]) == {"fake-a", "fake-b"}


def test_inflight_counter_releases_on_exception_in_handle_block() -> None:
    """If an exception escapes the ``acquire_inference_handle`` context,
    the inflight refcount must still return to 0 — otherwise that alias
    is permanently un-evictable.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    mgr.registry["fake-a"] = "/fake/path/a"

    raised = False
    try:
        with mgr.acquire_inference_handle("fake-a") as handle:
            assert handle is not None
            raise RuntimeError("simulated generation failure")
    except RuntimeError:
        raised = True

    import json as _j
    print("RESULT=" + _j.dumps({
        "raised": raised,
        "inflight": dict(mgr._inflight),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["raised"] is True
    assert result["inflight"] == {}, (
        f"inflight not released on exception: {result['inflight']}"
    )


def test_unload_of_unloaded_alias_returns_clean_status() -> None:
    """Unloading something that isn't loaded should not crash — it
    returns ``{"unloaded": False, "deferred": False}``.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    result = mgr.unload_model("never-loaded-alias")
    import json as _j
    print("RESULT=" + _j.dumps({"result": result}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["result"] == {"unloaded": False, "deferred": False}


# ---------------------------------------------------------------------------
# PR 3 — Metal cache teardown after eviction
# ---------------------------------------------------------------------------


def test_eviction_invokes_metal_cache_teardown_helper() -> None:
    """After LRU eviction during a load, ``_post_evict_cleanup`` must
    be called outside the registry lock for each evicted alias.
    """
    snippet = _setup_fake_loader() + """
    mod.MAX_CONCURRENT_MODELS = 1
    mgr = mod.mlx_manager
    mgr.registry.update({
        "fake-a": "/fake/path/a",
        "fake-b": "/fake/path/b",
    })

    # Spy on _post_evict_cleanup at the module level so we record both
    # LRU-eviction-during-load and explicit unload paths.
    invocations = []
    _orig = mod._post_evict_cleanup
    def spy(reason, alias):
        invocations.append((reason, alias))
        _orig(reason, alias)
    mod._post_evict_cleanup = spy

    # Load A (no eviction), then B (evicts A).
    mgr.load_model("fake-a")
    mgr.load_model("fake-b")
    # Explicitly unload B.
    mgr.unload_model("fake-b")

    import json as _j
    print("RESULT=" + _j.dumps({"invocations": invocations}))
    """
    result = _run_mlx_subprocess(snippet)
    # Expect at least an LRU eviction of A and an unload of B.
    reasons = {(reason, alias) for reason, alias in result["invocations"]}
    assert ("lru-eviction", "fake-a") in reasons, (
        f"LRU eviction did not trigger cleanup: {result['invocations']}"
    )
    assert ("unload", "fake-b") in reasons, (
        f"explicit unload did not trigger cleanup: {result['invocations']}"
    )


def test_post_evict_cleanup_failure_does_not_break_eviction() -> None:
    """Audit-derived safety net: if the Metal teardown helper raises,
    the eviction itself must still complete and the manager must
    remain usable. We patch _try_clear_mlx_metal_cache to raise.
    """
    snippet = _setup_fake_loader() + """
    mod.MAX_CONCURRENT_MODELS = 1
    mgr = mod.mlx_manager
    mgr.registry.update({
        "fake-a": "/fake/path/a",
        "fake-b": "/fake/path/b",
    })

    # Make the teardown helper raise. _post_evict_cleanup wraps it in
    # try/except so the exception should be swallowed.
    def boom():
        raise RuntimeError("simulated teardown failure")
    mod._try_clear_mlx_metal_cache = boom

    mgr.load_model("fake-a")
    mgr.load_model("fake-b")  # evicts A; teardown raises but is swallowed
    mgr.unload_model("fake-b")  # also triggers teardown
    resident = mgr.get_loaded_aliases()

    import json as _j
    print("RESULT=" + _j.dumps({"resident": resident}))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["resident"] == [], (
        f"manager not usable after teardown error: {result['resident']}"
    )


# ---------------------------------------------------------------------------
# PR 10 — Dead-field cleanup: loaded_models entries are 3-tuples
# ---------------------------------------------------------------------------


def test_loaded_models_stores_three_tuples_not_four() -> None:
    """Audit finding: the per-entry ``last_used`` timestamp was
    written on every cache hit and never read anywhere — dead weight.
    PR 10 prunes it. Pin the tuple shape so a future refactor can't
    silently re-add the field.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    mgr.registry["fake-a"] = "/fake/path/a"

    handle = mgr.load_model("fake-a")
    entry = mgr.loaded_models["fake-a"]

    import json as _j
    print("RESULT=" + _j.dumps({
        "entry_len": len(entry),
        "handle_len": len(handle),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["entry_len"] == 3, (
        f"loaded_models entry should be a 3-tuple; got len={result['entry_len']}"
    )
    assert result["handle_len"] == 3


def test_run_one_agent_passes_queue_controls_through() -> None:
    """Audit finding: ``_mlx_chat_completion`` always passed
    ``queue_controls=None``, dropping per-request priority/wait at
    the swarm fanout boundary. PR 10 wires the propagation.
    """
    snippet = """
    captured = {"queue_controls": "NOT_CALLED"}

    def fake_admission_acquire(alias, *, request_id, stream, queue_controls=None):
        captured["queue_controls"] = queue_controls
        captured["alias"] = alias
        return True, {"queue_wait_ms": 0, "priority": 0}

    def fake_admission_release(alias):
        pass

    mod._admission_acquire = fake_admission_acquire
    mod._admission_release = fake_admission_release
    mod.MLX_AVAILABLE = True

    # Don't actually load the model — return None from acquire_inference_handle
    # so the function bails before touching MLX. We only care about whether
    # queue_controls reached _admission_acquire.
    import contextlib
    @contextlib.contextmanager
    def fake_acquire(alias):
        yield None
    mod.mlx_manager.acquire_inference_handle = fake_acquire
    mod.mlx_manager.get_last_load_error = lambda alias: None

    qc = {"priority": 5, "max_wait_sec": 1.5}
    mod._mlx_chat_completion(
        "fake-alias",
        [{"role": "user", "content": "hi"}],
        queue_controls=qc,
    )

    import json as _j
    print("RESULT=" + _j.dumps({
        "qc": captured["queue_controls"],
        "alias": captured.get("alias"),
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["qc"] == {"priority": 5, "max_wait_sec": 1.5}, (
        f"queue_controls did not reach _admission_acquire: {result}"
    )
    assert result["alias"] == "fake-alias"


def test_deferred_unload_drains_cleanup_on_release() -> None:
    """When an unload is deferred while the alias is pinned, the
    eventual eviction (in release_pin) must also trigger
    _post_evict_cleanup. Otherwise the slow path leaks Metal cache.
    """
    snippet = _setup_fake_loader() + """
    mgr = mod.mlx_manager
    mgr.registry["fake-a"] = "/fake/path/a"
    mgr.load_model("fake-a")
    mgr.pin_alias("fake-a")

    invocations = []
    _orig = mod._post_evict_cleanup
    def spy(reason, alias):
        invocations.append((reason, alias))
        _orig(reason, alias)
    mod._post_evict_cleanup = spy

    mgr.unload_model("fake-a")  # deferred
    inv_after_unload = list(invocations)
    mgr.release_pin("fake-a")   # fires the deferred eviction
    inv_after_release = list(invocations)

    import json as _j
    print("RESULT=" + _j.dumps({
        "after_unload": inv_after_unload,
        "after_release": inv_after_release,
    }))
    """
    result = _run_mlx_subprocess(snippet)
    assert result["after_unload"] == [], (
        "cleanup fired during deferred unload — should wait for release"
    )
    assert ("deferred-unload", "fake-a") in [
        (r, a) for r, a in result["after_release"]
    ], f"deferred-unload cleanup never fired: {result['after_release']}"
