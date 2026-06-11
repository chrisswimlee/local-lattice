"""Per-model serialization for swarm fanouts.

Pinned regression: LM Studio crashes large-context MoE models (observed
with qwen3.5-122b-a10b at 128k ctx) when fed two concurrent inference
jobs. The per-model semaphore must serialize agents that resolve to the
same id, while still letting agents on distinct ids run in parallel
under the global ``MAX_PARALLEL_MODEL_CALLS`` cap.
"""

from __future__ import annotations

import threading
import time as _time

from tests._helpers import _load_middle_layer


def test_per_model_semaphore_serializes_same_model_calls() -> None:
    """When two ``_run_one_agent`` calls resolve to the same LM Studio id,
    they must hold the per-model semaphore one-at-a-time. Regression:
    under the previous code a 3-way swarm with all specs resolving to a
    single loaded MoE crashed LM Studio because 2 inference jobs hit the
    same model in parallel.
    """
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
        # Force every spec to the same model so the per-model semaphore is
        # the only thing that can prevent overlap.
        mod.resolve_model_id = lambda req, avail, loaded=None: ("loaded-122b", None)
        mod._per_model_semaphores = {}
        mod.LM_STUDIO_PER_MODEL_INFLIGHT_CAP = 1
        specs = [
            {"model": "role:reasoner"},
            {"model": "role:coder"},
            {"model": "role:fast"},
        ]
        threads = []

        def run(spec):
            mod._run_one_agent(
                spec,
                [{"role": "user", "content": "hi"}],
                {},
                ["loaded-122b"],
                loaded=["loaded-122b"],
            )

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


def test_per_model_semaphore_lets_distinct_models_run_in_parallel() -> None:
    """The per-model serializer only affects same-model concurrency. Two
    agents resolving to different ids must still be able to overlap.
    """
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
            mod._run_one_agent(
                spec,
                [{"role": "user", "content": "hi"}],
                {},
                ["model-a", "model-b"],
            )

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
