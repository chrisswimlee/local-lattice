"""End-to-end swarm route tests against the in-process Flask test client.

These cover Layer 1 (plumbing) and Layer 2 (configuration sanity) for the
``/swarm/*`` endpoints:

* ``/swarm/fanout``  — N parallel candidates, real wall-clock parallelism.
* ``/swarm/vote``    — judge picks via ``_parse_judge_verdict``; verifies
  both a clean pick and the unparseable-verdict fallback path.
* ``/swarm/pipeline`` — ``{{previous}}``/``{{step_name}}`` substitution
  survives stage outputs that contain literal ``{`` / ``}``.
* ``/swarm/debate``  — round 2 agents actually see the round 1 transcript.
* ``/swarm/models``  — shape sanity (max_parallel, roles, defaults).

The chat backend is stubbed inside the subprocess so the test is
deterministic and CI-cheap. Real-model coverage belongs in
``scripts/swarm_eval.py`` (Layer 3).

The subprocess pattern matches ``tests/test_smoke.py``: MLX init/teardown
in the pytest process segfaults on Python 3.14, so we exec in a fresh
interpreter and return one ``RESULT=`` JSON line.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_swarm_subprocess(snippet: str) -> dict:
    bootstrap = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import time
        from types import SimpleNamespace

        spec = importlib.util.spec_from_file_location(
            "middle_layer_mlx_e2e", r"{REPO_ROOT / 'middle_layerMLX.py'}"
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
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"swarm e2e subprocess failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT="):
            return json.loads(line[len("RESULT=") :])
    raise AssertionError(f"no RESULT= line in stdout:\n{proc.stdout}")


# ---------------------------------------------------------------------------
# All e2e cases share one subprocess bootstrap so the (fairly expensive)
# module import is amortized. Each case sets the stub state it needs and
# tags its results under a unique key.
# ---------------------------------------------------------------------------


_E2E_SNIPPET = r"""
# ---- Stubs ---------------------------------------------------------------

CALLS = []                       # every backend invocation (alias, messages, latency_ms)
JUDGE_REPLY = {"value": "B"}     # what the judge model returns (mutable)
PER_CALL_SLEEP_S = {"value": 0.20}  # candidate inference latency

def _fake_chat(alias, messages, max_tokens=None, temperature=None, top_p=None, prompt=None, **kwargs):
    # Deterministic fake for ``_mlx_chat_completion``. ``**kwargs``
    # absorbs forward-compat additions like ``queue_controls`` and
    # ``request_id`` added in the audit hardening pass without
    # requiring the test fake to mirror every signature change.
    is_judge = any(
        isinstance(m, dict)
        and "strict judge" in (m.get("content") or "").lower()
        for m in messages
    )
    is_synth = any(
        isinstance(m, dict)
        and "chief synthesizer" in (m.get("content") or "").lower()
        for m in messages
    )

    if is_judge:
        text = JUDGE_REPLY["value"]
        sleep_s = 0.02
    elif is_synth:
        text = "SYNTHESIS: a unified answer that incorporates all round outputs."
        sleep_s = 0.02
    else:
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = (m.get("content") or "")
                break
        text = f"[{alias}] reply to: {last_user[:120]}"
        sleep_s = PER_CALL_SLEEP_S["value"]

    t0 = time.monotonic()
    time.sleep(sleep_s)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    CALLS.append({
        "alias": alias,
        "messages": [
            {"role": m.get("role"), "content": (m.get("content") or "")}
            for m in messages if isinstance(m, dict)
        ],
        "latency_ms": elapsed_ms,
        "is_judge": is_judge,
        "is_synth": is_synth,
    })

    return {
        "id": "fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_meta": {"latency_ms": elapsed_ms, "backend": "mlx"},
    }, None


FAKE_ALIASES = [
    "mlx-community/granite-4.1-8b-6bit",
    "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "lmstudio-community/Qwen3-Coder-Next-MLX-8bit",
    "mlx-community/gpt-oss-120b-mxfp4-bf16",
]

mod.mlx_manager = SimpleNamespace(
    get_available_aliases=lambda: list(FAKE_ALIASES),
    get_loaded_aliases=lambda: list(FAKE_ALIASES[:2]),
    context_windows={},
    root_path="/tmp/fake",
    get_memory_stats=lambda: {},
    load_model=lambda alias: None,
    get_last_load_error=lambda alias: None,
    unload_model=lambda alias: False,
)
mod._mlx_dash = None                  # silence dashboard event recording
mod.MAX_PARALLEL_MODEL_CALLS = 4      # enough headroom to see real parallelism
mod.SWARM_PER_CALL_TIMEOUT = 60
mod.SWARM_FANOUT_TIMEOUT = 0
mod.DEFAULT_MODEL = ""
mod._GRAB = None
mod._mlx_chat_completion = _fake_chat

# Give role lookup something useful.
mod.MODEL_ROLES = {
    "fast":     [FAKE_ALIASES[0], FAKE_ALIASES[1]],
    "coder":    [FAKE_ALIASES[2]],
    "reasoner": [FAKE_ALIASES[3]],
    "default":  [FAKE_ALIASES[0]],
}

client = mod.app.test_client()
out = {}


def _reset_calls():
    CALLS.clear()


def _post(path, body):
    rv = client.post(path, json=body)
    try:
        payload = json.loads(rv.get_data(as_text=True))
    except Exception:
        payload = {"_raw": rv.get_data(as_text=True)}
    return rv.status_code, payload


# ---- Case 1: /swarm/fanout returns N candidates and is actually parallel ----
_reset_calls()
PER_CALL_SLEEP_S["value"] = 0.20
t0 = time.monotonic()
status, body = _post("/swarm/fanout", {
    "models":   ["role:fast", "role:coder", "role:reasoner"],
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 16,
})
wall_ms = int((time.monotonic() - t0) * 1000)
out["fanout"] = {
    "status": status,
    "n_responses": len(body.get("responses") or []),
    "all_ok": all((r or {}).get("ok") for r in (body.get("responses") or [])),
    "models_resolved": [(r or {}).get("model") for r in (body.get("responses") or [])],
    "errors": [(r or {}).get("error") for r in (body.get("responses") or [])],
    "wall_ms": wall_ms,
    "per_call_sleep_ms": int(PER_CALL_SLEEP_S["value"] * 1000),
    "n_backend_calls": len(CALLS),
    "raw_body": body if status != 200 else None,
}


# ---- Case 2: /swarm/vote with a clean judge verdict picks the right one ----
_reset_calls()
JUDGE_REPLY["value"] = "B"
PER_CALL_SLEEP_S["value"] = 0.05
status, body = _post("/swarm/vote", {
    "models":   ["role:fast", "role:coder", "role:reasoner"],
    "judge":    "role:reasoner",
    "messages": [{"role": "user", "content": "which is best?"}],
    "max_tokens": 16,
})
out["vote_clean"] = {
    "status": status,
    "winner": (body.get("swarm") or {}).get("winner"),
    "strategy": (body.get("swarm") or {}).get("strategy"),
    "rationale": (body.get("swarm") or {}).get("rationale"),
    "fell_back_to_longest": "fell back to longest" in (
        (body.get("swarm") or {}).get("rationale") or ""
    ),
    "judge_was_called": any(c["is_judge"] for c in CALLS),
}


# ---- Case 3: /swarm/vote with a verbose-but-parseable judge reply ----
_reset_calls()
JUDGE_REPLY["value"] = "Looking at all the candidates carefully,\nthe best answer is **C** because it is more thorough."
status, body = _post("/swarm/vote", {
    "models":   ["role:fast", "role:coder", "role:reasoner"],
    "judge":    "role:reasoner",
    "messages": [{"role": "user", "content": "pick one"}],
    "max_tokens": 16,
})
out["vote_verbose_judge"] = {
    "status": status,
    "winner": (body.get("swarm") or {}).get("winner"),
    "rationale": (body.get("swarm") or {}).get("rationale"),
    "fell_back_to_longest": "fell back to longest" in (
        (body.get("swarm") or {}).get("rationale") or ""
    ),
}


# ---- Case 4: judge reply with no usable label falls back honestly ----
_reset_calls()
JUDGE_REPLY["value"] = "All three are equivalent, I cannot pick."
status, body = _post("/swarm/vote", {
    "models":   ["role:fast", "role:coder", "role:reasoner"],
    "judge":    "role:reasoner",
    "messages": [{"role": "user", "content": "tie?"}],
    "max_tokens": 16,
})
out["vote_unparseable"] = {
    "status": status,
    "rationale": (body.get("swarm") or {}).get("rationale"),
    "fell_back_to_longest": "fell back to longest" in (
        (body.get("swarm") or {}).get("rationale") or ""
    ),
}


# ---- Case 5: /swarm/pipeline substitutes {{previous}} even with braces ----
_reset_calls()

original_fake_chat = mod._mlx_chat_completion

def _stage_aware_chat(alias, messages, **kw):
    # Stage 1 emits code with literal braces; stage 2 should still see it.
    last_sys = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            last_sys = m.get("content") or ""
    if "review" in last_sys.lower() or "critique" in last_sys.lower():
        return {
            "id": "fake", "object": "chat.completion", "created": 0, "model": alias,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "looks ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_meta": {"latency_ms": 1, "backend": "mlx"},
        }, None
    # Stage 1: emit code with literal braces — the old _fmt would crash on this.
    code = 'def f(x):\n    return {"k": x, 1: 2}  # f-string: f"x={x}"'
    return {
        "id": "fake", "object": "chat.completion", "created": 0, "model": alias,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": code}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_meta": {"latency_ms": 1, "backend": "mlx"},
    }, None

# Wrap the original fake to record calls AND apply stage-aware logic.
def _piped_fake(alias, messages, **kw):
    resp, err = _stage_aware_chat(alias, messages, **kw)
    CALLS.append({
        "alias": alias,
        "messages": [
            {"role": m.get("role"), "content": (m.get("content") or "")}
            for m in messages if isinstance(m, dict)
        ],
        "latency_ms": 1, "is_judge": False, "is_synth": False,
    })
    return resp, err

mod._mlx_chat_completion = _piped_fake

status, body = _post("/swarm/pipeline", {
    "messages": [{"role": "user", "content": "Topic: monotonic stacks"}],
    "steps": [
        {"name": "draft",  "model": "role:coder",
         "system": "Write a 4-line Python example.", "max_tokens": 200},
        {"name": "review", "model": "role:reasoner",
         "system": "Review this code:\n{{draft}}\nAlso: the previous text was: {{previous}}",
         "max_tokens": 200},
    ],
})

# Find the stage-2 system prompt that was actually delivered.
stage_2_system = ""
for c in CALLS:
    for m in c["messages"]:
        if m["role"] == "system" and "Review this code" in m["content"]:
            stage_2_system = m["content"]
            break
    if stage_2_system:
        break

out["pipeline"] = {
    "status": status,
    "n_history": len(((body.get("swarm") or {}).get("history") or [])),
    "stage_2_system": stage_2_system,
    "stage_2_saw_draft": ("def f(x):" in stage_2_system) and ("return {" in stage_2_system),
    "stage_2_has_no_unsubstituted_placeholder": "{{draft}}" not in stage_2_system and "{{previous}}" not in stage_2_system,
    "final_text": (body.get("choices") or [{}])[0].get("message", {}).get("content", ""),
}

# Restore the broad fake for any later cases.
mod._mlx_chat_completion = original_fake_chat


# ---- Case 6: /swarm/debate round 2 actually sees round 1 transcript ----
_reset_calls()
PER_CALL_SLEEP_S["value"] = 0.02
JUDGE_REPLY["value"] = "synthesis fallback"  # debate uses synthesizer prompt, not letter judge

status, body = _post("/swarm/debate", {
    "models":   ["role:fast", "role:coder", "role:reasoner"],
    "rounds":   2,
    "judge":    "role:reasoner",
    "messages": [{"role": "user", "content": "vector DB vs full-text search"}],
    "max_tokens": 64,
})

# Round 2 calls should carry the round-1 transcript in their system prompt.
round2_calls = [c for c in CALLS if not c["is_judge"] and not c["is_synth"]
                and any("round 2" in (m.get("content") or "").lower()
                        for m in c["messages"] if m["role"] == "system")]
round2_systems = []
for c in round2_calls:
    for m in c["messages"]:
        if m["role"] == "system":
            round2_systems.append(m["content"])
            break

out["debate"] = {
    "status": status,
    "transcript_size": len(((body.get("swarm") or {}).get("transcript") or [])),
    "n_round2_calls": len(round2_calls),
    "round2_sees_round1": all(
        "[" in s and "]" in s and "reply to:" in s
        for s in round2_systems
    ) and len(round2_systems) > 0,
    "synthesizer_called": any(c["is_synth"] for c in CALLS),
}


# ---- Case 7: /swarm/models shape and config visibility ----
_reset_calls()
rv = client.get("/swarm/models")
models_body = json.loads(rv.get_data(as_text=True))
out["models_route"] = {
    "status": rv.status_code,
    "has_models":      "models" in models_body,
    "has_roles":       "roles" in models_body,
    "max_parallel":    models_body.get("max_parallel"),
    "default_strategy": models_body.get("swarm_chat_default_strategy"),
    "swarm_chat_enabled": models_body.get("swarm_chat_enabled"),
}


print("RESULT=" + json.dumps(out))
"""


def test_swarm_e2e_matrix() -> None:
    result = _run_swarm_subprocess(_E2E_SNIPPET)

    # ---- /swarm/fanout ----
    f = result["fanout"]
    assert f["status"] == 200, f
    assert f["n_responses"] == 3, f
    assert f["all_ok"] is True, f
    assert f["n_backend_calls"] == 3
    # Parallelism: 3 calls × 200ms sequential = 600ms; with parallelism ≥ 2,
    # wall time should be well under that. Use a generous bound to avoid CI flakes.
    assert f["wall_ms"] < int(f["per_call_sleep_ms"] * 2.5), (
        f"fanout looks sequential: wall_ms={f['wall_ms']} "
        f"per_call_sleep_ms={f['per_call_sleep_ms']}; "
        "check MAX_PARALLEL_MODEL_CALLS"
    )
    # Each spec resolves to a distinct model.
    assert len(set(f["models_resolved"])) == 3, f["models_resolved"]

    # ---- /swarm/vote with bare-letter verdict ----
    v = result["vote_clean"]
    assert v["status"] == 200, v
    assert v["judge_was_called"] is True
    assert v["fell_back_to_longest"] is False, (
        "judge said 'B' but vote fell back to longest — "
        "verdict parser regression"
    )
    # The judge replied "B" → second successful candidate wins.
    # Successes come back in the order specs were submitted, so winner == role:coder model.
    assert "Qwen3-Coder" in (v["winner"] or "")

    # ---- /swarm/vote with verbose judge reply ----
    vv = result["vote_verbose_judge"]
    assert vv["status"] == 200, vv
    assert vv["fell_back_to_longest"] is False, (
        "verbose judge reply containing '**C**' should still be parsed; "
        "the old strict ^A regex would have fallen back"
    )

    # ---- /swarm/vote with truly unparseable judge ----
    vu = result["vote_unparseable"]
    assert vu["status"] == 200, vu
    assert vu["fell_back_to_longest"] is True, (
        "judge said something without any label letter — must fall back "
        "honestly and surface that in the rationale"
    )

    # ---- /swarm/pipeline with literal braces in stage output ----
    p = result["pipeline"]
    assert p["status"] == 200, p
    assert p["n_history"] == 2
    assert p["stage_2_saw_draft"] is True, (
        "stage 2 system prompt did not contain stage 1's draft text; "
        "{{draft}} substitution broken. "
        f"stage_2_system={p['stage_2_system']!r}"
    )
    assert p["stage_2_has_no_unsubstituted_placeholder"] is True, (
        "stage 2 system prompt still contains literal {{...}} placeholders; "
        "substitution returned the raw template (the old _fmt failure mode). "
        f"stage_2_system={p['stage_2_system']!r}"
    )

    # ---- /swarm/debate round 2 references round 1 ----
    d = result["debate"]
    assert d["status"] == 200, d
    # 3 models × 2 rounds = 6 candidate calls; transcript should have all successes.
    assert d["transcript_size"] == 6, d
    assert d["n_round2_calls"] == 3, d
    assert d["round2_sees_round1"] is True, (
        "round 2 system prompts did not include the round 1 transcript "
        "from peer models"
    )
    assert d["synthesizer_called"] is True

    # ---- /swarm/models shape ----
    m = result["models_route"]
    assert m["status"] == 200, m
    assert m["has_models"] is True
    assert m["has_roles"] is True
    assert isinstance(m["max_parallel"], int) and m["max_parallel"] >= 2, (
        f"MAX_PARALLEL_MODEL_CALLS={m['max_parallel']} — "
        "set to >= 2 or swarm is sequential"
    )
    assert m["default_strategy"] in {"best-of-n", "first-success", "longest", "fanout"}
    assert m["swarm_chat_enabled"] is True
