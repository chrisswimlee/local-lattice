#!/usr/bin/env python3
"""Swarm-intelligence benchmark: is the swarm actually smarter than one model?

Runs a JSONL dataset of ``{prompt, expected}`` pairs against three methods
(``single``, ``vote``, ``debate``), grades each response, and prints a side-by-side
accuracy/latency report so operators can answer "is the swarm worth the cost
on my prompt distribution" with a number instead of vibes.

This is the Layer-3 piece that complements ``tests/test_swarm_e2e.py`` (Layer 1
plumbing + Layer 2 config sanity). It hits a *live* Lattice server and pays
real inference cost; intended for nightly / on-demand runs, not CI.

Usage
-----

    # Cheap smoke against the bundled 8-prompt starter set:
    python scripts/swarm_eval.py

    # Real evaluation:
    python scripts/swarm_eval.py \\
        --server         http://localhost:5001 \\
        --dataset        scripts/swarm_eval_dataset.jsonl \\
        --methods        single,vote,debate \\
        --single-model   role:reasoner \\
        --vote-models    "role:reasoner,role:coder,role:fast" \\
        --vote-judge     role:reasoner \\
        --debate-models  "role:reasoner,role:coder,role:fast" \\
        --debate-rounds  2 \\
        --debate-judge   role:reasoner \\
        --judge-model    role:reasoner \\
        --max-tokens     512 \\
        --out            swarm_eval_results.jsonl

Dataset format (JSONL, one record per line)
-------------------------------------------

    {"id": "math-01", "prompt": "What is 17 * 23?", "expected": "391"}
    {"id": "open-01", "prompt": "Summarize MCTS in one sentence.",
     "expected": null, "grader": "llm_judge"}

If ``expected`` is set and ``grader`` is omitted, ``substring`` grading is used.
If ``expected`` is ``null``, ``llm_judge`` is required.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "scripts" / "swarm_eval_dataset.jsonl"


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------


def _post(url: str, body: dict, token: str | None, timeout: float) -> tuple[dict | None, str | None]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return None, f"network: {e}"
    if r.status_code >= 400:
        return None, f"http {r.status_code}: {r.text[:200]}"
    try:
        return r.json(), None
    except ValueError:
        return None, f"non-json response: {r.text[:200]}"


def call_single(server: str, model: str, prompt: str, max_tokens: int,
                token: str | None, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    t0 = time.monotonic()
    resp, err = _post(f"{server}/v1/chat/completions", body, token, timeout)
    wall_ms = int((time.monotonic() - t0) * 1000)
    return {
        "method": "single",
        "ok": err is None,
        "error": err,
        "wall_ms": wall_ms,
        "answer": _extract_answer(resp),
        "usage": (resp or {}).get("usage") or {},
    }


def call_vote(server: str, models: list[str], judge: str, prompt: str,
              max_tokens: int, token: str | None, timeout: float) -> dict:
    body = {
        "models": models,
        "judge": judge,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "strategy": "best-of-n",
    }
    t0 = time.monotonic()
    resp, err = _post(f"{server}/swarm/vote", body, token, timeout)
    wall_ms = int((time.monotonic() - t0) * 1000)
    return {
        "method": "vote",
        "ok": err is None,
        "error": err,
        "wall_ms": wall_ms,
        "answer": _extract_answer(resp),
        "usage": (resp or {}).get("usage") or {},
        "winner": ((resp or {}).get("swarm") or {}).get("winner"),
        "rationale": ((resp or {}).get("swarm") or {}).get("rationale"),
    }


def call_debate(server: str, models: list[str], judge: str, rounds: int,
                prompt: str, max_tokens: int, token: str | None, timeout: float) -> dict:
    body = {
        "models": models,
        "judge": judge,
        "rounds": rounds,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    t0 = time.monotonic()
    resp, err = _post(f"{server}/swarm/debate", body, token, timeout)
    wall_ms = int((time.monotonic() - t0) * 1000)
    return {
        "method": "debate",
        "ok": err is None,
        "error": err,
        "wall_ms": wall_ms,
        "answer": _extract_answer(resp),
        "usage": (resp or {}).get("usage") or {},
        "transcript_len": len(((resp or {}).get("swarm") or {}).get("transcript") or []),
    }


def _extract_answer(resp: dict | None) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return (msg.get("content") or "").strip()


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade(answer: str, row: dict, server: str, judge_model: str,
          token: str | None, timeout: float) -> dict:
    grader = row.get("grader") or ("substring" if row.get("expected") else "llm_judge")
    expected = row.get("expected")

    if not answer:
        return {"grader": grader, "correct": False, "reason": "empty answer"}

    if grader == "substring":
        if not isinstance(expected, str) or not expected:
            return {"grader": grader, "correct": False, "reason": "no expected string"}
        ok = expected.lower() in answer.lower()
        return {"grader": grader, "correct": ok,
                "reason": f"substring {'found' if ok else 'not found'}: {expected!r}"}

    if grader == "exact":
        if not isinstance(expected, str):
            return {"grader": grader, "correct": False, "reason": "no expected string"}
        ok = answer.strip().lower() == expected.strip().lower()
        return {"grader": grader, "correct": ok, "reason": "exact"}

    if grader == "regex":
        if not isinstance(expected, str):
            return {"grader": grader, "correct": False, "reason": "no regex"}
        try:
            ok = re.search(expected, answer, re.IGNORECASE | re.DOTALL) is not None
        except re.error as e:
            return {"grader": grader, "correct": False, "reason": f"bad regex: {e}"}
        return {"grader": grader, "correct": ok, "reason": f"regex /{expected}/"}

    if grader == "llm_judge":
        rubric = row.get("rubric") or (
            "You are a strict grader. The user asked a question and a model gave an answer. "
            "Decide whether the answer is correct and on-topic. Reply with exactly one word: "
            "YES or NO, followed by a one-sentence reason."
        )
        body = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user", "content": (
                    f"Question:\n{row.get('prompt', '')}\n\n"
                    f"Answer:\n{answer}\n\n"
                    "Is the answer correct and on-topic?"
                )},
            ],
            "max_tokens": 80,
            "temperature": 0.0,
        }
        resp, err = _post(f"{server}/v1/chat/completions", body, token, timeout)
        if err:
            return {"grader": grader, "correct": False, "reason": f"judge call failed: {err}"}
        verdict = _extract_answer(resp)
        ok = bool(re.match(r"\s*yes\b", verdict, re.IGNORECASE))
        return {"grader": grader, "correct": ok, "reason": f"judge: {verdict[:140]}"}

    return {"grader": grader, "correct": False, "reason": f"unknown grader: {grader}"}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate(rows: list[dict], method: str) -> dict:
    method_rows = [r for r in rows if r["method"] == method]
    n = len(method_rows)
    ok_rows = [r for r in method_rows if r["call"]["ok"]]
    correct_rows = [r for r in method_rows if r["grade"]["correct"]]
    latencies = [r["call"]["wall_ms"] for r in ok_rows]
    completion_tokens = [
        (r["call"].get("usage") or {}).get("completion_tokens") or 0
        for r in ok_rows
    ]
    return {
        "method": method,
        "n": n,
        "ok": len(ok_rows),
        "correct": len(correct_rows),
        "accuracy": (len(correct_rows) / n) if n else 0.0,
        "mean_wall_ms": int(statistics.mean(latencies)) if latencies else 0,
        "median_wall_ms": int(statistics.median(latencies)) if latencies else 0,
        "mean_completion_tokens": int(statistics.mean(completion_tokens))
        if completion_tokens else 0,
    }


def _win_matrix(rows: list[dict], methods: list[str]) -> dict:
    by_id: dict[str, dict[str, bool]] = {}
    for r in rows:
        slot = by_id.setdefault(r["id"], {})
        slot[r["method"]] = bool(r["grade"]["correct"])
    counts: dict[str, int] = {}
    for results in by_id.values():
        key = "+".join(m for m in methods if results.get(m)) or "none"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _print_table(summaries: list[dict]) -> None:
    cols = ["method", "n", "ok", "correct", "accuracy",
            "mean_wall_ms", "median_wall_ms", "mean_completion_tokens"]
    widths = {c: max(len(c), max((len(str(s[c])) for s in summaries), default=0)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for s in summaries:
        row = []
        for c in cols:
            v = s[c]
            if c == "accuracy":
                row.append(f"{v * 100:5.1f}%".ljust(widths[c]))
            else:
                row.append(str(v).ljust(widths[c]))
        print(" | ".join(row))


def _print_deltas(summaries: list[dict]) -> None:
    by_m = {s["method"]: s for s in summaries}
    if "single" not in by_m:
        return
    base = by_m["single"]
    print("\nDelta vs single (positive accuracy delta = swarm is helping):")
    for m in ("vote", "debate"):
        if m not in by_m:
            continue
        s = by_m[m]
        d_acc = (s["accuracy"] - base["accuracy"]) * 100
        if base["mean_wall_ms"] > 0:
            lat_mult = s["mean_wall_ms"] / base["mean_wall_ms"]
        else:
            lat_mult = float("inf")
        print(
            f"  {m:<6} accuracy delta: {d_acc:+5.1f}pp   "
            f"latency multiplier: {lat_mult:.2f}x"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"dataset line {i} is not valid JSON: {e}") from e
    for i, row in enumerate(rows):
        row.setdefault("id", f"row-{i}")
        if not row.get("prompt"):
            raise SystemExit(f"row {row['id']} missing 'prompt'")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--server", default="http://localhost:5001",
                   help="Lattice base URL (default: http://localhost:5001)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help=f"JSONL dataset (default: {DEFAULT_DATASET.relative_to(REPO_ROOT)})")
    p.add_argument("--methods", default="single,vote,debate",
                   help="Comma-separated subset of: single,vote,debate")
    p.add_argument("--single-model", default="role:reasoner")
    p.add_argument("--vote-models", default="role:reasoner,role:coder,role:fast")
    p.add_argument("--vote-judge", default="role:reasoner")
    p.add_argument("--debate-models", default="role:reasoner,role:coder,role:fast")
    p.add_argument("--debate-rounds", type=int, default=2)
    p.add_argument("--debate-judge", default="role:reasoner")
    p.add_argument("--judge-model", default="role:reasoner",
                   help="Model used by the grader when grader=llm_judge")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Per-HTTP-request timeout in seconds")
    p.add_argument("--token", default=None,
                   help="Bearer token if the server requires auth (MIDDLE_LAYER_TOKEN)")
    p.add_argument("--out", type=Path, default=None,
                   help="Write per-row results to this JSONL file")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N prompts (for quick iteration)")
    args = p.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in {"single", "vote", "debate"}:
            raise SystemExit(f"unknown method: {m}")

    rows = _load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    print(f"server : {args.server}")
    print(f"dataset: {args.dataset} ({len(rows)} prompts)")
    print(f"methods: {methods}")
    print()

    out_fh = args.out.open("w", encoding="utf-8") if args.out else None
    all_results: list[dict] = []

    try:
        for row_idx, row in enumerate(rows, 1):
            prompt = row["prompt"]
            prompt_id = row["id"]
            print(f"[{row_idx}/{len(rows)}] {prompt_id}: {prompt[:80]}")

            for method in methods:
                if method == "single":
                    call = call_single(args.server, args.single_model, prompt,
                                       args.max_tokens, args.token, args.timeout)
                elif method == "vote":
                    call = call_vote(args.server,
                                     [m.strip() for m in args.vote_models.split(",")],
                                     args.vote_judge, prompt,
                                     args.max_tokens, args.token, args.timeout)
                elif method == "debate":
                    call = call_debate(args.server,
                                       [m.strip() for m in args.debate_models.split(",")],
                                       args.debate_judge, args.debate_rounds, prompt,
                                       args.max_tokens, args.token, args.timeout)
                else:  # pragma: no cover - argparse guards this
                    continue

                if call["ok"]:
                    g = grade(call["answer"], row, args.server, args.judge_model,
                              args.token, args.timeout)
                else:
                    g = {"grader": "n/a", "correct": False,
                         "reason": f"call failed: {call['error']}"}

                rec: dict[str, Any] = {
                    "id": prompt_id,
                    "method": method,
                    "prompt": prompt,
                    "expected": row.get("expected"),
                    "call": call,
                    "grade": g,
                }
                all_results.append(rec)
                if out_fh:
                    out_fh.write(json.dumps(rec) + "\n")
                    out_fh.flush()

                tag = "OK " if g["correct"] else "MISS"
                print(f"    {method:<6} {tag} ({call['wall_ms']}ms) {g['reason'][:100]}")
    finally:
        if out_fh:
            out_fh.close()

    print()
    summaries = [_aggregate(all_results, m) for m in methods]
    _print_table(summaries)
    _print_deltas(summaries)

    print("\nPer-prompt correctness combinations:")
    matrix = _win_matrix(all_results, methods)
    for combo, n in sorted(matrix.items(), key=lambda kv: -kv[1]):
        print(f"  {combo or '(none)':<30} {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
