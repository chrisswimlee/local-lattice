<!-- README -->

# Local Lattice

**The capability layer between your agents and your LLM compute. Agents
describe what they need; Lattice picks the right model, routes, swarms, and
falls back. One OpenAI-compatible API, local-first.**

[![CI](https://github.com/chrisswimlee/local-lattice/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisswimlee/local-lattice/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Project status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status-and-roadmap)

**Canonical repository:** [github.com/chrisswimlee/local-lattice](https://github.com/chrisswimlee/local-lattice).
The PyPI distribution name is **`local-lattice`**; the importable Python package remains **`middle_layer`** until Pass 3.

## The problem

Every agent framework hardcodes models. You write `model="gpt-4o"` or
`model="qwen2.5-coder-32b-instruct"` and ship it. The agent breaks when:

- the user has different models on disk,
- the operator wants to swap providers without redeploying,
- a small local model could have answered, but the agent went straight to
  the cloud anyway,
- one model wasn't enough and you needed a second opinion.

Agents shouldn't know model identifiers. They should declare **capabilities**
(`role:coder`, `role:reasoner`, vision, long context, low latency), and the
infrastructure should pick the best available local model — or fall back to
cloud — without the agent code changing.

## What Local Lattice is

A small Flask server that speaks the **OpenAI HTTP API** and adds a capability
layer on top of it:

- **Capability-based resolution.** `model="role:coder"`, priority lists
  (`"model-a,model-b,fallback"`), wildcards (`"*coder*"`), and automatic
  routing on vision content, prompt length, and a `X-MLX-Latency-Tier`
  header. Backed by [`mlx_roles.json`](./mlx_roles.json) and
  [`model_profiles.json`](./model_profiles.json) — see
  [`docs/capabilities.md`](./docs/capabilities.md) for the full grammar.
- **Swarm primitives, exposed as HTTP routes.** `/swarm/fanout`,
  `/swarm/vote`, `/swarm/pipeline`, `/swarm/debate` — let an agent ask for
  N opinions, a moderated vote, a sequential pipeline, or a multi-round
  debate without writing the orchestration itself.
- **Direct MLX execution on Apple Silicon** via
  [`mlx_lm`](https://github.com/ml-explore/mlx), with an LM Studio proxy
  backend for Linux / x86. Adding more backends (vLLM, llama.cpp) is on
  the roadmap.
- **Hybrid local-plus-cloud.** Optional escalation to Anthropic Claude
  when a request exceeds local capacity or requests it explicitly.
- **Production-shaped ops.** Multi-model LRU, per-model concurrency caps,
  bounded admission queue with priority and retry-after, and an in-process
  metrics dashboard at `/dashboard/`.

The HTTP shape is just OpenAI. Any agent framework that can point at a custom
OpenAI base URL works — see [`docs/integrations/`](./docs/integrations/) for
LangGraph and OpenAI Agents SDK examples.

> **Status (0.3.1): alpha.** The HTTP surface is stable in practice (every
> route is pinned by `docs/_internal/baseline/` regression captures) but the
> Python API and internal module layout will change before 1.0. Pin the
> version if you embed this. See [`docs/why-lattice.md`](./docs/why-lattice.md)
> for the longer "why this exists" story.

## 30-second quickstart

```bash
git clone https://github.com/chrisswimlee/local-lattice.git
cd local-lattice

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"     # Apple Silicon only; use `[lmstudio]` elsewhere

# Probe LM Studio or MLX on this machine and write role:* mappings
local-lattice-init
# or: local-lattice-init --backend lmstudio --dry-run

# point at a folder with MLX weights (LM Studio's default is fine)
export MLX_MODEL_ROOT="$HOME/.lmstudio/models"
export MIDDLE_LAYER_API_KEY="$(uuidgen)"   # enable auth; deny-by-default

local-lattice-mlx serve --host 127.0.0.1 --port 5001
# back-compat: middle-layer-mlx is the same entry point
```

In another shell:

```bash
curl -sS -H "X-API-Key: $MIDDLE_LAYER_API_KEY" \
     http://127.0.0.1:5001/v1/models | jq .

curl -sS http://127.0.0.1:5001/v1/chat/completions \
     -H "X-API-Key: $MIDDLE_LAYER_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"role:fast","messages":[{"role":"user","content":"ping"}]}'
```

The dashboard is at `http://127.0.0.1:5001/dashboard/` (set the same API
key in its sessionStorage prompt). Disable it with
`MLX_DASHBOARD_ENABLED=0`.

### 60-second demo

With either gateway running, [`scripts/demo.sh`](./scripts/demo.sh) walks
the whole pitch against your live model set:

```bash
./scripts/demo.sh                            # LM Studio gateway on :5000
BASE_URL=http://127.0.0.1:5001 ./scripts/demo.sh   # MLX gateway
```

It lists models, sends the same agent code at `role:fast` and
`role:coder` (watch them resolve to *different* loaded models), then asks
`/swarm/vote` for a judged second opinion. Swap what's loaded and run it
again — the calls don't change.

See [Swarm in 60 seconds](#swarm-in-60-seconds) below for sequence diagrams
and copy-paste examples.

## Swarm in 60 seconds

Swarm routes are the fastest way to get **multiple local models working
together** without writing orchestration. You keep sending capabilities
(`role:coder`, `"auto"`, …); Lattice resolves them against whatever is
loaded and runs fanout, judge, or sequential steps for you.

> **Tip:** Local reasoning models often need `max_tokens >= 2000` or they
> spend the whole budget on hidden chain-of-thought and return empty
> `content`. The snippets below default to 2000.

### How `/swarm/vote` works

One HTTP call → parallel fanout → judge picks a winner → you get a normal
OpenAI-shaped `chat.completion` plus a `swarm` object with candidates.

```mermaid
sequenceDiagram
    participant Client
    participant Lattice
    participant A as role:fast
    participant B as role:coder
    participant C as role:reasoner
    participant Judge as judge (role:reasoner)

    Client->>Lattice: POST /swarm/vote
    par Fanout (parallel)
        Lattice->>A: chat completion
        Lattice->>B: chat completion
        Lattice->>C: chat completion
    end
    A-->>Lattice: answer A
    B-->>Lattice: answer B
    C-->>Lattice: answer C
    Lattice->>Judge: rank anonymized candidates
    Judge-->>Lattice: winner + rationale
    Lattice-->>Client: chat.completion + swarm.winner
```

Use `"models": "auto"` to fan out to every loaded chat-capable model
(subject to the gateway's auto cap — see config table below).

### How `/swarm/pipeline` works

Sequential steps. Later steps reference earlier output via `{{plan}}`,
`{{code}}`, or `{{previous}}` in each step's `system` prompt.

```mermaid
sequenceDiagram
    participant Client
    participant Lattice
    participant Plan as plan (role:reasoner)
    participant Code as code (role:coder)
    participant Review as review (role:reasoner)

    Client->>Lattice: POST /swarm/pipeline
    Lattice->>Plan: step 1 — outline approach
    Plan-->>Lattice: plan text
    Lattice->>Code: step 2 — system includes {{plan}}
    Code-->>Lattice: code text
    Lattice->>Review: step 3 — system includes {{code}}
    Review-->>Lattice: final answer
    Lattice-->>Client: chat.completion (+ swarm.history)
```

### Copy-paste examples

**1. Single capability route** — drop-in for any OpenAI client:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",  # or :5001 for MLX
    api_key=os.environ.get("MIDDLE_LAYER_API_KEY", "local"),
)
resp = client.chat.completions.create(
    model="role:coder",
    messages=[{"role": "user", "content": "Reverse a string in Python."}],
    max_tokens=2000,
)
print(resp.model)   # concrete model id that answered — log this
print(resp.choices[0].message.content)
```

**2. Judged second opinion** — fanout + judge in one call:

```python
import os, requests

BASE = "http://127.0.0.1:5000"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": os.environ.get("MIDDLE_LAYER_API_KEY", ""),
}

resp = requests.post(
    f"{BASE}/swarm/vote",
    headers=headers,
    json={
        "models": "auto",
        "strategy": "best-of-n",
        "judge": "role:reasoner",
        "messages": [
            {"role": "user", "content": "Name a coffee-shop WiFi network."}
        ],
        "max_tokens": 2000,
    },
    timeout=300,
)
data = resp.json()
print("winner:", data.get("swarm", {}).get("winner"))
print(data["choices"][0]["message"]["content"])
```

**3. Plan → code → review pipeline:**

```python
resp = requests.post(
    f"{BASE}/swarm/pipeline",
    headers=headers,
    json={
        "messages": [
            {"role": "user", "content": "Build a CLI that counts words in a file."}
        ],
        "steps": [
            {
                "name": "plan",
                "model": "role:reasoner",
                "system": "Outline the approach in bullet points.",
                "max_tokens": 512,
            },
            {
                "name": "code",
                "model": "role:coder",
                "system": "Implement this plan:\n\n{{plan}}",
            },
            {
                "name": "review",
                "model": "role:reasoner",
                "system": "Critique and suggest fixes:\n\n{{code}}",
            },
        ],
        "max_tokens": 2000,
    },
    timeout=300,
)
```

**4. OpenAI client shortcut** — same vote, no new endpoint:

```python
resp = client.chat.completions.create(
    model="swarmCouncil",
    messages=[{"role": "user", "content": "Pros and cons of SQLite for a side project?"}],
    max_tokens=2000,
    extra_body={
        "swarm": {
            "models": "auto",
            "strategy": "best-of-n",
            "judge": "role:reasoner",
        },
    },
)
```

### Swarm route cheat sheet

| Route | What it does | Returns |
|-------|----------------|---------|
| `POST /swarm/fanout` | Same prompt → N models in parallel | All answers (`object: swarm.fanout`) |
| `POST /swarm/vote` | Fanout + judge | OpenAI `chat.completion` + `swarm.winner` |
| `POST /swarm/pipeline` | Sequential steps with `{{name}}` templates | Final step as `chat.completion` |
| `POST /swarm/debate` | Multi-round argument + judge synthesis | **MLX gateway only** (`:5001`) |
| `POST /v1/chat/completions` with `model: swarmCouncil` | Vote via plain chat API | Supports `stream: true` |

Full request/response contracts: [`docs/capabilities.md`](./docs/capabilities.md).
Agent-oriented reference: [`llms.txt`](./llms.txt).

Try it live: [`scripts/demo.sh`](./scripts/demo.sh) (steps 2–4 exercise
capability routing and `/swarm/vote`).

### Performance / routing overhead

Every timed response includes standard headers:

| Header | Meaning |
|--------|---------|
| `X-Lattice-Resolve-Ms` | Capability → model id (and swarm model expansion) |
| `X-Lattice-Queue-Ms` | Admission / queue wait before inference starts |
| `X-Lattice-Upstream-Ms` | LM Studio HTTP hop or MLX generation time |
| `X-Lattice-Total-Ms` | End-to-end handler wall time |

Legacy MLX headers `X-MLX-Latency-Ms` and `X-MLX-Queue-Wait-Ms` are still
set on the MLX gateway for one minor.

Structured logs (enabled by default, disable with `LATTICE_LOG_TIMING=0`):

```text
lattice.request resolve_ms=2 queue_ms=0 upstream_ms=840 total_ms=845 path=/v1/chat/completions status=200
```

Typical routing overhead on the **MLX direct path** is a few milliseconds.
The **LM Studio proxy** adds roughly **3–10ms** per request for the
localhost HTTP hop on top of resolve time. Streaming responses expose
queue/resolve timing in headers at stream start; upstream time reflects
generation and is also visible per chunk in the dashboard.

## Which gateway should I run?

Local Lattice ships **two interchangeable gateways** that speak the same
OpenAI-compatible HTTP surface. Pick one based on what you already have
running on the box:

| You have… | Run | Launcher | Port |
|---|---|---|---|
| **LM Studio installed and loading your models** | **`lmstudio` proxy** (recommended for most operators) | `./start_middle_layer.sh` or `./scripts/start.sh --profile lmstudio` | 5000 |
| MLX-converted weights and no LM Studio | `mlx` direct gateway | `./start_middle_layerMLX.sh` or `./scripts/start.sh --profile mlx` | 5001 |
| Memory-tight Mac running MoE / 70B+ models | `mlx` direct gateway in stable mode | `./scripts/start.sh --profile stable=safe` | 5001 |

### Pick `lmstudio` when…

- You already use LM Studio's UI as your model browser and download tool.
- You want a separate OS process serving inference (crash isolation: a bad
  load takes down LM Studio, not your gateway).
- You're running mixed model formats (GGUF, MLX, EXL2) — LM Studio's
  loader handles all of them; the MLX gateway only loads MLX weights.
- You don't care about the ~3–10ms HTTP roundtrip overhead per request.

This is the **primary path most operators want.** All the dynamic-by-
default behavior (strict loaded-model policy, curated swarm fanout)
lands here automatically when you use the launcher.

### Pick `mlx` when…

- You're running pure Apple-Silicon MLX models and want the lowest
  per-request latency (no HTTP hop, direct `mlx_lm.generate`).
- You want to ship MiddleLayer as a self-contained unit without
  requiring operators to install LM Studio separately.
- You need fine-grained in-process control over model lifecycle
  (programmatic load/unload, per-alias admission caps, real-time
  Metal-allocator hints).
- You're benchmarking — MLX shaves first-token latency on streaming
  endpoints.

The MLX gateway can run side-by-side with the LM Studio gateway on a
different port if you want both options available without switching.

### Pick `stable` when…

- You're on a memory-tight Mac (16 GB) and a single inference job can
  consume most of RAM. The stable profile tunes
  `MAX_CONCURRENT_MODELS=1`, `MAX_PARALLEL_MODEL_CALLS=1`,
  `MLX_PER_MODEL_INFLIGHT_CAP=1` and trims queue and token caps so
  the runtime never tries to coexist a second model with the first.
- Use `--profile stable=safe` (most conservative),
  `--profile stable=balanced`, or `--profile stable=faster` for the
  three pre-tuned tiers.

## How it compares

| Capability                                       | Local Lattice | `mlx_lm.server` | Ollama | LM Studio | LiteLLM |
|--------------------------------------------------|:-------------:|:---------------:|:------:|:---------:|:-------:|
| OpenAI `/v1/chat/completions` + `/v1/models`     |     ✅         |        ✅          |   ✅     |    ✅       |    ✅      |
| Streaming SSE (`data: ... [DONE]`)               |     ✅         |        ✅          |   ✅     |    ✅       |    ✅      |
| Capability routing (`role:coder`, vision, tier)  |     ✅         |        —           |   —      |     —      |     ~      |
| Auto-routing on prompt content (vision/long ctx) |     ✅         |        —           |   —      |     —      |     —      |
| Swarm: fanout / vote / pipeline / debate         |     ✅         |        —           |   —      |     —      |     —      |
| Hybrid local + cloud (Anthropic escalation)      |     ✅         |        —           |   —      |     —      |     ✅      |
| Direct MLX execution on Apple Silicon            |     ✅         |        ✅          |   —      |     —      |     —      |
| Multi-model LRU + per-model concurrency cap      |     ✅         |        —           |   ✅     |    ✅       |     —      |
| Admission queue with priority + retry-after      |     ✅         |        —           |   —      |     —      |     —      |
| In-process observability dashboard               |     ✅         |        —           |   —      |     —      |     —      |
| `pip install`, OpenAI-compatible API key auth    |     ✅         |        —           |   —      |     —      |     ✅      |

Local Lattice is **not** a replacement for `mlx_lm.server` or Ollama — it sits
*in front of* them and adds the capability layer that lets agent code stop
caring which model is running. If you only need a single model with raw
throughput, prefer the underlying runtime directly.

## Installing

### Apple Silicon (the main path)

```bash
pip install -e ".[mlx]"
```

Pulls `mlx-lm`, `huggingface_hub`, and `flask-cors`. The MiddleLayer CLI
auto-discovers `~/.lmstudio/models`, `~/.cache/lm-studio/models`,
`~/.cache/mlx-models` (in that order). Override with `MLX_MODEL_ROOT` or
`--model-root`.

### Linux / x86 (LM Studio proxy)

```bash
pip install -e ".[lmstudio,anthropic]"
```

This installs the cross-platform pieces only (no `mlx_lm`). The
`local-lattice-lmstudio` console script (alias: `middle-layer-lmstudio`) runs the legacy proxy that talks
to a separate LM Studio instance at `LM_STUDIO_URL=http://127.0.0.1:1234`.

### Everything cross-platform

```bash
pip install -e ".[all]"   # equivalent to [lmstudio,anthropic,dashboard,dev]
```

### Compatibility shims

For one minor version we still honour the previous workflow:

```bash
pip install -r requirements-mlx.txt           # == pip install -e .[mlx] (local-lattice)
pip install -r requirements-mlx-gateway.txt   # == pip install -e .[mlx,anthropic]
```

Both files print a deprecation note in their comments. They will be
removed in 0.4.0.

## Configuration

Configuration today is all environment variables, read at process start.
**Pass 2 is currently consolidating these into a typed
[`middle_layer.config.Settings`](./middle_layer/) object**; this README
will then auto-generate a complete table from the schema. Until that
lands, the canonical inventory of every variable, its default, and its
file location is the Pass-0 ground-truth document at
[`docs/_internal/CURRENT_STATE.md`](./docs/_internal/CURRENT_STATE.md)
(checked in on the `pass/0-discovery` branch).

Quick reference of the most common knobs:

| Env var                     | Default     | What it does                                            |
|-----------------------------|-------------|---------------------------------------------------------|
| `HOST`                      | `127.0.0.1` | Bind address. Refuses to start on a public interface without `MIDDLE_LAYER_API_KEY`. |
| `PORT`                      | `5001`      | TCP port for the gateway.                               |
| `MIDDLE_LAYER_API_KEY`      | _(unset)_   | If set, every request needs `X-API-Key` or `Bearer`. Compared constant-time. |
| `MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH` | _(unset)_ | Override the public-bind safety check. Use only behind a trusted auth-enforcing proxy. |
| `MIDDLE_LAYER_MAX_REQUEST_BYTES` | `10485760` | Max HTTP request body in bytes (default 10 MiB).      |
| `MLX_MODEL_ROOT`            | auto        | Where to look for MLX model directories.                |
| `DEFAULT_MODEL`             | _(empty)_   | Alias returned for `model: ""`/`auto`/`default`.        |
| `MAX_CONCURRENT_MODELS`     | `2`         | LRU bound on resident MLX models.                       |
| `MAX_PARALLEL_MODEL_CALLS`  | `2`         | Global concurrent-generation cap.                       |
| `MLX_PER_MODEL_INFLIGHT_CAP`| `1`         | Per-alias generation cap (MLX gateway). `0` disables admission (legacy; emits `DeprecationWarning` when unset before 0.4.0). |
| `MLX_FORCE_GC_ON_EVICT`     | `0`         | When `1`, run `gc.collect()` after every MLX eviction in addition to the Metal-cache release. Tighter peak RSS on memory-tight Macs at the cost of small extra wall-clock latency per swap. |
| `EXTRA_PLACEHOLDER_MODELS`  | _(unset → legacy OpenClaw set + `DeprecationWarning`)_ | Comma-separated extra "you pick" aliases; set to empty to exclude legacy ids. |
| `PREFER_LOADED_MODELS`      | `strict`    | LM Studio gateway loaded-id policy. `strict` never JIT-loads installed-but-not-loaded ids; `1` falls back to the installed set on a miss; `0` ignores loaded vs installed. Unset emits a `DeprecationWarning` (legacy default was `1`). |
| `SWARM_CHAT_DEFAULT_MODELS` | `auto`      | Default `swarm.models` list when a swarm chat request omits it. `auto`/`loaded`/`*` expand to the currently-loaded chat-capable set (filtered to exclude embedding models, capped at `SWARM_CHAT_AUTO_MAX`); or a comma-separated list of ids/`role:*` lookups. Unset emits a `DeprecationWarning` (legacy default was `role:reasoner,role:coder,role:fast`). |
| `SWARM_CHAT_AUTO_MAX`       | `3`         | Cap on how many loaded ids the `auto` sentinel contributes to a default-shaped swarm. Keeps fanout-vs-latency reasonable on boxes with many loaded models. Set to `0` to disable the cap. Dedicated `/swarm/fanout` HTTP endpoint ignores this. |
| `SWARM_CHAT_DEFAULT_STRATEGY` | `best-of-n` | Default swarm winner-pick when the request omits `swarm.strategy`. `best-of-n` (judge picks from candidates), `first-success` (returns on first temporally successful agent, cancels pending peers), `longest`, `fanout`. |
| `ANTHROPIC_API_KEY`         | _(unset)_   | Enables optional Claude escalation for long tasks.      |
| `ANTHROPIC_AUTO_ROUTE`      | `1`         | Auto-escalate big tasks. Will default off in 0.4.0.     |
| `MLX_DASHBOARD_ENABLED`     | `1`         | Mount the in-process dashboard at `/dashboard/`.        |
| `MLX_DASHBOARD_CAPTURE_PROMPTS` | `0`     | Keep prompts in the dashboard ring. Off by default.     |

## Security defaults (deny-by-default)

- **Constant-time API key check.** Both gateway backends and the
  dashboard compare keys with `hmac.compare_digest`. Send the key as
  either `X-API-Key: <key>` or `Authorization: Bearer <key>`.
- **Refuse to bind public without auth.** Starting on a non-loopback
  interface without `MIDDLE_LAYER_API_KEY` set exits with a clear error.
  Override with `MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH=1` only when an
  upstream proxy is enforcing authentication.
- **Request body size cap.** Default 10 MiB; tune with
  `MIDDLE_LAYER_MAX_REQUEST_BYTES`. Oversize requests get a Flask-native
  413.
- **Standard hardening headers** on every response:
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`.
  Dashboard responses additionally carry a strict
  `Content-Security-Policy` that disallows inline scripts and remote
  sources.
- **Dashboard model-load allowlist.** `/dashboard/api/models/load` only
  accepts aliases that pass a syntactic filter *and* appear in the live
  on-disk model set discovered by the MLX manager.
- **CORS off.** Set `CORS_ORIGINS=https://your.app` to allowlist a
  specific origin. `*` is accepted today but is rejected when combined
  with credentials.
- **Prompt logging off.** `MLX_DASHBOARD_CAPTURE_PROMPTS=0`. Turning it
  on stores recent user prompts in process memory only (never to disk
  in this release); a regex redactor is on the Pass-5+ roadmap.

Still open and tracked for future passes: per-IP rate limiting, HSTS
guidance behind TLS, and CSP nonce-mode for the dashboard. See the
[hardening roadmap](./SECURITY.md#hardening-roadmap) for the full list.

A full threat model and the responsible-disclosure address live in
[SECURITY.md](./SECURITY.md).

## Docs

- [`llms.txt`](./llms.txt) — **self-contained integration guide for AI
  agents**: feed this one file to a coding agent and it has every
  endpoint shape, the `model` grammar, and the error contract needed to
  integrate without human help.
- [`docs/why-lattice.md`](./docs/why-lattice.md) — the longer "why this
  exists" story: capability routing as an agent-infra primitive.
- [`docs/capabilities.md`](./docs/capabilities.md) — formal spec of the
  capability protocol: resolver grammar, role registry, auto-routing,
  swarm endpoint contracts.
- [`docs/integrations/`](./docs/integrations/) — drop-in examples for
  LangGraph, OpenAI Agents SDK, and other agent frameworks.
- [CONTRIBUTING.md](./CONTRIBUTING.md) — dev loop, commits, tests.
- [SECURITY.md](./SECURITY.md) — vulnerability reporting.
- [CHANGELOG.md](./CHANGELOG.md) — semver-shaped release notes.
- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) — Contributor Covenant 2.1.
- `docs/configuration.md` (Pass 2) — every setting, auto-generated.
- `docs/openapi.yaml` (Pass 8) — hand-curated OpenAPI spec.

## Project status and roadmap

This repository is mid-migration from "useful internal code" to a
polished open-source release. The migration is broken into named
passes, each landed as its own PR. Pass-by-pass progress lives in
[CHANGELOG.md](./CHANGELOG.md). A high-level summary:

- **Pass 0** (done) — read-only discovery, baseline regression captures.
- **Pass 1** (this release) — legal foundation, build system,
  documentation, launcher consolidation, branding scrub.
- **Pass 2** — configuration consolidation (`pydantic-settings`).
- **Pass 3** — restructure the two monoliths into a typed package.
- **Pass 4** — security hardening (auth, CORS, rate limits, CSP).
- **Pass 5** — tests, types, linting, CI.
- **Pass 6** — observability (`structlog`, `/metrics`, optional OTEL).
- **Pass 7** — distribution (PyPI, Dockerfile, devcontainer).
- **Pass 8** — docs, OpenAPI, dashboard UX overhaul.
- **Pass 9** — polish and 1.0.

## License

Apache-2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
