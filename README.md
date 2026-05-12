<!-- README -->

# MiddleLayer

**MLX-native OpenAI-compatible gateway with capability routing, an admission
queue, and a hybrid local-plus-cloud swarm — built for Apple Silicon.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Project status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

> **Repository URL:** README links may reference `github.com/middle-layer/middle-layer`
> while the live remote is `github.com/chrisswimlee/local-lattice`. Maintainers
> will align canonical URLs before a public announcement.

MiddleLayer is a small Flask server that speaks the OpenAI HTTP API but runs
your models directly via [Apple `mlx_lm`](https://github.com/ml-explore/mlx).
On top of that it adds the production-shaped pieces an everyday inference
gateway needs but `mlx_lm.server` does not: capability-aware model routing, a
bounded admission queue with per-model concurrency caps, a hybrid local-plus-
cloud "swarm" (`/swarm/fanout|vote|pipeline|debate`), an in-process metrics
dashboard, and optional escalation to Anthropic Claude for long-form work.

> **Status (0.1.0): alpha.** The HTTP surface is stable in practice (every
> route is pinned by `docs/_internal/baseline/` regression captures) but the
> Python API and internal module layout will change before 1.0. Pin the
> version if you embed this.

## 30-second quickstart

```bash
git clone https://github.com/middle-layer/middle-layer.git
cd middle-layer

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"     # Apple Silicon only; use `[lmstudio]` elsewhere

# point at a folder with MLX weights (LM Studio's default is fine)
export MLX_MODEL_ROOT="$HOME/.lmstudio/models"
export MIDDLE_LAYER_API_KEY="$(uuidgen)"   # enable auth; deny-by-default

middle-layer-mlx serve --host 127.0.0.1 --port 5001
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

## What you get

| Capability                                  | MiddleLayer | `mlx_lm.server` | Ollama | LM Studio | LiteLLM |
|---------------------------------------------|:-----------:|:---------------:|:------:|:---------:|:-------:|
| OpenAI `/v1/chat/completions` + `/v1/models`|    ✅        |        ✅          |   ✅     |    ✅       |    ✅      |
| Streaming SSE (`data: ... [DONE]`)          |    ✅        |        ✅          |   ✅     |    ✅       |    ✅      |
| Direct MLX execution (no LM Studio needed)  |    ✅        |        ✅          |   —      |     —      |     —      |
| Multi-model LRU + per-model concurrency cap |    ✅        |        —           |   ✅     |    ✅       |     —      |
| Capability routing (`role:coder`, latency tier) | ✅      |        —           |   —      |     —      |     ~      |
| Admission queue with priority + retry-after |    ✅        |        —           |   —      |     —      |     —      |
| Swarm: fanout / vote / pipeline / debate    |    ✅        |        —           |   —      |     —      |     —      |
| Optional Anthropic Opus escalation          |    ✅        |        —           |   —      |     —      |     ✅      |
| In-process observability dashboard          |    ✅        |        —           |   —      |     —      |     —      |
| `pip install`, OpenAI-compatible API key auth |   ✅       |        —           |   —      |     —      |     ✅      |

The gateway is **not** a replacement for `mlx_lm.server` — it sits in front
of `mlx_lm` and adds operator-shaped behaviour. If you only need a single
model with raw throughput, prefer `mlx_lm.server`.

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
`middle-layer-lmstudio` console script runs the legacy proxy that talks
to a separate LM Studio instance at `LM_STUDIO_URL=http://127.0.0.1:1234`.

### Everything cross-platform

```bash
pip install -e ".[all]"   # equivalent to [lmstudio,anthropic,dashboard,dev]
```

### Compatibility shims

For one minor version we still honour the previous workflow:

```bash
pip install -r requirements-mlx.txt           # == pip install -e .[mlx]
pip install -r requirements-mlx-gateway.txt   # == pip install -e .[mlx,anthropic]
```

Both files print a deprecation note in their comments. They will be
removed in 0.2.0.

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
| `HOST`                      | `127.0.0.1` | Bind address. Stay local unless you have auth on.       |
| `PORT`                      | `5001`      | TCP port for the gateway.                               |
| `MIDDLE_LAYER_API_KEY`      | _(unset)_   | If set, every request needs `X-API-Key` or `Bearer`.    |
| `MLX_MODEL_ROOT`            | auto        | Where to look for MLX model directories.                |
| `DEFAULT_MODEL`             | _(empty)_   | Alias returned for `model: ""`/`auto`/`default`.        |
| `MAX_CONCURRENT_MODELS`     | `2`         | LRU bound on resident MLX models.                       |
| `MAX_PARALLEL_MODEL_CALLS`  | `2`         | Global concurrent-generation cap.                       |
| `MLX_PER_MODEL_INFLIGHT_CAP`| `0` (∞)     | Per-alias generation cap.                               |
| `EXTRA_PLACEHOLDER_MODELS`  | _(unset → legacy OpenClaw set + `DeprecationWarning`)_ | Comma-separated extra "you pick" aliases; set to empty to exclude legacy ids. |
| `ANTHROPIC_API_KEY`         | _(unset)_   | Enables optional Claude escalation for long tasks.      |
| `ANTHROPIC_AUTO_ROUTE`      | `1`         | Auto-escalate big tasks. Will default off in 0.2.0.     |
| `MLX_DASHBOARD_ENABLED`     | `1`         | Mount the in-process dashboard at `/dashboard/`.        |
| `MLX_DASHBOARD_CAPTURE_PROMPTS` | `0`     | Keep prompts in the dashboard ring. Off by default.     |

## Security defaults (deny-by-default)

- **Auth on, listen local.** `HOST=127.0.0.1` and `MIDDLE_LAYER_API_KEY`
  is the supported posture for any host that is not a developer laptop.
  Pass 4 will refuse to start when bound to a public interface without
  an API key.
- **CORS off.** Set `CORS_ORIGINS=https://your.app` to allowlist a
  specific origin. `*` is accepted today but will fail loudly when
  combined with credentials in Pass 4.
- **Prompt logging off.** `MLX_DASHBOARD_CAPTURE_PROMPTS=0`. Turning it
  on stores recent user prompts in process memory only (never to disk
  in this release); a regex redactor is on the Pass-4 roadmap.
- **API key comparison is being moved to `hmac.compare_digest`** in
  Pass 4 (currently `!=` — flagged as RISK_REGISTER P4-02).

A full threat model and the responsible-disclosure address live in
[SECURITY.md](./SECURITY.md).

## Docs

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
