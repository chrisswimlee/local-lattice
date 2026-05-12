# MiddleLayer — Current State Report (Pass 0, baseline only)

> **Read-only report.** No source files were modified to produce this
> document. It is the ground truth for every later pass.
>
> Generated: 2026-05-12. Author: Pass 0 discovery.

## 1. Repo layout

```
MiddleLayer/
├── middle_layer.py                          # 57 541 B  Flask gateway in front of LM Studio (+ Anthropic via litellm)
├── middle_layer.py.corrupted-backup         #  1 593 B  half-formed `*** End Patch` diff (DELETE in Pass 1)
├── middle_layerMLX.py                       # 154 235 B Flask gateway running models directly via mlx_lm (the main artefact)
├── mlx_dashboard.py                         #  14 399 B Flask blueprint: bounded in-memory metrics + JSON API + static
├── dashboard/index.html                     #  ~3.4 kB  minimal observability UI
├── dashboard/app.js                         #  ~6.0 kB  ES5/IIFE; uses `innerHTML` extensively (CSP-hostile)
├── model_profiles.json                      #  ~1.8 kB  per-alias / pattern profiles (context, ceiling, vision, tier)
├── mlx_roles.json                           #  ~0.7 kB  role -> ordered alias/substring list (priority resolver)
├── requirements-mlx.txt                     #  ~0.1 kB  thin (mlx-lm, flask, requests, huggingface_hub, litellm)
├── requirements-mlx-gateway.txt             #  ~0.1 kB  overlapping (flask, mlx-lm, hf, requests, flask-cors, gunicorn, litellm)
├── start_middle_layer.sh                    # LM Studio backend launcher (PORT 5000, picks workspace venv)
├── start_middle_layerMLX.sh                 # MLX backend launcher  (PORT 5001, prefers .venv → ../.venv → middle_layer_venv)
├── start_middle_layerMLX_5001_stable.sh     # Same launcher with safe|balanced|faster stability profiles
├── run_middle_layer_mlx.sh                  # Minimal pass-through to `python middle_layerMLX.py serve`
├── run_with_venv.sh                         # Bootstrap-or-reuse .venv, then run MLX backend
├── setup_mlx.sh                             # One-command venv create → pip install → optional grab + serve
└── __pycache__/                             # CHECKED IN — DELETE in Pass 1 + add .gitignore
```

**Not present (legitimate Pass 1 work):** `LICENSE`, `README.md`,
`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`,
`AGENTS.md`, `pyproject.toml`, `.gitignore`, `tests/`, `.github/`,
`docs/` (other than this internal report), `.editorconfig`, `Makefile`,
`.env.example`, `.pre-commit-config.yaml`.

## 2. Repository state

- **Not a git repository.** `git status` from inside the directory:
  `fatal: not a git repository (or any of the parent directories): .git`.
  `git init` is the very first action of Pass 1.
- The repo lives at `/Users/chrisswimlee/.openclaw/MiddleLayer`,
  i.e. **inside the parent OpenClaw project**. Both shell scripts and
  Python code reach across the boundary:
  - `start_middle_layer.sh` activates `$WS_ROOT/middle_layer_venv`
    (the workspace's venv).
  - `start_middle_layerMLX.sh` and `_5001_stable.sh` cascade through
    five candidate venv paths spanning both projects.
  - Hardcoded "OpenClaw" placeholder model ids are checked into
    `PLACEHOLDER_MODELS` in both backends (`openclaw`, `middlelayer`,
    `mlxmiddlelayer`, …).
- `middle_layer_venv` referenced by scripts is actually a **symlink** in
  the parent dir to `/Users/chrisswimlee/.openclaw/workspace/middle_layer_venv`.
- A baseline server was successfully brought up on port 5099 against
  `mlx-community/Nemotron-Mini-4B-Instruct-4bit-mlx` and all routes
  responded; raw responses live under `docs/_internal/baseline/`.

## 3. HTTP routes (full inventory)

### `middle_layerMLX.py` (the canonical backend)

| Method | Path                          | Handler                         | Auth | Notes                                             |
|--------|-------------------------------|---------------------------------|------|---------------------------------------------------|
| GET    | `/healthz`                    | `healthz`                       | yes  | Returns full runtime status JSON                  |
| GET    | `/v1/models`                  | `list_models`                   | yes  | OpenAI list shape; alphabetical                   |
| DELETE | `/v1/models/<path:alias>`     | `unload_model`                  | yes  | Drops alias from the LRU                          |
| POST   | `/v1/chat/completions`        | `chat_completions`              | yes  | Streaming or batch; OpenAI shape                  |
| POST   | `/v1/completions`             | `completions`                   | yes  | Legacy completion shape                           |
| GET    | `/swarm/models`               | `swarm_models`                  | yes  | Roles / availability snapshot                     |
| POST   | `/swarm/fanout`               | `swarm_fanout`                  | yes  | Body: `{models, messages, max_parallel?}`         |
| POST   | `/swarm/vote`                 | `swarm_vote`                    | yes  | `strategy ∈ {best-of-n, first-success, longest}`  |
| POST   | `/swarm/pipeline`             | `swarm_pipeline`                | yes  | `steps[]` with `{{previous}}` / `{{step_name}}` templating |
| POST   | `/swarm/debate`               | `swarm_debate`                  | yes  | `len(models) >= 2`; multi-round + judge synthesis |
| OPTIONS| `/<path:path>` (and `/`)      | `_cors_preflight`               | n/a  | **Only registered when `CORS_ORIGINS` is set AND `flask_cors` is unavailable** |

Dashboard (registered as Flask Blueprint by `mlx_dashboard.py`):

| Method | Path                              | Handler              | Auth     | Notes                          |
|--------|-----------------------------------|----------------------|----------|--------------------------------|
| GET    | `/dashboard`                      | `dashboard_redirect_slash` | none | 302 → `/dashboard/`            |
| GET    | `/dashboard/`                     | `dashboard_index`    | **none** | Static HTML                    |
| GET    | `/dashboard/<path:name>`          | `dashboard_static`   | **none** | Static asset (path-traversal guarded) |
| GET    | `/dashboard/api/snapshot`         | `api_snapshot`       | yes      | Live JSON state                |
| GET    | `/dashboard/api/config`           | `api_config`         | yes      | Public dashboard config        |
| POST   | `/dashboard/api/preferences`      | `api_preferences`    | yes      | Runtime default model + presets|
| POST   | `/dashboard/api/models/load`      | `api_models_load`    | yes      | **Triggers `mlx_lm.load(alias)` — alias not validated against allowlist** |

### `middle_layer.py` (legacy LM Studio path)

| Method | Path                          | Handler             | Auth | Notes                                                     |
|--------|-------------------------------|---------------------|------|-----------------------------------------------------------|
| GET    | `/healthz`                    | `healthz`           | yes  | Reports LM Studio + Anthropic + role config               |
| ANY    | `/v1/<path:endpoint>`         | `proxy`             | yes  | Generic OpenAI proxy to LM Studio + Anthropic escalation  |
| GET    | `/swarm/models`               | `swarm_models`      | yes  | Roles vs LM Studio loaded list                            |
| POST   | `/swarm/fanout`               | `swarm_fanout`      | yes  | Same shape as MLX backend                                 |
| POST   | `/swarm/vote`                 | `swarm_vote`        | yes  | Same shape as MLX backend                                 |
| POST   | `/swarm/pipeline`             | `swarm_pipeline`    | yes  | Same shape as MLX backend                                 |

There is **no `/swarm/debate`** in the LM Studio backend.

## 4. Configuration env vars

> Source-of-truth for Pass 2. All values below are read at module import
> time via direct `os.environ.get(...)` calls in `middle_layerMLX.py`,
> `middle_layer.py`, and `mlx_dashboard.py`.

### Server

| Var | Default | Used in | Notes |
|-----|---------|---------|-------|
| `HOST` | `127.0.0.1` | `middle_layerMLX.py:3776,3825`, `middle_layer.py:1472` | argparse default + fallback |
| `PORT` | `5001` (MLX) / `5000` (LM) | both | argparse default + fallback |
| `MIDDLE_LAYER_API_KEY` | unset → no auth | `middle_layerMLX.py:156`, `middle_layer.py:44`, `mlx_dashboard.py:43` | Compared with `!=` (not constant-time) |
| `CORS_ORIGINS` | `""` (off) | `middle_layerMLX.py:216` | Comma-list; `*` permitted |
| `MAX_WORKERS` | `4` | `middle_layerMLX.py:161` | Flask threaded=True hint |

### MLX runtime

| Var | Default | Notes |
|-----|---------|-------|
| `MLX_MODEL_ROOT` | first existing of `~/.lmstudio/models`, `~/.cache/lm-studio/models`, `~/.cache/mlx-models`, else last | auto-discover |
| `MAX_CONCURRENT_MODELS` | `2` | LRU bound for `MLXManager` |
| `PRELOAD_MODELS` | `""` | comma list of aliases to load at boot |
| `MLX_GRAB_MODEL` | unset | single-model "grab" mode (HF id or path) |
| `MLX_GRAB_DISPLAY_NAME` | `mlx` | id reported to clients in grab mode |
| `MLX_GRAB_CACHE` | `~/.cache/mlx-middle-layer-grab` | local cache for downloaded grab repos |
| `MLX_FORCE_DEFAULT_MODEL` | `0` | always route to `DEFAULT_MODEL` |
| `MLX_SKIP_STARTUP_MODEL_PROMPT` | unset | skip TTY prompt for default model |
| `MLX_SKIP_STARTUP_ROOT_PROMPT` | unset | skip TTY prompt for model root |

### Generation

| Var | Default | Notes |
|-----|---------|-------|
| `DEFAULT_MAX_TOKENS` | `1024` | per-request fallback |
| `MAX_TOKENS_CEILING` | `16384` | hard cap |
| `GENERATION_TIMEOUT` | `300` | per-call wall clock (seconds) |
| `MLX_CONTEXT_OVER_BUDGET` | `error` (`error`/`trim`) | what to do when prompt exceeds context |
| `MLX_CONTEXT_TRIM_BUFFER` | `8` | tokens of slack when trimming |

### Routing

| Var | Default | Notes |
|-----|---------|-------|
| `DEFAULT_MODEL` | `""` | preferred alias for placeholder requests |
| `ON_MODEL_MISS` | `fallback` (`fallback`/`error`) | unknown alias policy |
| `MLX_ROUTE_LONG_PROMPT_CHARS` | `48000` | char threshold for capability routing |
| `MODEL_ROLES_JSON` | unset | inline JSON overrides `mlx_roles.json` |
| `MODEL_ROLES_FILE` | unset | path to roles JSON |
| `MODEL_PROFILES_FILE` | `model_profiles.json` next to script | per-alias capability profiles |

### Big-task heuristics (drives Anthropic auto-route)

| Var | Default |
|-----|---------|
| `BIG_TASK_MIN_WORDS` | `80` |
| `BIG_TASK_MIN_CHARS` | `500` |
| `BIG_TASK_MIN_BULLETS` | `4` |
| `BIG_TASK_MIN_STEP_MARKERS` | `3` |
| `ANTHROPIC_AUTO_ROUTE` | `1` (truthy strings) — **default on** |

### Anthropic / litellm

| Var | Default | Notes |
|-----|---------|-------|
| `ANTHROPIC_API_KEY` | unset | enables Opus escalation |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | server-config only (do not honour from request) |
| `ANTHROPIC_MODEL` | `claude-4-opus-20250522` | upstream model |
| `ANTHROPIC_VERSION` | `2025-04-14` | header `anthropic-version` |
| `USE_LITELLM_FOR_ANTHROPIC` | `1` (LM Studio backend only) | route Anthropic through LiteLLM |
| `ENABLE_LITELLM_PREFIX_ROUTING` | `1` (LM Studio backend only) | accept `provider/model` shape |
| `LITELLM_TIMEOUT_SECONDS` | `120` | LiteLLM call timeout |

### LM Studio (legacy backend)

| Var | Default | Notes |
|-----|---------|-------|
| `LM_STUDIO_URL` | `http://127.0.0.1:1234` | upstream URL |
| `MODEL_LIST_TTL` | `30` | seconds — model-list cache |

### Admission scheduler / queue

| Var | Default | Notes |
|-----|---------|-------|
| `MAX_PARALLEL_MODEL_CALLS` | `2` | global concurrent generation cap |
| `MLX_PER_MODEL_INFLIGHT_CAP` | `0` (unbounded) | per-alias generation cap |
| `MLX_PER_MODEL_ADMISSION_CAP` | `0` | **legacy alias for above** (shim) |
| `MLX_QUEUE_MAX_PER_MODEL` | `32` | queue depth per alias |
| `MLX_QUEUE_MAX_TOTAL` | `128` | global queue depth |
| `MLX_QUEUE_WAIT_TIMEOUT_SEC` | `20.0` | wait timeout |
| `MLX_QUEUE_RETRY_AFTER_SEC` | `2` | suggested retry hint |
| `MLX_QUEUE_RETRY_JITTER_SEC` | `1` | jitter on hint |
| `MLX_QUEUE_PRIORITY_MIN` | `-10` | clamp lower bound |
| `MLX_QUEUE_PRIORITY_MAX` | `10` | clamp upper bound |
| `MLX_QUEUE_DEFAULT_PRIORITY` | `0` | default priority |

### Swarm

| Var | Default | Notes |
|-----|---------|-------|
| `SWARM_PER_CALL_TIMEOUT` | `180` | per-agent timeout (sec) |
| `SWARM_FANOUT_TIMEOUT` | `0` (derive) | wall-clock cap (sec) |
| `SWARM_CHAT_ENABLED` | `1` | enable `model: swarmCouncil` etc. |
| `SWARM_CHAT_DEFAULT_MODELS` | `role:reasoner,role:coder,role:fast` | default ensemble |
| `SWARM_CHAT_DEFAULT_STRATEGY` | `best-of-n` | default vote |
| `SWARM_CHAT_DEFAULT_JUDGE` | `role:reasoner` | default judge |
| `SWARM_STREAM_CHUNK_CHARS` | `64` | synthetic-SSE chunk size for swarm stream |

### Dashboard

| Var | Default |
|-----|---------|
| `MLX_DASHBOARD_ENABLED` | `1` |
| `MLX_DASHBOARD_PREVIEW_CHARS` | `200` |
| `MLX_DASHBOARD_MAX_EVENTS` | `200` |
| `MLX_DASHBOARD_CAPTURE_PROMPTS` | `0` (off — keep this default) |
| `MLX_DASHBOARD_MAX_PROMPT_CHARS` | `8000` |
| `MLX_DASHBOARD_MAX_ERROR_CHARS` | `500` |

### Cache TTLs (legacy LM Studio backend only)

| Var | Default |
|-----|---------|
| `MODEL_LIST_TTL` | `30` |
| `CACHE_TTL_SECONDS` | `60` (hard-coded constant, not env) |

**Total env vars enumerated: 53** (52 user-facing + the legacy
`MLX_PER_MODEL_ADMISSION_CAP` shim). Pass 2 reorganises every one of
these into `Settings` sections; nothing new should be added until then.

## 5. Module-level constants of interest

| Constant | File | Notes |
|----------|------|-------|
| `PLACEHOLDER_MODELS` | both backends, lines `middle_layerMLX.py:219`, `middle_layer.py:63` | **Identical sets** — duplicated |
| `DEFAULT_MODEL_ROLES` | `middle_layerMLX.py:236`, `middle_layer.py:86` | Almost identical (MLX adds `deepseek` to reasoner) |
| `MLX_AVAILABLE` | `middle_layerMLX.py:96` | global; `_mlx_make_sampler` optional |
| `_litellm_import_error` | `middle_layer.py:14-17` | captured at import for `/healthz` reporting |
| `_cached_model_id`, `_cached_model_ids`, `_cached_model_ids_ts` | `middle_layer.py:47,56,57` | LM Studio caches |
| `_MODEL_PROFILES_DOC` | `middle_layerMLX.py:266` | lazy memoised JSON load |
| `_GRAB` | `middle_layerMLX.py` (set by `init_mlx_grab_model`) | (model, tokenizer, gen_lock, path, label) tuple in grab mode |
| `_admission_scheduler` | `middle_layerMLX.py` | global `_AdmissionScheduler` instance |
| `mlx_manager` | `middle_layerMLX.py` | global `MLXManager` instance |

## 6. JSON config files (schema)

### `model_profiles.json`

```jsonc
{
  "defaults": {                       // applied first
    "context_window": 128000,
    "default_max_tokens": 512,
    "max_tokens_ceiling": 16384,
    "temperature_default": 0.7,
    "top_p_default": 0.95,
    "supports_vision": false,
    "supports_tools": true,
    "supports_json_mode": false,
    "latency_tier": "medium",         // "fast" | "medium" | "slow"
    "memory_gb_estimate": 8.0
  },
  "aliases": {                        // exact-alias overlay
    "<alias>": { /* same shape as defaults, partial */ }
  },
  "patterns": [                       // first matching substring wins
    { "substring": "vl",     "profile": { ... } },
    { "substring": "vision", "profile": { ... } },
    { "substring": "70b",    "profile": { ... } }
  ]
}
```

### `mlx_roles.json`

```jsonc
{
  "_comment": "Resolver roles. Each list checked in order; first existing model wins.",
  "fast":     ["nemotron-mini-4b-instruct-mlx", "...", "qwen3.5-9b-optiq"],
  "coder":    ["qwen/qwen3-coder-next", "coder"],
  "reasoner": ["qwen3.6-40b-...-thinking", "nousresearch/hermes-4-70b", ...],
  "vision":   ["vl", "vision", "llava"],
  "default":  ["granite-4.1-8b", "qwen3.5-9b-optiq"]
}
```

The `_comment` key currently leaks into `/healthz` output as a "role"
with a literal docstring as its value. Pass 3 should ignore underscored
keys when loading.

## 7. Module / file dependency graph

```
                    ┌────────────────────────┐
                    │   model_profiles.json   │
                    │   mlx_roles.json        │
                    └───────────┬─────────────┘
                                │ (loaded by both)
            ┌───────────────────┼───────────────────────┐
            │                   │                       │
            ▼                   ▼                       ▼
  ┌──────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
  │  middle_layer.py │ │ middle_layerMLX.py │ │   mlx_dashboard.py  │
  │  (LM Studio +    │ │  (canonical MLX    │ │   (Blueprint;       │
  │   litellm path)  │ │   gateway)         │ │   imported by MLX)  │
  └────────┬─────────┘ └─────────┬──────────┘ └──────────┬──────────┘
           │ import requests     │ import mlx_lm         │ import flask
           │ import flask        │ import flask          │
           │ import litellm      │ import mlx_dashboard ─┘
           │                     │ import flask_cors? (optional)
           ▼                     │
   ┌────────────────┐            ▼
   │  LM Studio :1234│   ┌────────────────────┐
   └─────────────────┘   │  ~/.lmstudio/models │
                         │  + HF cache         │
                         └─────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │  api.anthropic.com  │  (optional, both)
                       └─────────────────────┘

  dashboard/index.html ───loaded by──▶ /dashboard/
  dashboard/app.js     ───loaded by──▶ /dashboard/app.js (static)
```

Crucially, `middle_layer.py` and `middle_layerMLX.py` **never import
from each other** — the swarm logic, resolver, big-task heuristics, and
Anthropic translator have been copied between them. This is the largest
refactor opportunity (Pass 3).

## 8. Duplicated code (Pass 3 targets)

Each entry below is a function or constant that appears in both
backends with effectively identical logic. Line numbers are
`middle_layerMLX.py` ↔ `middle_layer.py`.

| Symbol                                  | MLX line | LM Studio line | Notes                                                         |
|-----------------------------------------|----------|----------------|---------------------------------------------------------------|
| `PLACEHOLDER_MODELS`                    | 219      | 63             | **Identical** — extract to `routing/placeholders.py`          |
| `DEFAULT_MODEL_ROLES`                   | 236      | 86             | Near-identical (MLX adds `deepseek`) — extract                |
| `_load_model_roles`                     | 245      | 95             | Identical except `log.warning` vs `print` — extract           |
| `_is_placeholder`                       | 1422     | 255            | Identical — extract                                           |
| `_match_one`                            | 1430     | 264            | Identical — extract                                           |
| `_resolve_role`                         | 1443     | 278            | Identical — extract                                           |
| `resolve_model_alias` / `resolve_model_id` | 1454  | 290            | Different signatures, same intent — unify in `routing.resolver` |
| `_looks_like_code`                      | 1874     | 343            | Identical — extract to `utils.text`                           |
| `_is_big_task`                          | 1882     | 351            | Identical — extract                                           |
| `_extract_user_intent_text`             | 1898     | 379            | Identical — extract                                           |
| `_should_route_to_anthropic`            | 1920     | 409            | Identical — extract to `backends.anthropic`                   |
| `_openai_messages_to_anthropic`         | 1941     | 427            | Identical — extract                                           |
| `_anthropic_to_openai_chat_completion`  | 1983     | 489            | Identical — extract                                           |
| `_call_anthropic_chat`                  | 2011     | 836            | Differs: LM Studio version branches to LiteLLM. Unify behind a `Backend.anthropic.send()` strategy |
| `_extract_text`                         | 2126     | 874            | Identical — extract                                           |
| `_normalize_agent_spec`                 | 2143     | 887            | Identical — extract to `swarm.spec`                           |
| `_run_one_agent`                        | 2151     | 896            | Differs: MLX version calls `_mlx_chat_completion`, LM Studio calls `_lmstudio_chat_completion`. Behaviour is parallel — unify with backend dispatch |
| `_fanout`                               | 2205     | 933            | Differs: MLX adds wall-clock deadline + non-blocking shutdown |
| `_is_swarm_chat_model`                  | 2321     | 978            | Identical — extract                                           |
| `_run_swarm_chat_completion`            | 2334     | 991            | ~130 lines each, mostly identical — unify                     |
| `_swarm_body_to_sse_response`           | 2468     | 1098           | Identical — extract to `http.sse`                             |
| `_auth_guard`                           | 2550     | 576            | Identical (modulo `_ALLOWED_BEARER`) — extract to `auth.py`   |

Boolean env parsing (`.strip().lower() not in {"0","false","no","off"}`)
appears 8+ times across the three Python files; consolidate into one
`utils.bools.parse_bool` helper in Pass 2.

## 9. CLI surface (`middle_layerMLX.py`)

```
middle_layerMLX [serve|download]

serve --host HOST          [env HOST=127.0.0.1]
      --port PORT          [env PORT=5001]
      --grab REPO_OR_PATH  [env MLX_GRAB_MODEL]
      --display-name NAME  [env MLX_GRAB_DISPLAY_NAME=mlx]
      --no-grab            [ignore env grab; force multi-model]
      --model-root DIR     [override MLX_MODEL_ROOT]
      --preload "a,b"      [comma list; preload at startup]
      --no-pick-model      [skip TTY default-model prompt]

download REPO              [hf id, e.g. mlx-community/Qwen3-8B-MLX]
```

`middle_layer.py` has **no CLI** — it relies on env vars + bare
`python middle_layer.py`.

## 10. Five overlapping shell launchers

| Script | Backend | Purpose | Pass 1 disposition |
|--------|---------|---------|---------------------|
| `start_middle_layer.sh` | LM Studio | OpenClaw-aware launcher (PORT 5000) | merge into `scripts/start.sh --profile lmstudio` |
| `start_middle_layerMLX.sh` | MLX | Default MLX launcher (PORT 5001) | merge into `--profile mlx` |
| `start_middle_layerMLX_5001_stable.sh` | MLX | Stability profiles (`safe`/`balanced`/`faster`) | merge into `--profile mlx-stable[=safe|balanced|faster]` |
| `run_middle_layer_mlx.sh` | MLX | Bare 5-line passthrough | redirect to new `scripts/start.sh` |
| `run_with_venv.sh` | MLX | Bootstrap-or-reuse venv, then run | move logic into `scripts/dev-bootstrap.sh` |
| `setup_mlx.sh` | MLX | One-command setup (venv + pip + optional grab + serve) | keep as `scripts/setup.sh`, point at new layout |

All five duplicate the same venv-discovery cascade logic.

## 11. External Python deps (today)

From `requirements-mlx.txt` ∪ `requirements-mlx-gateway.txt` ∪ implicit:

| Package        | Required by                       | Pass-2 extras placement |
|----------------|-----------------------------------|-------------------------|
| `flask>=3.0`   | both backends, dashboard          | core                    |
| `requests>=2.28` | both backends                   | core (replaced by `httpx` in Pass 3) |
| `mlx-lm>=0.19` | `middle_layerMLX.py`              | `[mlx]`                 |
| `mlx-metal`    | transitive of `mlx-lm` on Apple Silicon | `[mlx]`           |
| `huggingface_hub>=0.20` | grab/download paths      | `[mlx]`                 |
| `flask-cors>=5.0` | optional in MLX                | core (it is small)      |
| `gunicorn>=22.0` | LM Studio backend production     | `[lmstudio]`            |
| `litellm`      | LM Studio backend Anthropic call  | `[lmstudio]` (and `[anthropic]`) |
| `numpy`, `safetensors`, `transformers`, `tokenizers`, `sentencepiece`, `protobuf` | transitive of `mlx-lm` | `[mlx]` |

## 12. Baseline summary

A live MLX gateway was started against the smallest available local
model (`mlx-community/Nemotron-Mini-4B-Instruct-4bit-mlx`,
≈ 2.6 GB). Every public route returned the expected status and the
responses are pinned under `docs/_internal/baseline/`. See
`docs/_internal/baseline/README.md` for the per-file index. A few
behavioral notes that should NOT be silently changed in later passes
without an explicit RFC:

1. The unauth response is the literal JSON `{"error": "Unauthorized"}`
   with HTTP 401 and no `WWW-Authenticate` header.
2. Stream responses always end with `data: [DONE]\n\n` even when an
   underlying error occurs.
3. The `X-Model-Resolution` response header carries the human reason a
   fallback alias was selected.
4. `MLX_PER_MODEL_ADMISSION_CAP` is read alongside the modern
   `MLX_PER_MODEL_INFLIGHT_CAP` and is reflected in `/healthz` under
   `mlx_per_model_admission_cap_legacy` for visibility.
5. Dashboard static HTML is auth-exempt; the JSON API under it is not.

Pass 0 acceptance: ✅ no source files modified; baseline curls saved;
this report and `RISK_REGISTER.md` exist.
