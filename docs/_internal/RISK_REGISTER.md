# MiddleLayer — Risk Register (Pass 0)

> Living document. Each row is a concrete, named risk found while reading the
> repo or while running the Pass 0 baseline. Severity follows the scale below.
> Each risk lists the pass that will close it; PRs that fix a row must
> reference the ID here and the linked baseline file (if any) so a
> regression can be replayed.

## Severity scale

| Tag | Meaning |
|-----|---------|
| **C** (critical) | Exploitable bug, data leak, footgun that can affect deployed instances |
| **H** (high)     | Likely to bite an OSS contributor or operator within the first week |
| **M** (medium)   | Real correctness/perf/maintainability issue, not blocking adoption |
| **L** (low)      | Minor polish; does not affect correctness or trust |

## Risks

### Pass 1 — Project hygiene & legal foundation

| ID  | Sev | Risk | Evidence | Fix (Pass 1) |
|-----|-----|------|----------|--------------|
| P1-01 | H | Repo is **not a git repository**, so contributors have no history to fork from and CI cannot run. | `git status` → `fatal: not a git repository` | `git init`, set `main` branch, add `.gitignore` first commit. |
| P1-02 | H | No `LICENSE` file. The repo is currently legally unredistributable. | `ls` shows none | Confirm Apache-2.0 with the user; commit. |
| P1-03 | H | No `README`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, `SECURITY`, `CHANGELOG`. | `ls` | Add per the spec; README points at config table generated in Pass 2. |
| P1-04 | M | `__pycache__/` checked into the tree. | `ls __pycache__/` | `git rm -r __pycache__/`, add to `.gitignore`. |
| P1-05 | M | `middle_layer.py.corrupted-backup` is a half-formed `*** End Patch` diff committed alongside source. | file present, mode `-rw-------` | `git rm middle_layer.py.corrupted-backup`. |
| P1-06 | M | Two overlapping requirements files (`requirements-mlx.txt`, `requirements-mlx-gateway.txt`) with subtle differences (mlx, flask-cors, gunicorn pinning). | file diff | Move into `pyproject.toml` extras (`[mlx]`, `[lmstudio]`, `[anthropic]`); keep one-release shims. |
| P1-07 | M | Five overlapping shell launchers all re-implement venv discovery (`start_middle_layer.sh`, `start_middle_layerMLX*.sh`, `run_middle_layer_mlx.sh`, `run_with_venv.sh`, `setup_mlx.sh`). | files compared in CURRENT_STATE §10 | Consolidate into `scripts/start.sh --profile {mlx,lmstudio,stable}`; keep old names as one-line shims that warn. |
| P1-08 | M | Hardcoded "OpenClaw" branding leaks into placeholder model ids (`openclaw`, `middlelayer`, `mlxmiddlelayer`, …) and dashboard copy. | `middle_layerMLX.py:219`, dashboard messages | Move OpenClaw-specific aliases behind `EXTRA_PLACEHOLDER_MODELS` env or config; keep the generic `auto`/`default`/`""` set in core. |
| P1-09 | L | `dashboard/index.html` title is `middle_layerMLX` and copy says "Routing and throughput only". Acceptable, but rename to project name in Pass 1. | `dashboard/index.html:6,32` | Change title to `MiddleLayer Dashboard`. |
| P1-10 | L | Repo lives **inside** the parent `.openclaw/` folder; shell scripts reach across the boundary via `WS_ROOT="$(cd "$ML_HOME/.." && pwd)"`. | every `start_*.sh` | Document the OpenClaw integration as an *external example* in Pass 1; remove cross-boundary lookups in Pass 3. |

### Pass 2 — Configuration consolidation

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P2-01 | H | 53 distinct env vars read at module import time via direct `os.environ.get(...)` — impossible to override programmatically, no validation, no documentation surface. | CURRENT_STATE §4 | Pydantic `Settings` with sectioned models; one `os.environ` reference site (`middle_layer.config`). |
| P2-02 | H | `MLX_PER_MODEL_ADMISSION_CAP` (legacy) is silently aliased to `MLX_PER_MODEL_INFLIGHT_CAP` with no `DeprecationWarning`. | `middle_layerMLX.py:202-205` | Issue `DeprecationWarning` once at startup and keep alias for one minor version. |
| P2-03 | M | Boolean parsing (`.strip().lower() not in {"0","false","no","off"}`) is duplicated across 8+ env reads; the truthy/falsy sets diverge subtly between sites (e.g. `MLX_DASHBOARD_CAPTURE_PROMPTS` uses `not in (...)` of `("1","true","yes","on")` which inverts the polarity). | `middle_layerMLX.py:145,168,231,1538,1596`, `middle_layer.py:28,31,118`, `mlx_dashboard.py:21,27` | Single `parse_bool` helper + `Field(... )` validator. |
| P2-04 | M | `MODEL_ROLES_JSON` overrides `MODEL_ROLES_FILE` silently; ordering is implicit. | `middle_layerMLX.py:245-259` | Make precedence explicit and surface in `Settings`. |
| P2-05 | M | `MODEL_ROLES_FILE` JSON allows a `_comment` key whose string value is later listed as a "role" in `/healthz`. | baseline file `03_healthz_authed.txt` (and server log) | `_load_model_roles` should ignore underscore-prefixed keys. |
| P2-06 | M | `ANTHROPIC_AUTO_ROUTE` defaults to **on** even when no `ANTHROPIC_API_KEY` is set. The shell stable launcher then explicitly turns it off — implying the default is wrong. | `middle_layerMLX.py:145`, `start_middle_layerMLX_5001_stable.sh:78` | Default off; only auto-route if `anthropic.api_key` is set AND user opts in. |
| P2-07 | L | Magic strings (`"fallback"`, `"error"`, `"trim"`, `"fast"/"medium"/"slow"`) compared with raw `==`. | resolver, profiles | Replace with Enum types in Pass 2. |

### Pass 3 — Code restructure

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P3-01 | H | Two ~57 KB and ~154 KB single-file backends with **22 duplicated functions/constants** between them (full table in CURRENT_STATE §8). | both files | Extract into `src/middle_layer/` package per Pass 3 layout; LM Studio backend becomes `backends/lmstudio.py`. |
| P3-02 | H | `_handle_chat_request` (~167 LOC), `_run_swarm_chat_completion` (~134 LOC), `_handle_grab_chat` (~270+ LOC) exceed the 200-LOC ceiling and mix routing, generation, admission, and dashboard recording. | `middle_layerMLX.py:2703,2334,1125` | Split into named steps; add per-step docstring. |
| P3-03 | M | `requests` is used everywhere with **timeouts attached only sometimes** (`get_lmstudio_model_ids` has `timeout=5`, but the Anthropic POST in `middle_layer.py:855-869` has no explicit timeout). | grep `requests.(post\|get)` in both files | Switch to `httpx.Client` with mandatory `timeout=`. |
| P3-04 | M | `concurrent.futures.ThreadPoolExecutor` is created ad-hoc inside swarm functions — no shared shutdown discipline and no max-pool limit on the LM Studio backend. | `middle_layer.py:933`, `middle_layerMLX.py:2205` | Single `_fanout` helper; bounded executor; deadline-aware. |
| P3-05 | M | No type annotations on most public functions; no `py.typed`; cannot be used as a typed library. | grep `def .*\)` shows many untyped params | Add `from __future__ import annotations`, `mypy --strict` clean by end of Pass 3. |
| P3-06 | M | Module-level globals (`mlx_manager`, `_GRAB`, `_admission_scheduler`, `MLX_*_CAP`) make unit testing impossible without monkey-patching. | grep `^[a-z_]+ = ` in MLX file | Pass `Settings` and a constructed `App` into route handlers via Flask app context. |
| P3-07 | L | `_litellm_response_to_dict` uses `json.loads(json.dumps(resp, default=str))` as last resort. | `middle_layer.py:146-154` | Replace with explicit Pydantic conversion. |

### Pass 4 — Security hardening

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P4-01 | **C** | Auth optional by default. With no `MIDDLE_LAYER_API_KEY`, every route is open. | `middle_layerMLX.py:2551-2552`, `mlx_dashboard.py:43-49` | When `host != 127.0.0.1` and no key set: refuse to start unless `--allow-anonymous-public` is passed. |
| P4-02 | **C** | API key compared with `!=` — vulnerable to timing oracle. | `middle_layerMLX.py:2562`, `middle_layer.py` `_auth_guard`, `mlx_dashboard.py:49` | Use `hmac.compare_digest`. Reject empty string at config load. |
| P4-03 | **C** | `POST /dashboard/api/models/load` accepts arbitrary alias strings and passes them straight to `mlx_lm.load(...)`. With a leaked API key an attacker can pull arbitrary HF repos onto disk. | `mlx_dashboard.py:335-355`, `middle_layerMLX.py:988+` | Validate against `^[A-Za-z0-9._/-]{1,200}$` AND a configurable `MLX_MODEL_ALLOWLIST` (HF org list or substring list). |
| P4-04 | **C** | No `Path.resolve()` containment check on aliases passed to `MLXManager.load_model`; a path like `../foo/bar` could escape `MLX_MODEL_ROOT`. | `middle_layerMLX.py:907-940, 988+` | Resolve to absolute path and verify `Path.resolve().is_relative_to(MLX_MODEL_ROOT)` before load. |
| P4-05 | H | No request body size limit. A 60 000-char prompt was accepted in baseline (`26_chat_oversize_prompt.txt`). Easy DoS. | baseline file | Set Flask `MAX_CONTENT_LENGTH` (default 4 MiB; configurable via `server.max_request_bytes`). |
| P4-06 | H | No rate limit. | grep | Add `flask-limiter` (or extend admission scheduler). Default: 60 req/min/IP for `/v1/*`, 6 req/min for `/swarm/*`. |
| P4-07 | H | `dashboard/app.js` builds DOM via `innerHTML` after a hand-rolled `esc()` filter. Any new field that forgets to call `esc()` is XSS. | `dashboard/app.js:71-141` | Switch to `textContent` for data fields; add strict CSP header `default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'`. |
| P4-08 | H | No security headers. No `X-Content-Type-Options`, no `Referrer-Policy`, no CSP, no `Permissions-Policy`. | grep `headers` | Add `secure` package or simple `after_request` hook. |
| P4-09 | H | CORS, when enabled, allows `Authorization` header on all origins (`Access-Control-Allow-Origin: <CORS_ORIGINS>` with `*` literally permitted). With auth enabled this is a credential-leak hazard. | `middle_layerMLX.py:866-886` | If `CORS_ORIGINS=="*"` AND auth is enabled → loud warning at startup; refuse to allow `Authorization` in preflight. |
| P4-10 | H | Errors include raw upstream messages and possibly stack traces in JSON bodies. `_mlx_error_with_guidance` interpolates `str(exc)`. | `middle_layerMLX.py:856-860` | Uniform error envelope; full traceback only in server log with `request_id`. |
| P4-11 | M | `ANTHROPIC_BASE_URL` is server-config only today, but there is no test or comment guarding against future request-controlled override. | `middle_layerMLX.py:142, 2011+` | Add explicit unit test that asserts the config knob is never read from request data. |
| P4-12 | M | `MLX_DASHBOARD_CAPTURE_PROMPTS=1` would store raw user prompts in memory with no redaction. Default is correctly off, but there is no `MLX_DASHBOARD_REDACT_PATTERNS` even when on. | `mlx_dashboard.py:27-30,103-117` | Add regex list (emails, sk- keys, JWT-shape, credit-card-shape) applied to both prompt and completion previews. |
| P4-13 | M | Anthropic POST in `middle_layer.py:836-869` has **no explicit `timeout=`**; only LiteLLM path has `LITELLM_TIMEOUT_SECONDS`. | `middle_layer.py:854-869` | Add explicit timeout to every outbound call; CI test that imports module and asserts via monkeypatch. |
| P4-14 | M | `_log_request` logs path/status/latency/model. It does **not** log the request body, but the dashboard does (when capture enabled). Document this distinction. | `middle_layerMLX.py:2573-2581` | Add explicit doc + test asserting request body is never logged by stdlib logger. |
| P4-15 | M | `/dashboard/api/snapshot` includes the value of `default_model_env` and `model_roles` — neither is sensitive today, but if config grows, redaction is needed. | `mlx_dashboard.py:201-235` | Always run snapshot through a redactor; introduce `Settings.redacted()`. |
| P4-16 | M | When `flask-cors` is unavailable, the manual fallback registers a wildcard `OPTIONS /<path:path>` handler that reflects `CORS_ORIGINS` literally. If `CORS_ORIGINS="*"`, this echoes `*` in `Access-Control-Allow-Origin`, which combined with credentials would be invalid (and is a footgun). | `middle_layerMLX.py:864-886` | Reject `*` + credentials combo with a clear error. |
| P4-17 | M | The dashboard HTML stores the API key in `sessionStorage` and sends it via `X-API-Key`. `sessionStorage` is XSS-readable. | `dashboard/app.js:1-19` | After CSP is in place, `sessionStorage` becomes acceptable. Document trade-off in SECURITY.md. |
| P4-18 | M | No `HSTS`, no `Strict-Transport-Security` advertisement when behind TLS termination. | grep | Pass 4 adds `Strict-Transport-Security: max-age=63072000; includeSubDomains` when `X-Forwarded-Proto: https` present. |
| P4-19 | L | Dashboard polls `/dashboard/api/snapshot` every 2 s — denies forensic answers and stresses the snapshot lock. | `dashboard/app.js:210` | Replaced by SSE in Pass 6. |
| P4-20 | L | `bandit -r .` and `pip-audit` not part of any workflow. | absent | Wire into Pass 5 CI. |

### Pass 5 — Tests, types, linting, CI

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P5-01 | H | Zero tests. | `ls tests/` returns nothing | Establish `pytest` baseline using `docs/_internal/baseline/` as fixtures. |
| P5-02 | H | No CI. A single typo can break the server today. | no `.github/` | GitHub Actions: matrix `{macos-14, ubuntu-latest} × {3.11, 3.12, 3.13}`. |
| P5-03 | M | No formatter or linter pinned. | absent | `ruff` + `mypy --strict`. |
| P5-04 | M | The admission scheduler has subtle priority/queue logic with no concurrency tests. | `middle_layerMLX.py:457-696` | Targeted threading tests. |
| P5-05 | M | The resolver has many branches (placeholder, exact, comma list, role, substring, fallback policy) and is invoked from at least three callsites — no tests. | `middle_layerMLX.py:1454-1525` | Property-test using `hypothesis`. |
| P5-06 | M | The SSE stream contract (`data: ...\n\n`, `[DONE]` sentinel) has only the baseline curl as evidence. | `12_chat_stream.txt` | Snapshot test. |

### Pass 6 — Observability

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P6-01 | H | Stdlib `logging.basicConfig` only — text format, no `request_id`, no per-route latency, no `model` binding. | `middle_layerMLX.py:111-116` | Replace with `structlog`; bind context per request. |
| P6-02 | H | No `/metrics`. Operators have to read the dashboard JSON to know anything. | absent | Add `prometheus-client`; auth-gated `/metrics`. |
| P6-03 | M | No tracing. | absent | Optional OpenTelemetry behind `OTEL_ENABLED=true`. |
| P6-04 | M | `/healthz` returns the same status before and after a model is preloaded. There is no `/readyz`. | `middle_layerMLX.py:2589-2644` | Split into `healthz` + `readyz`. |

### Pass 7 — Distribution

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P7-01 | H | No installable package — `pyproject.toml` absent, console script absent. Users must `git clone` to use it. | grep | Pass 7 publishes to PyPI as `middle-layer`. |
| P7-02 | M | No `Dockerfile` for the LM Studio + Anthropic profile. (MLX path stays Mac-only.) | absent | Multi-stage `python:3.12-slim`, non-root `USER`, healthcheck. |
| P7-03 | M | No `.devcontainer/` for contributors. | absent | Add minimal devcontainer. |
| P7-04 | L | No release automation. | absent | `release-please` or `git-cliff` driven by conventional commits. |

### Pass 8 — Docs & dashboard UX

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P8-01 | M | No OpenAPI spec; clients have to read source to know route shapes. | absent | Hand-write `docs/openapi.yaml`; serve at `/openapi.json`. |
| P8-02 | M | Dashboard UX is plain `setInterval(2s)` polling and bare HTML. | `dashboard/app.js` | Rewrite with HTMX + Alpine, sparkline charts, kill-switch. |
| P8-03 | M | No quickstart / examples folder for downstream clients (Continue, Cline, etc.). | absent | Add `examples/`. |
| P8-04 | L | No architecture diagram. | absent | Add `docs/architecture.svg`. |

### Pass 9 — Polish & 1.0

| ID  | Sev | Risk | Evidence | Fix |
|-----|-----|------|----------|-----|
| P9-01 | M | Public API surface (HTTP routes vs internal Python API) is not declared anywhere. | absent | `docs/stability.md`. |
| P9-02 | M | No public roadmap. | absent | `docs/roadmap.md`. |
| P9-03 | L | Conventional commits not enforced. | n/a | `commitizen` + `release-please`. |

## Summary by severity

| Sev | Count |
|-----|------:|
| **C** (critical) | 4  (P4-01, P4-02, P4-03, P4-04) |
| **H** (high)     | 16 |
| **M** (medium)   | 27 |
| **L** (low)      | 8  |
| **Total**        | 55 |

The 4 criticals all collapse into Pass 4. None of them are exploitable
remotely **today** without first leaking the (currently optional) API
key, but every one becomes dangerous the moment a contributor exposes
the gateway on `0.0.0.0`.
