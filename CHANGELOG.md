# Changelog

All notable changes to MiddleLayer are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the **HTTP surface** is treated as the public stability
boundary; the **internal Python API** is "use at your own risk" and
will be reorganised without notice during the 0.x line. Pass 9 will add
`docs/stability.md` with the formal declaration.

## [Unreleased]

### Added

- ``local-lattice-init`` (alias ``middle-layer-init``): probes LM Studio or MLX
  model directories, classifies chat models into ``role:*`` buckets, and writes
  ``lmstudio_roles.json`` / ``mlx_roles.json``. Detects Ollama but does not
  configure it yet.

## [0.3.1] — 2026-06-11

### Added

- README **Swarm in 60 seconds** walkthrough with mermaid diagrams, Python
  snippets, and a route cheat sheet.
- **`X-Lattice-*-Ms` timing headers** and structured ``lattice.request`` logs
  (`middle_layer/timing.py`; disable with ``LATTICE_LOG_TIMING=0``).

### Changed (LM Studio gateway: Pass 3 module extraction)

- Extracted LM Studio HTTP probes and chat into
  `middle_layer/lmstudio_client.py`.
- Extracted Anthropic / LiteLLM cloud escalation into
  `middle_layer/cloud_escalation.py`.
- Extracted Flask route handlers into `middle_layer/lmstudio_routes.py`;
  `middle_layer.py` is now a thin config + registration shell (~700 lines).
- Added unit tests for the new modules (`tests/test_lmstudio_client_module.py`,
  `tests/test_cloud_escalation_module.py`).
- Fixed `docs/capabilities.md` pipeline docs to match the live ``steps`` API.

## [0.3.0] — 2026-06-11

### Changed (MLX gateway: cleanup — dead field + queue_controls + unload UI)

Final polish pass closing the audit's lower-priority cleanup items:

- `MLXManager.loaded_models` entries are now `(model, tokenizer,
  gen_lock)` 3-tuples. The previous `last_used` timestamp was written
  on every cache hit and never read — LRU recency is tracked entirely
  via `OrderedDict.move_to_end`. Pinned with a regression test so a
  future refactor can't silently re-add the field.
- `_mlx_chat_completion` gained `queue_controls` and `request_id`
  kwargs; `_run_one_agent` and `_fanout` thread them through. Audit
  finding: the swarm fanout used to always pass `queue_controls=None`,
  silently dropping per-request priority / wait budgets that the
  HTTP handler had parsed from the request body.
- Dashboard UI: each loaded-model pill now has an unload (×) button
  wired to `DELETE /v1/models/<alias>` with a confirm dialog. Also
  surfaces `load_error_count` from the snapshot for quick triage.
  Operators no longer need a separate curl to free Metal RAM.

Tests: 2 new tests in `tests/test_mlx_loader.py` covering the
3-tuple shape regression pin and the queue_controls propagation
contract.

### Added (MLX gateway: discovery hardening + runtime registry rescan)

Closes the audit's discovery findings:

- `MLXManager._scan` previously had a bare `except Exception: pass`
  around publisher subdir scans, silently swallowing permission
  errors and broken symlinks. Operators saw "0 models found" with
  no clue why. Now logs at WARNING with the path and exception type.
  Root-level scan failures are also caught and logged the same way.
- `mlx_context_windows.json` malformed JSON used to be silently
  swallowed → operators never knew their per-model context-window
  hints weren't being applied. Now logs at WARNING.
- Discovery was startup-only: new model dirs required restart. New
  `MLXManager.rescan()` re-walks `MLX_MODEL_ROOT` and returns a
  `{added, removed, unchanged}` diff. Loaded models stay loaded
  (operator-controlled eviction); only the registry is refreshed.
- New `POST /dashboard/api/admin/rescan` endpoint (auth-required)
  triggers the rescan and returns the diff for dashboard / CLI use.
- `main()` no longer re-instantiates `MLXManager` when the chosen
  root resolves to the same abspath as the import-time manager —
  avoids a redundant full directory walk on every CLI serve.

Tests: 4 new tests in `tests/test_mlx_discovery.py` covering the
permission-error WARNING, malformed-context-windows WARNING,
rescan picking up new dirs, and rescan dropping removed dirs.

### Added (MLX gateway: focused test suites for discovery and admission)

Closes the audit's "zero MLX-specific test coverage for core load/
evict/admit paths" finding. The LM Studio gateway already has
comprehensive `test_resolver.py` / `test_swarm_intents.py` /
`test_concurrency.py` coverage; this brings the MLX side roughly
in parity.

- `tests/test_mlx_discovery.py` — 9 tests covering flat layout,
  publisher layout, mixed layouts, missing root, skip-non-config
  dirs, ignore files at root, env override semantics, missing
  context-windows file, and fresh-manager empty load-error state.
  All use real tmp directory trees so the actual ``_scan`` walk
  executes end-to-end.
- `tests/test_mlx_admission.py` — 7 tests covering admission-disabled
  no-op (cap=0 legacy mode), cap=1 serialization (worker thread
  proves blocking), per-model queue overflow 429, global queue
  overflow 429, wait-timeout 429, release decrements inflight,
  snapshot reports state for /healthz consumers.

### Fixed (cross-gateway: shared OOM classifier with word-boundary regex)

Closes the audit finding that MLX-native OOM exception strings were
classifying as `error_kind="unknown"` in structured swarm
`error_details` — the LM Studio gateway's `_OOM_PHRASES` list was
LM-Studio-targeted ("insufficient system resources", "would likely
overload your system") and didn't recognize MLX wording ("out of
memory", "MPS backend out of memory", "std::bad_alloc"). Also the
naive `any(marker in text)` substring check false-positived on
"zoom", "room", and any other word containing "oom".

- New canonical `middle_layer.swarm.is_probable_oom_error(exc)` uses
  a word-boundary regex (`(?ix) \b(?:out of memory|oom|mps backend
  out of memory|resource[\s_-]exhausted|killed|std::bad_alloc|
  allocation failed)\b`) plus the legacy LM-Studio phrase fragments.
- `middle_layerMLX.py` re-exports the shared helper as
  `_is_probable_oom_error` (back-compat alias) and drops its own
  in-file substring implementation.
- `classify_swarm_error()` uses the shared helper so MLX OOMs now
  classify as `error_kind="oom"` in swarm structured responses.

Tests: 35 new tests in `tests/test_oom_classification.py` covering
a parametrized true/false-positive matrix (CUDA, Metal, OS-level,
bare OOM, allocation-failed, LM Studio specific wording on the
true side; "zoom"/"room"/"broomstick"/timeout/auth/empty/None/non-
string on the false side) plus integration tests confirming MLX
OOM strings now bucket as `"oom"` in `classify_swarm_error`.

### Added (MLX gateway: end-to-end error observability)

Closes the audit's "load errors only visible in logs" and "non-stream
generation failures unlogged" findings.

- `MLXManager.get_recent_load_errors()` returns `{alias: {error, ts}}`
  for every alias with a sticky load failure. Backed by a new
  `_last_load_error_ts` dict updated in lockstep with
  `_last_load_errors` so operators can see when each failure happened.
- `/healthz` now includes `recent_load_errors` — the same snapshot.
  Operators can answer "why isn't model X serving?" without grep.
- Dashboard snapshot now includes `recent_load_errors` and a
  `load_error_count` for quick triage.
- `MLXManager.get_memory_stats()` also exposes
  `recent_load_errors_count` for compact health views.
- Non-stream generation 500s (both grab and multi-model paths) now
  log at WARNING with alias + request_id + exception class +
  elapsed_ms BEFORE the response is built. Previously these only
  showed up in dashboard `record_event` — invisible to operators
  who don't have the dashboard enabled.

Tests: 3 new tests in `tests/test_mlx_health.py` covering
`get_recent_load_errors` snapshot shape, `/healthz` exposing the
field, and `get_memory_stats` including the count.

### Changed (MLX gateway: honest generation timeout semantics)

Closes the audit's "timeout misleads operators and clients" finding.
`GENERATION_TIMEOUT` was documented as soft, but the codebase
contained three dead `except TimeoutError` handlers around the timed
generation helper that never fires, plus a `/healthz` field that
advertised the timeout as if enforced.

- Removed three dead 504 paths (grab chat, `_mlx_chat_completion`,
  non-streaming chat handler). Each is now a single
  `except Exception` that delegates to `_mlx_error_with_guidance`.
- Renamed `/healthz` field `generation_timeout_sec` to
  `generation_advisory_timeout_sec` to make the soft-budget
  semantics explicit. The legacy field is kept as an alias for
  one minor with a sibling `generation_timeout_sec_deprecated`
  field carrying the migration message (AGENTS.md rule 1).
- Documented in `_mlx_generate_text_timed`'s NOTE that `mlx_lm`
  generation is not safely cancellable mid-flight, so per-request
  `max_tokens` is the real budget control.

Hard generation cancellation is deliberately out of scope:
cancelling MLX mid-stream can leave KV cache in an inconsistent
state, and the `mlx_lm` public API does not currently offer a safe
interrupt path. Revisit if/when one lands.

Tests: 3 new tests in `tests/test_mlx_health.py` covering the new
field name, the deprecation-alias contract, and that
`_mlx_generate_text_timed` never raises TimeoutError (pins the
invariant against future regression).

### Fixed (MLX gateway: graceful HF downloads + grab/dashboard exclusion)

Three operator-facing safety gaps from the audit:

- `init_mlx_grab_model()` and `_download_model()` (the `download`
  CLI subcommand) let `huggingface_hub.snapshot_download` exceptions
  surface as raw tracebacks. Now wrapped in `try/except` with clean
  error strings, non-zero exit codes, and a partial-download hint
  pointing operators to delete-and-retry. The "config.json missing"
  case also gets a clearer message about MLX layout requirements.
- Dashboard `POST /dashboard/api/models/load` was not aware of grab
  mode: it would happily push extra models into the LRU even though
  the chat API only serves the grabbed model, wasting RAM. Now
  returns 400 with a "model loading is disabled in grab mode" error.
- Dashboard `POST /dashboard/api/models/load` used to drop the
  `MLXManager.get_last_load_error()` detail and return a generic
  `"could not load '<alias>'"` message. Now surfaces the full
  guided string (which already includes the OOM remediation hint
  when applicable) as a 503 with `error` set to the guided text.

Tests: 5 new tests in `tests/test_mlx_grab.py` covering grab init
HF failure → clean error string, grab init missing-config-after-
download error message, `download` subcommand failure → exit code 1,
dashboard load blocked in grab mode, dashboard load surfaces guided
error detail.

### Performance (MLX gateway: explicit Metal cache teardown after eviction)

Closes the audit finding that eviction was registry-only: dropping the
`OrderedDict` reference and trusting refcount + GC to free Metal
allocations. On macOS, that left peak RSS noticeably higher than
steady-state for minutes after a model swap.

- New `_try_clear_mlx_metal_cache()` helper feature-detects the
  available teardown API (`mx.metal.clear_cache` → `mx.clear_cache`)
  so this works across `mlx_lm` versions. All failures are swallowed
  at the cleanup-helper boundary — Metal teardown is opportunistic
  and never blocks eviction from completing.
- `_post_evict_cleanup(reason, alias)` runs after both LRU eviction
  (during a load) and explicit unload (including deferred-on-pin).
  Always invoked *outside* `_registry_lock` so Metal teardown
  doesn't block other manager operations.
- New `MLX_FORCE_GC_ON_EVICT=1` (default off) adds a `gc.collect()`
  after the Metal release for memory-tight Macs that want tighter
  immediate RSS reclamation. Off by default because `gc.collect()`
  is noticeable wall time and the Metal allocator usually reclaims
  promptly once Python refs drop.

Tests: 3 new tests in `tests/test_mlx_loader.py` covering: cleanup
fires on both LRU-eviction and explicit-unload paths, teardown
errors are swallowed so the eviction still completes, and deferred
unloads fire cleanup on release rather than on the original
`unload_model` call.

### Fixed (MLX gateway: pin in-flight models against eviction races)

Closes the audit's highest-severity correctness finding. The LRU
registry used to happily evict a model while a long-running
generation still held a reference to it, with two failure modes:

1. **RAM cap violation.** With `MAX_CONCURRENT_MODELS=1`, a streaming
   request holding model A plus a load of model B left both full
   weight sets resident until the stream ended.
2. **Per-alias serialization break.** After eviction + reload of the
   same alias, a second resident copy was created with a different
   `gen_lock`. Two generations could then race on different MLX
   instances of what the caller thought was "the same model" — a
   known cause of MLX KV-cache corruption.

Changes:
- New `MLXManager.acquire_inference_handle(alias)` context manager
  and lower-level `pin_alias` / `release_pin` methods. Every
  inference site now pins the alias for the duration of the call;
  the streaming path uses the explicit `pin/release` pair so the
  pin survives the function return that hands the SSE generator
  back to Flask.
- `_ensure_capacity_locked` now picks the oldest *unpinned* alias
  as the eviction victim. If every resident alias is pinned, the
  new load proceeds and exceeds the cap rather than deadlocking;
  a WARNING is logged so operators see they need to raise
  `MAX_CONCURRENT_MODELS` or reduce request concurrency.
- `unload_model` now returns `{"unloaded": bool, "deferred": bool}`.
  Unloading a pinned alias defers the actual drop until the last
  holder releases (HTTP `DELETE /v1/models/<alias>` returns 202
  Accepted in that case). The deferred eviction fires automatically
  in `release_pin` — operators do not need to retry.
- `_loading_locks` is now pruned on every eviction and unload,
  closing the unbounded-growth finding.

Tests: 8 new tests in `tests/test_mlx_loader.py`. Each runs in a
subprocess with a mocked `_mlx_load_model` so no real MLX weights
are touched. Tests cover: pin/release counter, eviction skipping
pinned aliases, deferred-unload fires on release, no duplicate
resident copy on reload-during-pin, `_loading_locks` pruning,
all-pinned over-cap with warning, exception-cleanup of the
inflight refcount, and unload-of-unloaded clean status.

### Changed (MLX gateway: boot validation + admission default flip)

Closes the audit findings on boot-time correctness gaps and the
admission scheduler being silently bypassed by default. AGENTS.md
rule 1: every env-var default change ships with a one-shot
`DeprecationWarning` so operators can pin the legacy behavior.

- **`MAX_CONCURRENT_MODELS` is now validated at startup.** `=0` used
  to `KeyError` on the first load because `_ensure_capacity_locked`
  called `OrderedDict.popitem(last=False)` on an empty dict. Now
  fails fast with an actionable message. Same validation for
  `MAX_PARALLEL_MODEL_CALLS` (must be >= 1) and
  `MLX_PER_MODEL_INFLIGHT_CAP` (must be >= 0).
- **`MLX_PER_MODEL_INFLIGHT_CAP` default flipped from `0` to `1`.**
  The legacy default disabled the admission scheduler entirely, so
  direct `python middle_layerMLX.py` invocations had no per-alias
  back-pressure beyond `gen_lock`'s implicit thread pile-up. New
  default matches the stable launcher. Unset emits a one-shot
  `DeprecationWarning`; pin the legacy behavior via
  `MLX_PER_MODEL_INFLIGHT_CAP=0`.
- **`MLX_PER_MODEL_ADMISSION_CAP` (historical name) honored with
  `DeprecationWarning`.** Now folded into the standard env-var
  fallback chain.
- **`MAX_WORKERS` deprecated and ignored.** Was logged as "Flask
  threads" but never wired — `app.run(threaded=True)` doesn't take
  a worker cap. Setting it now emits a `DeprecationWarning` pointing
  operators to upstream WSGI server config. The legacy default of
  `MAX_WORKERS=1` in the stable launcher is removed.

Tests: 10 new tests in `tests/test_mlx_boot.py` covering validation
rejections, default-flip with deprecation, legacy alias fallback,
and `MAX_WORKERS` warn-on-explicit-set behavior. All tests use a
subprocess harness so MLX init never leaks into the pytest process.

### Changed (LM Studio gateway: dynamic-by-default)

- **`PREFER_LOADED_MODELS` default flipped from `"1"` to `"strict"`.** With
  the legacy `"1"` (prefer-loaded with fall-through to the installed set),
  a role/`DEFAULT_MODEL` preference whose first substring matched an
  installed-but-not-loaded id would resolve to that id and cause LM Studio
  to silently JIT-load a model the operator never staged — frequently a
  giant MoE that would then evict the model they were actually using.
  The new `"strict"` default never falls through to the installed set, so
  resolution always lands on something LM Studio already has resident.
  Unset environments now emit a one-shot `DeprecationWarning` explaining
  how to pin the legacy behavior (`PREFER_LOADED_MODELS=1`). Legacy value
  removed in 0.2.0. The `scripts/start.sh --profile lmstudio` launcher
  already exported `strict`, so this only changes behavior for direct
  `python middle_layer.py` invocations.

### Changed (swarm chat: dynamic-by-default)

- **`SWARM_CHAT_DEFAULT_MODELS` default flipped from
  `"role:reasoner,role:coder,role:fast"` to `"auto"`.** The old default
  fanned every default-shaped swarm chat completion out to three role
  lookups, each of which (under the legacy `PREFER_LOADED_MODELS=1`)
  could JIT-load a different installed model. The new `"auto"` token
  expands to whatever LM Studio currently has resident at request time,
  matching the launcher behavior. Unset environments emit a
  `DeprecationWarning`; pin the legacy value via
  `SWARM_CHAT_DEFAULT_MODELS=role:reasoner,role:coder,role:fast`. Legacy
  value removed in 0.2.0.

### Added — Swarm intelligence effectiveness pass

The previous default of `SWARM_CHAT_DEFAULT_MODELS="auto"` quietly
turned default-shaped `swarmCouncil` calls into N-way fanouts against
every id `/v1/models` reported — including embedding models and
installed-but-not-loaded JIT candidates. With `MAX_PARALLEL_MODEL_CALLS=2`
that meant ~9 sequential batches per "swarm" call, no consensus, no
diversity benefit, and one wasted slot per embedding model loaded. This
pass turns swarm into a real swarm:

- **`auto` now expands against the truly-loaded, chat-capable set.**
  `run_swarm_chat_completion` and the LM Studio gateway's
  `_expand_swarm_models` wrapper both now prefer
  `get_loaded_lmstudio_model_ids` (LM Studio `/api/v0/models`,
  `state=loaded`) over `get_lmstudio_model_ids` (installed). The result
  is then filtered through a new `is_chat_capable_model_id` heuristic so
  embedding models (`text-embedding-*`, `nomic-embed-*`, `bge-*`, `e5-*`,
  …) never end up in a chat fanout. Falls back to the installed list
  with the same chat filter when the loaded probe is unreachable.
- **New `SWARM_CHAT_AUTO_MAX` env var (default `3`).** Caps how many
  loaded chat ids the `auto` / `loaded` / `*` sentinels expand to, so a
  default `swarmCouncil` against a box with 7 loaded models gives you a
  three-way swarm, not a seven-way slow queue. Three is the sweet spot
  for diversity-vs-cost (one reasoner + one coder + one fast); set to
  `0` to disable the cap. The dedicated `/swarm/fanout` HTTP endpoint
  intentionally does **not** apply this cap — callers that hit
  `/swarm/fanout` explicitly want every candidate.
- **`first-success` strategy now actually exits early.** `fanout` gained
  an `early_exit_on_first_success` parameter; when the first agent
  returns `ok=True` with non-empty text, pending peers are cancelled and
  the result is returned immediately. Previously `first-success` waited
  for every agent in the fanout to complete and just picked
  `successes[0]` in input order — the same total latency as
  `best-of-n` minus the judge call. `best-of-n` and `longest` still
  wait-for-all because the judge / max-by-length needs every candidate.
  `run_swarm_chat_completion` passes `early_exit=True` for both
  `first-success` and `fanout` intents; the dedicated `/swarm/fanout`
  HTTP endpoint always returns all candidates.
- **Default strategy kept at `best-of-n`.** With the curated 3-model
  default swarm and the embedding filter, the judge round-trip is no
  longer a wasted cost on a 17-way fanout — it's an actual consensus
  call across diverse models. Callers can still pass
  `swarm.strategy: "first-success"` per-request for latency-over-quality.

### Pass 4 — Security hardening (this section)

Closes the security debt flagged in 0.1.0 / 0.2.0 (`P4-01`, `P4-02`,
`P4-03` from `docs/_internal/RISK_REGISTER.md`, plus the missing size cap
and headers called out in `CHANGELOG.md` 0.1.0 §Security).

#### Security

- **Constant-time API key comparison.** Both gateway backends
  (`middle_layer.py` LM Studio proxy, `middle_layerMLX.py` MLX) and the
  dashboard (`mlx_dashboard.py`) now compare API keys with
  [`hmac.compare_digest`](https://docs.python.org/3/library/hmac.html#hmac.compare_digest)
  via the new `middle_layer.security.check_api_key` helper. Closes
  **P4-02** (timing-oracle on `==` / `!=` compares).
- **Deny-by-default on public bind.** The MLX and LM Studio entry points
  now refuse to start when `HOST` resolves to a non-loopback interface
  and `MIDDLE_LAYER_API_KEY` is unset. Override with
  `MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH=1` (logs a loud warning) only when
  an upstream proxy is enforcing authentication. Closes **P4-01**.
- **Dashboard model-load allowlist.** `/dashboard/api/models/load` now
  rejects aliases that contain disallowed characters (control chars,
  shell metacharacters) and additionally requires the alias to be in
  the live `mlx_manager.get_available_aliases()` set on disk. Returns
  400 with a clear error instead of silently invoking the loader on
  arbitrary input. Closes **P4-03**.
- **Request body size cap.** Both backends now set
  `MAX_CONTENT_LENGTH` from `MIDDLE_LAYER_MAX_REQUEST_BYTES` (default
  **10 MiB**). Oversize requests get a Flask-native 413 instead of
  consuming memory.
- **Standard security headers.** Every response carries
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, and
  `Cross-Origin-Resource-Policy: same-origin`. Dashboard responses
  additionally carry a strict
  `Content-Security-Policy: default-src 'self'; … frame-ancestors 'none'; object-src 'none'`.
  Existing values are not overwritten, so CORS-configured headers and
  custom routing headers continue to work.

#### Added

- **New module: `middle_layer/security.py`.** Shared, dependency-free
  primitives consumed by both backends and the dashboard: constant-time
  API key comparison, `Authorization: Bearer` extraction, public-bind
  detection, the safe-bind enforcement guard, request-size resolver, and
  the response-header / CSP application. Public API is the names listed
  in its `__all__`; everything else is private.
- **New env var: `MIDDLE_LAYER_MAX_REQUEST_BYTES`** (int, default
  `10485760` = 10 MiB). Maximum size of an HTTP request body. Invalid
  / non-positive values fall back to the default with a warning.
- **New env var: `MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH`** (bool, default
  off). Escape hatch for the deny-by-default public bind check. Use
  only behind a trusted reverse proxy.
- **Tests: `tests/test_security.py`.** 60 cases covering the helper
  module plus an integration test that wires the LM Studio backend's
  Flask test client through the auth guard, header pipeline, and
  413-on-oversize-body path. `make test` now exercises the security
  surface end-to-end on every run.

#### Notes for operators

- Existing configurations that already set `MIDDLE_LAYER_API_KEY` and
  `HOST=127.0.0.1` see **no behavior change** beyond the new headers
  and the 10 MiB body cap. The auth check is now constant-time, but
  the wire format (`X-API-Key`, `Authorization: Bearer`) is identical.
- Existing configurations that bound to `0.0.0.0` / a public interface
  *without* `MIDDLE_LAYER_API_KEY` set will now refuse to start. This
  was previously a silent vulnerability. Fix forward by setting an API
  key (recommended) or by setting `MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH=1`
  with a documented justification.

### Docs / positioning

- **README repositioning.** Project leads with *"capability layer between
  agents and LLM compute"* instead of *"MLX-native gateway"*. The MLX
  backend, swarm endpoints, and ops features are now framed as
  implementations of a capability protocol, not the product itself. No
  HTTP-surface or behavior changes.
- **New: [`docs/capabilities.md`](./docs/capabilities.md).** Formal,
  descriptive spec of the resolver grammar (`role:`, priority lists,
  wildcards, placeholders), the automatic capability inference
  (`needs_vision`, `min_context_window`, `prefers_fast`), the model
  profile schema, and the swarm endpoint contracts. Pins the surface
  for 0.x.
- **New: [`docs/why-lattice.md`](./docs/why-lattice.md).** Long-form
  "why this exists" positioning piece — capability routing as an
  agent-infra primitive, the case against hardcoded model identifiers
  in agent code.
- **New: [`docs/integrations/`](./docs/integrations/).** Drop-in
  examples for [OpenAI Agents SDK](./docs/integrations/openai-agents-sdk.md)
  and [LangGraph](./docs/integrations/langgraph.md), plus a per-folder
  README documenting the generic "point your framework at Lattice" pattern.

### Still open (deferred to Pass 5+)

- **Rate limiting** per IP / per API key. Needs a small token-bucket
  implementation or a new dependency (`flask-limiter`); deliberately
  out of scope for this pass.
- **HSTS / TLS termination guidance.** Meaningful only behind a TLS
  proxy; will be folded into the deployment docs in Pass 7.

### Pass 2 (planned)
- Configuration consolidation under `middle_layer.config.Settings`
  (`pydantic-settings`).
- Documented deprecation of every flat env var in favour of
  `MIDDLE_LAYER_<SECTION>_<KEY>` while keeping the old names as
  shims with `DeprecationWarning`.

## [0.2.0] — 2026-05-12

### Changed

- **PyPI distribution** is now **`local-lattice`** (was `middle-layer` during early
  Pass 1 drafts). Install with `pip install "local-lattice[mlx]"` etc.
- **Console scripts:** canonical **`local-lattice-mlx`** and **`local-lattice-lmstudio`**;
  **`middle-layer-mlx`** and **`middle-layer-lmstudio`** remain as identical
  entry-point aliases for OpenClaw and existing automation (scheduled removal
  after one minor with `DeprecationWarning` once callers migrate).
- **`[project.urls]`**, README, and CONTRIBUTING clone commands now use
  **`https://github.com/chrisswimlee/local-lattice`**.
- **Security / Code of Conduct:** placeholder `@*.invalid` addresses replaced with
  GitHub private vulnerability reporting and documented issue-based CoC intake.

### Docs

- README **Status** line bumped to 0.2.0; quickstart uses `local-lattice` paths.

## [0.1.0] — 2026-05-12

First release after the OSS-readiness migration's Pass 0 (discovery)
and Pass 1 (hygiene). The HTTP surface is **unchanged from the
internal-only version** that preceded this commit; every public route
captured by the Pass 0 baseline (`docs/_internal/baseline/` on
`pass/0-discovery`) replays byte-for-byte against this release.

### Added

- **License.** Apache-2.0 plus a `NOTICE` enumerating the optional
  third-party components.
- **Build system.** `pyproject.toml` with `hatchling`. Optional
  extras: `[mlx]`, `[lmstudio]`, `[anthropic]`, `[dashboard]`,
  `[dev]`, `[all]`.
- **Console scripts.** `middle-layer-mlx` and `middle-layer-lmstudio`
  resolve into the legacy top-level modules through a transitional
  shim in `middle_layer/cli.py`. Pass 3 will absorb the legacy code
  into the package and drop the shim.
- **Docs.** `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
  (Contributor Covenant 2.1), `SECURITY.md`, `CHANGELOG.md` (this
  file), `AGENTS.md`.
- **Makefile.** `make install / test / lint / fmt / run / docker /
  clean` so contributors do not need to remember commands.
- **Single launcher.** `scripts/start.sh --profile {mlx,lmstudio,stable}`
  replaces the five overlapping shell scripts. The legacy launchers
  remain as one-line shims that print a `DeprecationWarning` and
  exec the new entry point.
- **Configuration.** `EXTRA_PLACEHOLDER_MODELS` environment variable
  that contributes additional aliases to the resolver's
  "you-pick-a-model" set without recompiling. Defaults to the
  previous OpenClaw-coupled list so existing installations keep
  working; will default to empty in 0.2.0.

### Changed

- **Branding.** The dashboard title and header read "MiddleLayer
  Dashboard" instead of the previous internal codename.
- **Requirements files.** `requirements-mlx.txt` and
  `requirements-mlx-gateway.txt` are now thin compatibility shims
  that delegate to `pip install -e .[mlx]` and
  `pip install -e .[mlx,anthropic]` respectively. They will be
  removed in 0.2.0.

### Deprecated

- The OpenClaw-coupled placeholder model ids (`openclaw`,
  `middlelayer`, `middle-layer`, `middle_layer`, `mlxmiddlelayer`,
  `mlx-middle-layer`, `mlx_middle_layer`, `mlx`, `lmstudio`) are
  retained by default through `EXTRA_PLACEHOLDER_MODELS` for one
  minor version. Set `EXTRA_PLACEHOLDER_MODELS=""` to opt out today;
  the default becomes empty in 0.2.0.
- The five legacy launcher scripts (`start_middle_layer.sh`,
  `start_middle_layerMLX.sh`, `start_middle_layerMLX_5001_stable.sh`,
  `run_middle_layer_mlx.sh`, `run_with_venv.sh`, `setup_mlx.sh`)
  print a `DeprecationWarning` to stderr and forward to
  `scripts/start.sh`. They will be removed in 0.2.0.

### Removed

- `middle_layer.py.corrupted-backup` (half-formed `*** End Patch`
  diff that had been committed to the tree by accident).
- Committed `__pycache__/` directories.

### Security

- No exploitable changes in this release. The full pre-existing
  security debt is enumerated in `docs/_internal/RISK_REGISTER.md`
  on `pass/0-discovery`; remediation lands in Pass 4. Highlights of
  what is **not yet fixed in 0.1.0**:
  - API key compared with `==` (timing oracle) — Pass 4 P4-02.
  - Auth optional by default on any host — Pass 4 P4-01.
  - Dashboard `models/load` accepts arbitrary alias strings — Pass 4
    P4-03.
  - No request-size cap, no rate limit, no security headers, no CSP
    on the dashboard — Pass 4.
- Operators who deploy this release on anything other than
  `127.0.0.1` should set `MIDDLE_LAYER_API_KEY` and put TLS in
  front of the gateway. See [SECURITY.md](SECURITY.md).

[Unreleased]: https://github.com/chrisswimlee/local-lattice/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/chrisswimlee/local-lattice/releases/tag/v0.3.1
[0.3.0]: https://github.com/chrisswimlee/local-lattice/releases/tag/v0.3.0
[0.2.0]: https://github.com/chrisswimlee/local-lattice/releases/tag/v0.2.0
[0.1.0]: https://github.com/chrisswimlee/local-lattice/releases
