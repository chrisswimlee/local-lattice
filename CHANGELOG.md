# Changelog

All notable changes to MiddleLayer are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the **HTTP surface** is treated as the public stability
boundary; the **internal Python API** is "use at your own risk" and
will be reorganised without notice during the 0.x line. Pass 9 will add
`docs/stability.md` with the formal declaration.

## [Unreleased]

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

[Unreleased]: https://github.com/chrisswimlee/local-lattice/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/chrisswimlee/local-lattice/releases/tag/v0.2.0
[0.1.0]: https://github.com/chrisswimlee/local-lattice/releases
