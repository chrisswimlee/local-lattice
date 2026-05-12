# Changelog

All notable changes to MiddleLayer are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Until 1.0.0 the **HTTP surface** is treated as the public stability
boundary; the **internal Python API** is "use at your own risk" and
will be reorganised without notice during the 0.x line. Pass 9 will add
`docs/stability.md` with the formal declaration.

## [Unreleased]

Migration umbrella for everything between 0.1.0 and 0.2.0. See
`README.md` for the per-pass roadmap.

### Pass 2 (planned)
- Configuration consolidation under `middle_layer.config.Settings`
  (`pydantic-settings`).
- Documented deprecation of every flat env var in favour of
  `MIDDLE_LAYER_<SECTION>_<KEY>` while keeping the old names as
  shims with `DeprecationWarning`.

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

[Unreleased]: https://github.com/chrisswimlee/local-lattice/commits/main/
[0.1.0]: https://github.com/chrisswimlee/local-lattice/releases
