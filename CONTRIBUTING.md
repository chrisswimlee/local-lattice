# Contributing to MiddleLayer

Thanks for thinking about contributing — MiddleLayer is in active
migration to a polished OSS release, and pull requests are very welcome,
especially against the open passes listed in [README.md](./README.md#project-status-and-roadmap).

This document is the operational handbook. The shorter pitch for
why-we-do-things-this-way lives in the [pass documentation under
`docs/_internal/`](./docs/_internal/) on the `pass/0-discovery` branch.

## Quick start for contributors

```bash
git clone https://github.com/middle-layer/middle-layer.git
cd middle-layer

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"            # cross-platform dev set
# On Apple Silicon also:
pip install -e ".[mlx]"

make test
make lint
```

If `make test` is green on a clean checkout, your environment is good.

## Development loop

| Task                  | Command                          |
|-----------------------|----------------------------------|
| Install (Apple)       | `make install`                   |
| Install (cross-plat)  | `pip install -e ".[all]"`        |
| Run the server        | `make run`                       |
| Run the test suite    | `make test`                      |
| Run a single test     | `pytest tests/path/to/test.py::name -x` |
| Lint                  | `make lint` (ruff + mypy)        |
| Auto-format           | `make fmt`                       |
| Build a wheel         | `python -m build`                |

The `Makefile` is the source of truth for these targets. Prefer it over
remembering raw commands.

## Pull-request shape

Pull requests are the unit of review. Each PR should be:

1. **Small enough to read in one sitting.** Roughly < 400 lines diff,
   excluding generated files and tests. If your change is larger,
   please split.
2. **One concern.** A PR either fixes a bug, adds a feature, refactors
   something, or improves docs — not multiple at once. Drive-by
   formatting fixes go in a separate `chore:` PR.
3. **Aligned with a pass.** The OSS-readiness work is broken into
   numbered passes (see the README roadmap). If your PR is part of a
   pass, the branch name should be `pass/<N>-<slug>` and the PR title
   should start with the pass number, e.g.
   `pass/4: tighten CORS and API-key comparison`. Out-of-pass work is
   welcome — just label it clearly.
4. **Backwards-compatible by default.** The brief calls this out
   explicitly: "Never break the running server." If you must rename an
   env var or change a response shape, keep the old name working with a
   `DeprecationWarning` for at least one minor version. The
   `MLX_PER_MODEL_ADMISSION_CAP → MLX_PER_MODEL_INFLIGHT_CAP` shim is
   the model to copy.

## Commit-message style

We use [Conventional Commits](https://www.conventionalcommits.org/).
Allowed types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`,
`ci`, `perf`, `security`, `build`, `legal`. A scope is encouraged but
not required.

```
feat(routing): add latency-tier preference to /v1/chat/completions

The capability resolver now reads `X-MLX-Latency-Tier` from the request
and prefers aliases whose profile.latency_tier matches. Falls back to
the previous behaviour when the header is absent. Documented in
docs/configuration.md.

Closes #42.
```

Bodies are encouraged when the diff has a non-obvious "why" — and
required for anything tagged `security:` or `legal:`.

Commits MUST NOT include obvious narrating comments in code (e.g.
`// increment counter`); use comments only when they explain intent,
trade-offs, or invariants that the code itself cannot convey.

## Tests land with behaviour

> "Tests land in the same PR as the behavior they cover. No 'tests will
> come later.'"

Pass 5 introduces the formal test framework. Until then, when you fix a
bug, add a regression test using the Pass-0 baseline curls in
`docs/_internal/baseline/` (on the `pass/0-discovery` branch) as ground
truth.

Useful conventions:

- Place tests under `tests/` mirroring the package layout (`tests/cli/`,
  `tests/routing/`, …).
- Mark slow tests with `@pytest.mark.slow`; CI runs `-m "not slow"` by
  default.
- Mark MLX-only tests with `@pytest.mark.mlx`; they are skipped on
  non-Apple-Silicon runners.
- Mark tests that hit a real upstream with `@pytest.mark.network`.

## Configuration changes

If your PR adds or changes a configuration knob:

1. Add it to the central `Settings` object (introduced in Pass 2). Do
   **not** add a new `os.environ.get(...)` at module import time.
2. Document it in `docs/configuration.md` (which is generated from the
   `Settings` schema after Pass 2). Until then, also add a row to the
   "Quick reference" table in [README.md](./README.md).
3. Add a `.env.example` entry with a safe default.
4. Provide a backwards-compat shim if you renamed an existing env var,
   and add a `DeprecationWarning` that names both the old and new
   variable.

## Coding conventions

- **Python ≥ 3.11.** Use modern type hints (`X | None`, `list[int]`).
- **`from __future__ import annotations`** at the top of every new
  module.
- **No bare `except:`.** Catch the narrowest exception you can.
- **No untyped `def`s in the new package.** `mypy` runs against
  `middle_layer/` and is intended to be `--strict` clean by end of
  Pass 3. The legacy top-level modules are temporarily exempt.
- **Comments explain "why", not "what".** Avoid narrating obvious code.
- **No new module-level globals.** Pass `Settings` and constructed app
  state through function arguments / Flask app context.
- **Outgoing HTTP always has an explicit `timeout=`.** Pass 4 will
  enforce this in CI.

`make fmt` runs `ruff format`; `make lint` runs `ruff check --fix` and
`mypy`. Both must be green before a PR is merged.

## Filing issues

A good bug report:

- The MiddleLayer version (`middle-layer-mlx --version`) and Python
  version.
- The relevant `Settings` (run `middle-layer-mlx config show` once
  Pass 2 ships) **or** the env vars you set, with secrets redacted.
- A minimal reproduction: ideally a `curl` command and the response.
- For routing issues, the contents of `model_profiles.json` and
  `mlx_roles.json` (or the `MODEL_ROLES_JSON` / `MODEL_ROLES_FILE`
  envs).

Security-sensitive reports go to the address in
[SECURITY.md](./SECURITY.md); please do **not** file them as public
GitHub issues.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](./CODE_OF_CONDUCT.md).
By participating, you agree to abide by its terms. Reports of CoC
violations go to the contact in that document.
