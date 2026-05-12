# Agent and automation conventions

This file is for humans **and** coding agents (Cursor, Copilot, etc.) working on
this repository.

## Non-negotiables

1. **Never break the running server** without a deprecation path: keep env-var
   aliases and emit `DeprecationWarning` with the new name for at least one
   minor version.
2. **One pass = one PR.** Small, reviewable. No drive-by changes outside the
   pass scope.
3. **Tests ship with behavior.** No "tests later" PRs.
4. **Conventional commits** only: `feat:`, `fix:`, `refactor:`, `chore:`,
   `docs:`, `test:`, `ci:`, `perf:`, `security:`, `build:`, `legal:`.
5. **No new env-var sprawl** after Pass 2: every knob goes in `Settings` and the
   README config table (Pass 2).
6. **Read before you write** on each pass: re-read files you will touch and run
   the server / baseline curls when behavior could change.

## Layout (transitional)

- `middle_layerMLX.py` — canonical MLX gateway (monolith until Pass 3).
- `middle_layer.py` — LM Studio proxy backend (monolith).
- `middle_layer/` — importable package; `cli.py` forwards to legacy modules.
- `mlx_dashboard.py` + `dashboard/` — runtime dashboard blueprint and static UI.

Pass 3 moves logic into `src/middle_layer/` (or `middle_layer/` subpackages) per
the roadmap in `README.md`.

## Commands

Prefer the `Makefile` targets (`make install`, `make test`, `make lint`). On
Apple Silicon, `make install` uses `pip install -e ".[mlx]"` from a `.venv` in
the repo root.

## Internal-only paths

`docs/_internal/` is gitignored. Do not commit Pass-0-style discovery artefacts
to `main`; regenerate locally when needed.

## URLs

The canonical GitHub home may move (org vs personal fork). If `README.md`,
`pyproject.toml` `[project.urls]`, and `git remote` disagree, treat that as a
**known TODO** until maintainers align them.
