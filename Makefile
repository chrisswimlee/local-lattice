# MiddleLayer — developer Makefile
# Requires: Python 3.11+, pip

PYTHON ?= python3
VENV  ?= .venv
PIP   := $(VENV)/bin/pip
PY    := $(VENV)/bin/python

.PHONY: install install-mlx test test-mlx test-all lint fmt run run-mlx run-stable run-lmstudio docker clean

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

install: $(VENV)/bin/python
	$(PIP) install -e ".[all]"

install-mlx: $(VENV)/bin/python
	$(PIP) install -e ".[mlx,anthropic,dashboard,dev]"

# Default test target: fast suite (skips the slow MLX subprocess tests
# and any network-dependent tests). Run before every commit. ~5-10s.
test: $(VENV)/bin/python
	$(PY) -m pytest -m "not mlx and not network" $(PYTEST_ARGS)

# Opt-in MLX subprocess tests. Run when touching MLX gateway code
# (middle_layerMLX.py, mlx_dashboard.py, mlx_manager). ~30-60s.
test-mlx: $(VENV)/bin/python
	$(PY) -m pytest -m "mlx and not network" $(PYTEST_ARGS)

# Everything except network-dependent tests. ~40-70s.
test-all: $(VENV)/bin/python
	$(PY) -m pytest -m "not network" $(PYTEST_ARGS)

lint: $(VENV)/bin/python
	$(VENV)/bin/ruff check middle_layer
	$(VENV)/bin/mypy

fmt: $(VENV)/bin/python
	$(VENV)/bin/ruff format middle_layer

run: run-mlx

run-mlx:
	./scripts/start.sh --profile mlx

run-stable:
	./scripts/start.sh --profile stable

run-lmstudio:
	./scripts/start.sh --profile lmstudio

docker:
	@echo "Dockerfile is scheduled in Pass 7 (LM Studio + Anthropic path)." >&2
	@echo "Nothing to build yet." >&2
	@exit 1

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
