# MiddleLayer — developer Makefile
# Requires: Python 3.11+, pip

PYTHON ?= python3
VENV  ?= .venv
PIP   := $(VENV)/bin/pip
PY    := $(VENV)/bin/python

.PHONY: install install-mlx test lint fmt run run-mlx run-stable run-lmstudio docker clean

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

install: $(VENV)/bin/python
	$(PIP) install -e ".[all]"

install-mlx: $(VENV)/bin/python
	$(PIP) install -e ".[mlx,anthropic,dashboard,dev]"

test: $(VENV)/bin/python
	$(PY) -m pytest $(PYTEST_ARGS)

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
