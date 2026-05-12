#!/usr/bin/env bash
# Create .venv if missing, install deps, run middle_layerMLX with that Python.
set -euo pipefail
ML_HOME="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$(cd "$ML_HOME/.." && pwd)"
cd "$ML_HOME"
if [[ -n "${MIDDLE_LAYER_VENV:-}" ]]; then
  VENV="$MIDDLE_LAYER_VENV"
elif [[ -f "$WS_ROOT/.venv/bin/python" ]]; then
  VENV="$WS_ROOT/.venv"
else
  VENV="$ML_HOME/.venv"
fi
if [[ ! -f "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q -U pip
"$VENV/bin/python" -m pip install -q -r "$ML_HOME/requirements-mlx-gateway.txt"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5001}"
exec "$VENV/bin/python" "$ML_HOME/middle_layerMLX.py" serve "$@"
