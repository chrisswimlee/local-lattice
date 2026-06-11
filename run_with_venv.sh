#!/usr/bin/env bash
# Deprecated: create .venv and pip install -e ".[mlx]" then ./scripts/start.sh --profile mlx
echo "DeprecationWarning: run_with_venv.sh is deprecated; use ./scripts/start.sh --profile mlx after pip install -e '.[mlx]'" >&2
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
exec "$ML_HOME/scripts/start.sh" --profile mlx "$@"
