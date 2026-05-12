#!/bin/bash
# Start the MLX middle layer for OpenClaw.
# Listens on 5001 to match openclaw.json -> providers.mlxLocal.baseUrl.
# For OOM/timeouts or large models, prefer:
#   ./start_middle_layerMLX_5001_stable.sh

set -e

ML_HOME="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$(cd "$ML_HOME/.." && pwd)"
cd "$ML_HOME"

# Activate a venv that has mlx-lm + flask (MiddleLayer first, then workspace root).
_venv_has_mlx() { [ -f "$1/bin/activate" ] && "$1/bin/python" -c "import mlx_lm" 2>/dev/null; }
if _venv_has_mlx "$ML_HOME/.venv"; then source "$ML_HOME/.venv/bin/activate"
elif _venv_has_mlx "$WS_ROOT/.venv"; then source "$WS_ROOT/.venv/bin/activate"
elif _venv_has_mlx "$ML_HOME/middle_layer_venv"; then source "$ML_HOME/middle_layer_venv/bin/activate"
elif _venv_has_mlx "$WS_ROOT/middle_layer_venv"; then source "$WS_ROOT/middle_layer_venv/bin/activate"
elif [ -f "$ML_HOME/.venv/bin/activate" ]; then source "$ML_HOME/.venv/bin/activate"
elif [ -f "$WS_ROOT/.venv/bin/activate" ]; then source "$WS_ROOT/.venv/bin/activate"
elif [ -f "$ML_HOME/middle_layer_venv/bin/activate" ]; then source "$ML_HOME/middle_layer_venv/bin/activate"
elif [ -f "$WS_ROOT/middle_layer_venv/bin/activate" ]; then source "$WS_ROOT/middle_layer_venv/bin/activate"
fi

# Defaults; override anything by exporting before running this script.
export PORT="${PORT:-5001}"
export HOST="${HOST:-127.0.0.1}"
export MLX_MODEL_ROOT="${MLX_MODEL_ROOT:-}"
export MAX_CONCURRENT_MODELS="${MAX_CONCURRENT_MODELS:-2}"
export MAX_PARALLEL_MODEL_CALLS="${MAX_PARALLEL_MODEL_CALLS:-2}"
export DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-1024}"
export MAX_TOKENS_CEILING="${MAX_TOKENS_CEILING:-16384}"

# If the user has a roles JSON, use it. Safe to leave unset.
if [ -z "${MODEL_ROLES_FILE:-}" ]; then
  if [ -f "$ML_HOME/mlx_roles.json" ]; then export MODEL_ROLES_FILE="$ML_HOME/mlx_roles.json"
  elif [ -f "$WS_ROOT/mlx_roles.json" ]; then export MODEL_ROLES_FILE="$WS_ROOT/mlx_roles.json"
  fi
fi

echo "Starting MLX middle layer..."
echo "  Tip:                         Use ./start_middle_layerMLX_5001_stable.sh for crash-resistant defaults"
echo "  PORT:                       $PORT"
echo "  HOST:                       $HOST"
[ -n "$MLX_MODEL_ROOT" ] && echo "  MLX_MODEL_ROOT:             $MLX_MODEL_ROOT"
echo "  MAX_CONCURRENT_MODELS:      $MAX_CONCURRENT_MODELS"
echo "  MAX_PARALLEL_MODEL_CALLS:   $MAX_PARALLEL_MODEL_CALLS"
echo "  MAX_TOKENS_CEILING:         $MAX_TOKENS_CEILING"
[ -n "${MODEL_ROLES_FILE:-}" ] && echo "  MODEL_ROLES_FILE:           $MODEL_ROLES_FILE"
[ -n "${ANTHROPIC_API_KEY:-}" ] && echo "  Anthropic escalation:       enabled"

exec python3 middle_layerMLX.py serve --host "$HOST" --port "$PORT"
