#!/usr/bin/env bash
# Stable launcher for middle_layerMLX.py on port 5001.
# Recommended when large models or OOM/timeouts occur.

set -euo pipefail

ML_HOME="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$(cd "$ML_HOME/.." && pwd)"
cd "$ML_HOME"

# Prefer venvs that already contain mlx_lm (MiddleLayer first, then workspace root).
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

# Canonical local endpoint.
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5001}"

# Stability profiles:
#   safe     -> best for OOM mitigation (default)
#   balanced -> mild throughput bump after stable runs
#   faster   -> higher throughput; use only if memory remains healthy
PROFILE="${MLX_STABILITY_PROFILE:-safe}"
case "$PROFILE" in
  safe)
    : "${MAX_CONCURRENT_MODELS:=1}"
    : "${MAX_PARALLEL_MODEL_CALLS:=1}"
    : "${MAX_WORKERS:=1}"
    : "${MLX_PER_MODEL_INFLIGHT_CAP:=1}"
    : "${MLX_QUEUE_MAX_PER_MODEL:=8}"
    : "${MLX_QUEUE_MAX_TOTAL:=16}"
    : "${DEFAULT_MAX_TOKENS:=256}"
    : "${MAX_TOKENS_CEILING:=2048}"
    ;;
  balanced)
    : "${MAX_CONCURRENT_MODELS:=1}"
    : "${MAX_PARALLEL_MODEL_CALLS:=2}"
    : "${MAX_WORKERS:=2}"
    : "${MLX_PER_MODEL_INFLIGHT_CAP:=1}"
    : "${MLX_QUEUE_MAX_PER_MODEL:=12}"
    : "${MLX_QUEUE_MAX_TOTAL:=24}"
    : "${DEFAULT_MAX_TOKENS:=384}"
    : "${MAX_TOKENS_CEILING:=3072}"
    ;;
  faster)
    : "${MAX_CONCURRENT_MODELS:=1}"
    : "${MAX_PARALLEL_MODEL_CALLS:=2}"
    : "${MAX_WORKERS:=2}"
    : "${MLX_PER_MODEL_INFLIGHT_CAP:=1}"
    : "${MLX_QUEUE_MAX_PER_MODEL:=16}"
    : "${MLX_QUEUE_MAX_TOTAL:=32}"
    : "${DEFAULT_MAX_TOKENS:=512}"
    : "${MAX_TOKENS_CEILING:=4096}"
    ;;
  *)
    echo "Unknown MLX_STABILITY_PROFILE='$PROFILE' (use safe|balanced|faster)" >&2
    exit 2
    ;;
esac

export MAX_CONCURRENT_MODELS MAX_PARALLEL_MODEL_CALLS MAX_WORKERS
export MLX_PER_MODEL_INFLIGHT_CAP MLX_QUEUE_MAX_PER_MODEL MLX_QUEUE_MAX_TOTAL
export DEFAULT_MAX_TOKENS MAX_TOKENS_CEILING
export MLX_QUEUE_WAIT_TIMEOUT_SEC="${MLX_QUEUE_WAIT_TIMEOUT_SEC:-20}"
export MLX_CONTEXT_OVER_BUDGET="${MLX_CONTEXT_OVER_BUDGET:-trim}"
export MLX_CONTEXT_TRIM_BUFFER="${MLX_CONTEXT_TRIM_BUFFER:-16}"
export ON_MODEL_MISS="${ON_MODEL_MISS:-error}"

# Keep this path local-only by default and disable cloud auto-routing.
export ANTHROPIC_AUTO_ROUTE="${ANTHROPIC_AUTO_ROUTE:-0}"

if [ -z "${MODEL_ROLES_FILE:-}" ]; then
  if [ -f "$ML_HOME/mlx_roles.json" ]; then export MODEL_ROLES_FILE="$ML_HOME/mlx_roles.json"
  elif [ -f "$WS_ROOT/mlx_roles.json" ]; then export MODEL_ROLES_FILE="$WS_ROOT/mlx_roles.json"
  fi
fi

echo "Starting stable middle_layerMLX profile..."
echo "  MLX_STABILITY_PROFILE:     $PROFILE"
echo "  Endpoint:                  http://$HOST:$PORT"
echo "  MAX_CONCURRENT_MODELS:     $MAX_CONCURRENT_MODELS"
echo "  MAX_WORKERS:               $MAX_WORKERS"
echo "  MLX_PER_MODEL_INFLIGHT_CAP:$MLX_PER_MODEL_INFLIGHT_CAP"
echo "  DEFAULT_MAX_TOKENS:        $DEFAULT_MAX_TOKENS"
echo "  MAX_TOKENS_CEILING:        $MAX_TOKENS_CEILING"
echo "  ON_MODEL_MISS:             $ON_MODEL_MISS"
echo "  ANTHROPIC_AUTO_ROUTE:      $ANTHROPIC_AUTO_ROUTE"

exec python3 middle_layerMLX.py serve --host "$HOST" --port "$PORT"
