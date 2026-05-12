#!/bin/bash
# Start the middle layer for OpenClaw (LM Studio proxy).

ML_HOME="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$(cd "$ML_HOME/.." && pwd)"
cd "$WS_ROOT"

if [ -f "$WS_ROOT/middle_layer_venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$WS_ROOT/middle_layer_venv/bin/activate"
elif [ -f "$ML_HOME/middle_layer_venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ML_HOME/middle_layer_venv/bin/activate"
else
  echo "middle_layer_venv not found under $WS_ROOT or $ML_HOME" >&2
  exit 1
fi

export TARGET_MODEL=${TARGET_MODEL:-qwen/qwen3.5-35b-a3b}
export DEFAULT_MODEL=${DEFAULT_MODEL:-$TARGET_MODEL}
export LM_STUDIO_URL=${LM_STUDIO_URL:-http://127.0.0.1:1234}
export PORT=${PORT:-5000}

# Pick up roles file the same way start_middle_layerMLX_5001_stable.sh does so
# role:fast / role:coder / role:reasoner resolve against the LM Studio models
# you actually have, not loose substrings like "3b" matching 35B MoE models.
if [ -z "${MODEL_ROLES_FILE:-}" ]; then
  if [ -f "$ML_HOME/mlx_roles.json" ]; then
    export MODEL_ROLES_FILE="$ML_HOME/mlx_roles.json"
  elif [ -f "$WS_ROOT/mlx_roles.json" ]; then
    export MODEL_ROLES_FILE="$WS_ROOT/mlx_roles.json"
  fi
fi

# Keep swarm fanout single-model by default — two big LM Studio models loading
# in parallel will OOM on most local boxes. Override with SWARM_CHAT_DEFAULT_MODELS
# in the environment if you want fanout/vote out of the box.
export MAX_PARALLEL_MODEL_CALLS="${MAX_PARALLEL_MODEL_CALLS:-1}"
export SWARM_CHAT_DEFAULT_MODELS="${SWARM_CHAT_DEFAULT_MODELS:-role:fast}"
export SWARM_CHAT_DEFAULT_STRATEGY="${SWARM_CHAT_DEFAULT_STRATEGY:-first-success}"

echo "Starting middle layer..."
echo "Target model:                   $TARGET_MODEL"
echo "LM Studio URL:                  $LM_STUDIO_URL"
echo "Port:                           $PORT"
echo "MODEL_ROLES_FILE:               ${MODEL_ROLES_FILE:-<none>}"
echo "MAX_PARALLEL_MODEL_CALLS:       $MAX_PARALLEL_MODEL_CALLS"
echo "SWARM_CHAT_DEFAULT_MODELS:      $SWARM_CHAT_DEFAULT_MODELS"
echo "SWARM_CHAT_DEFAULT_STRATEGY:    $SWARM_CHAT_DEFAULT_STRATEGY"

exec python3 "$ML_HOME/middle_layer.py"
