#!/usr/bin/env bash
# Unified launcher for MiddleLayer (replaces five legacy shell scripts).
# Usage:
#   ./scripts/start.sh --profile mlx
#   ./scripts/start.sh --profile lmstudio
#   ./scripts/start.sh --profile stable          # MLX + safe stability defaults
#   ./scripts/start.sh --profile stable=balanced # explicit stability tier
# Extra args after -- are forwarded to the backend (MLX: local-lattice-mlx / middle-layer-mlx / python).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE=""
STABLE_TIER="safe"
FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      shift
      PROFILE="${1:?--profile requires an argument}"
      if [[ "$PROFILE" == stable=* ]]; then
        STABLE_TIER="${PROFILE#stable=}"
        PROFILE="stable"
      fi
      shift
      ;;
    --)
      shift
      FORWARD+=("$@")
      break
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  echo "usage: $0 --profile {mlx|lmstudio|stable} [-- ...extra args...]" >&2
  exit 2
fi

_venv_has_mlx() { [[ -f "$1/bin/activate" ]] && "$1/bin/python" -c "import mlx_lm" 2>/dev/null; }

_activate_mlx_venv() {
  local ws_root
  ws_root="$(cd "$REPO_ROOT/.." && pwd)"
  if _venv_has_mlx "$REPO_ROOT/.venv"; then source "$REPO_ROOT/.venv/bin/activate"
  elif _venv_has_mlx "$ws_root/.venv"; then source "$ws_root/.venv/bin/activate"
  elif _venv_has_mlx "$REPO_ROOT/middle_layer_venv"; then source "$REPO_ROOT/middle_layer_venv/bin/activate"
  elif _venv_has_mlx "$ws_root/middle_layer_venv"; then source "$ws_root/middle_layer_venv/bin/activate"
  elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then source "$REPO_ROOT/.venv/bin/activate"
  elif [[ -f "$ws_root/.venv/bin/activate" ]]; then source "$ws_root/.venv/bin/activate"
  elif [[ -f "$REPO_ROOT/middle_layer_venv/bin/activate" ]]; then source "$REPO_ROOT/middle_layer_venv/bin/activate"
  elif [[ -f "$ws_root/middle_layer_venv/bin/activate" ]]; then source "$ws_root/middle_layer_venv/bin/activate"
  fi
}

case "$PROFILE" in
  mlx)
    _activate_mlx_venv
    export PORT="${PORT:-5001}"
    export HOST="${HOST:-127.0.0.1}"
    export MLX_MODEL_ROOT="${MLX_MODEL_ROOT:-}"
    export MAX_CONCURRENT_MODELS="${MAX_CONCURRENT_MODELS:-2}"
    export MAX_PARALLEL_MODEL_CALLS="${MAX_PARALLEL_MODEL_CALLS:-2}"
    export DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-1024}"
    export MAX_TOKENS_CEILING="${MAX_TOKENS_CEILING:-16384}"
    if [[ -z "${MODEL_ROLES_FILE:-}" ]]; then
      if [[ -f "$REPO_ROOT/mlx_roles.json" ]]; then export MODEL_ROLES_FILE="$REPO_ROOT/mlx_roles.json"
      elif [[ -f "$(cd "$REPO_ROOT/.." && pwd)/mlx_roles.json" ]]; then
        export MODEL_ROLES_FILE="$(cd "$REPO_ROOT/.." && pwd)/mlx_roles.json"
      fi
    fi
    echo "MiddleLayer (profile=mlx) — http://$HOST:$PORT"
    if command -v local-lattice-mlx >/dev/null 2>&1; then
      exec local-lattice-mlx serve --host "$HOST" --port "$PORT" "${FORWARD[@]}"
    fi
    if command -v middle-layer-mlx >/dev/null 2>&1; then
      exec middle-layer-mlx serve --host "$HOST" --port "$PORT" "${FORWARD[@]}"
    fi
    exec python3 "$REPO_ROOT/middle_layerMLX.py" serve --host "$HOST" --port "$PORT" "${FORWARD[@]}"
    ;;

  stable)
    export MLX_STABILITY_PROFILE="${STABLE_TIER}"
    case "${MLX_STABILITY_PROFILE}" in
      safe|balanced|faster) ;;
      *)
        echo "unknown stability tier '${MLX_STABILITY_PROFILE}' (use safe|balanced|faster)" >&2
        exit 2
        ;;
    esac
    _activate_mlx_venv
    export HOST="${HOST:-127.0.0.1}"
    export PORT="${PORT:-5001}"
    case "${MLX_STABILITY_PROFILE}" in
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
    esac
    export MAX_CONCURRENT_MODELS MAX_PARALLEL_MODEL_CALLS MAX_WORKERS
    export MLX_PER_MODEL_INFLIGHT_CAP MLX_QUEUE_MAX_PER_MODEL MLX_QUEUE_MAX_TOTAL
    export DEFAULT_MAX_TOKENS MAX_TOKENS_CEILING
    export MLX_QUEUE_WAIT_TIMEOUT_SEC="${MLX_QUEUE_WAIT_TIMEOUT_SEC:-20}"
    export MLX_CONTEXT_OVER_BUDGET="${MLX_CONTEXT_OVER_BUDGET:-trim}"
    export MLX_CONTEXT_TRIM_BUFFER="${MLX_CONTEXT_TRIM_BUFFER:-16}"
    export ON_MODEL_MISS="${ON_MODEL_MISS:-error}"
    export ANTHROPIC_AUTO_ROUTE="${ANTHROPIC_AUTO_ROUTE:-0}"
    if [[ -z "${MODEL_ROLES_FILE:-}" ]]; then
      if [[ -f "$REPO_ROOT/mlx_roles.json" ]]; then export MODEL_ROLES_FILE="$REPO_ROOT/mlx_roles.json"
      elif [[ -f "$(cd "$REPO_ROOT/.." && pwd)/mlx_roles.json" ]]; then
        export MODEL_ROLES_FILE="$(cd "$REPO_ROOT/.." && pwd)/mlx_roles.json"
      fi
    fi
    echo "MiddleLayer (profile=stable, MLX_STABILITY_PROFILE=$MLX_STABILITY_PROFILE) — http://$HOST:$PORT"
    if command -v local-lattice-mlx >/dev/null 2>&1; then
      exec local-lattice-mlx serve --host "$HOST" --port "$PORT" "${FORWARD[@]}"
    fi
    if command -v middle-layer-mlx >/dev/null 2>&1; then
      exec middle-layer-mlx serve --host "$HOST" --port "$PORT" "${FORWARD[@]}"
    fi
    exec python3 "$REPO_ROOT/middle_layerMLX.py" serve --host "$HOST" --port "$PORT" "${FORWARD[@]}"
    ;;

  lmstudio)
    ws_root="$(cd "$REPO_ROOT/.." && pwd)"
    cd "$ws_root"
    if [[ -f "$ws_root/middle_layer_venv/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "$ws_root/middle_layer_venv/bin/activate"
    elif [[ -f "$REPO_ROOT/middle_layer_venv/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "$REPO_ROOT/middle_layer_venv/bin/activate"
    else
      echo "middle_layer_venv not found under $ws_root or $REPO_ROOT" >&2
      exit 1
    fi
    export TARGET_MODEL="${TARGET_MODEL:-qwen/qwen3.5-35b-a3b}"
    export DEFAULT_MODEL="${DEFAULT_MODEL:-$TARGET_MODEL}"
    export LM_STUDIO_URL="${LM_STUDIO_URL:-http://127.0.0.1:1234}"
    export PORT="${PORT:-5000}"
    if [[ -z "${MODEL_ROLES_FILE:-}" ]]; then
      if [[ -f "$REPO_ROOT/mlx_roles.json" ]]; then export MODEL_ROLES_FILE="$REPO_ROOT/mlx_roles.json"
      elif [[ -f "$ws_root/mlx_roles.json" ]]; then export MODEL_ROLES_FILE="$ws_root/mlx_roles.json"
      fi
    fi
    export MAX_PARALLEL_MODEL_CALLS="${MAX_PARALLEL_MODEL_CALLS:-1}"
    export SWARM_CHAT_DEFAULT_MODELS="${SWARM_CHAT_DEFAULT_MODELS:-role:fast}"
    export SWARM_CHAT_DEFAULT_STRATEGY="${SWARM_CHAT_DEFAULT_STRATEGY:-first-success}"
    echo "MiddleLayer (profile=lmstudio) — PORT=$PORT LM_STUDIO_URL=$LM_STUDIO_URL"
    if command -v local-lattice-lmstudio >/dev/null 2>&1; then
      exec local-lattice-lmstudio "${FORWARD[@]}"
    fi
    if command -v middle-layer-lmstudio >/dev/null 2>&1; then
      exec middle-layer-lmstudio "${FORWARD[@]}"
    fi
    exec python3 "$REPO_ROOT/middle_layer.py" "${FORWARD[@]}"
    ;;

  *)
    echo "unknown profile '$PROFILE' (use mlx|lmstudio|stable)" >&2
    exit 2
    ;;
esac
