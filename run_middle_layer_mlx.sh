#!/usr/bin/env bash
# Minimal launcher for middle_layerMLX.py (OpenAI-compatible MLX gateway).
# Needs: Python 3, mlx-lm, flask, requests on PATH; MLX weights under MLX_MODEL_ROOT.
# For heavy models or unstable hosts, use start_middle_layerMLX_5001_stable.sh instead.
#
# Usage:
#   ./run_middle_layer_mlx.sh
#   ./run_middle_layer_mlx.sh --port 5001
#   HOST=0.0.0.0 PORT=5001 ./run_middle_layer_mlx.sh
#
# Optional dashboard: keep mlx_dashboard.py + dashboard/ next to this script.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5001}"
exec python3 middle_layerMLX.py serve "$@"
