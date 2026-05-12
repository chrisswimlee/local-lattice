#!/bin/bash
# setup_mlx.sh — One-command setup for middle_layerMLX on any Apple Silicon Mac.
#
# Usage:
#   bash setup_mlx.sh                              # just install deps
#   bash setup_mlx.sh mlx-community/Qwen3-8B-MLX  # install + download + serve
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== middle_layerMLX setup ==="
echo "Working directory: $SCRIPT_DIR"

# Create venv if missing
if [ ! -d .venv ]; then
  echo "Creating Python venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements-mlx.txt -q

echo "Dependencies installed."

if [ -n "$1" ]; then
  echo ""
  echo "Downloading model: $1"
  python3 middle_layerMLX.py download "$1"
  echo ""
  echo "Starting server in grab mode..."
  python3 middle_layerMLX.py serve --grab "$1"
else
  echo ""
  echo "Setup complete. Next steps:"
  echo ""
  echo "  # Single model (e.g. on a remote machine):"
  echo "  python3 middle_layerMLX.py serve --grab mlx-community/Qwen3-8B-MLX"
  echo ""
  echo "  # Multi-model with swarm (e.g. on your M5 Max):"
  echo "  python3 middle_layerMLX.py serve --model-root ~/.lmstudio/models"
  echo ""
  echo "  # Download a model first:"
  echo "  python3 middle_layerMLX.py download mlx-community/Qwen3-8B-MLX"
fi
