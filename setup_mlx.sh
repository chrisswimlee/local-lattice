#!/bin/bash
# Deprecated: use pip install -e ".[mlx]" and ./scripts/start.sh --profile mlx
echo "DeprecationWarning: setup_mlx.sh is deprecated; use pip install -e '.[mlx]' and ./scripts/start.sh --profile mlx" >&2
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MiddleLayer MLX setup (legacy entry point) ==="
echo "Working directory: $SCRIPT_DIR"

if [ ! -d .venv ]; then
  echo "Creating Python venv..."
  python3 -m venv .venv
fi

# shellcheck source=/dev/null
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
  echo "  ./scripts/start.sh --profile mlx"
  echo "  # or with grab:"
  echo "  python3 middle_layerMLX.py serve --grab mlx-community/Qwen3-8B-MLX"
fi
