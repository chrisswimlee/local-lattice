#!/bin/bash
# Deprecated: use ./scripts/start.sh --profile mlx
echo "DeprecationWarning: start_middle_layerMLX.sh is deprecated; use ./scripts/start.sh --profile mlx" >&2
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/start.sh" --profile mlx "$@"
