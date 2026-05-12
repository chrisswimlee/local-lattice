#!/usr/bin/env bash
# Deprecated: use ./scripts/start.sh --profile mlx
echo "DeprecationWarning: run_middle_layer_mlx.sh is deprecated; use ./scripts/start.sh --profile mlx" >&2
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/start.sh" --profile mlx "$@"
