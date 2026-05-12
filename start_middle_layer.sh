#!/bin/bash
# Deprecated: use ./scripts/start.sh --profile lmstudio
echo "DeprecationWarning: start_middle_layer.sh is deprecated; use ./scripts/start.sh --profile lmstudio" >&2
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/start.sh" --profile lmstudio "$@"
