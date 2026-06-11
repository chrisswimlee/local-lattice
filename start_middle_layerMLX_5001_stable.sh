#!/usr/bin/env bash
# Deprecated: use ./scripts/start.sh --profile stable
echo "DeprecationWarning: start_middle_layerMLX_5001_stable.sh is deprecated; use ./scripts/start.sh --profile stable" >&2
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/start.sh" --profile stable "$@"
