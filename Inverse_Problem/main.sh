#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/config.j2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VISUALIZE_ONLY="${VISUALIZE_ONLY:-false}"

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/eki_test.py" "$CONFIG_PATH")

if [ "$VISUALIZE_ONLY" = "true" ]; then
  CMD+=("--visualize-only")
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"
