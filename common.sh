#!/usr/bin/env bash
# Shared environment for the project's task scripts. Source it; do not run it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Resolve the venv through its symlink. Running through /mnt/d/DL/.venv leaves
# sys.prefix on the slow 9p mount and makes imports ~10x slower.
VENV="$HERE/.venv"
if [ -L "$VENV" ]; then
  VENV="$(readlink -f "$VENV")"
fi
PY="$VENV/bin/python"
SRC="$HERE/src"
DATA="$HERE/data"
OUT="$HERE/out"
TEST_DIR="$HERE/test_noisy"
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

require_venv() {
  [ -x "$PY" ] || die "no Python interpreter at $PY (venv missing)"
}

require_checkpoint() {
  [ -f "$OUT/checkpoints/best.pt" ] || \
    die "no checkpoint at $OUT/checkpoints/best.pt (run ./train.sh first)"
}

print_header() {
  printf '\n== %s ==\n' "$*"
}
