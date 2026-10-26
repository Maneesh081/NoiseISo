#!/usr/bin/env bash
# Score the trained checkpoint on the official 824-clip VoiceBank+DEMAND test set.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Evaluate the checkpoint on the official test set.

Usage: ./eval.sh [eval.py flags]

Computes STOI (noisy vs enhanced) for every test clip and writes:
  out/results.json
  out/results.csv
PESQ is skipped (no C compiler available to build the pesq package).
EOF
    exit 0
    ;;
esac

require_venv
require_checkpoint
[ -d "$DATA/clean_testset_wav" ] || \
  die "test data missing in $DATA (run ./prepare.sh first)"

mkdir -p "$OUT"
print_header "evaluation"
"$PY" "$SRC/eval.py" \
  --checkpoint "$OUT/checkpoints/best.pt" \
  --clean_dir "$DATA/clean_testset_wav" \
  --noisy_dir "$DATA/noisy_testset_wav" \
  --out_dir "$OUT" \
  "$@"
