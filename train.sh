#!/usr/bin/env bash
# Train the U-Net on the real (16 kHz) dataset.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Train the enhancement U-Net.

Usage: ./train.sh [training.py flags]

Defaults: 40 epochs, early-stopping patience 8, 4 data workers, AMP on.
The previous checkpoint is backed up before training starts.
Uses data/clean_val_wav + data/noisy_val_wav when present (speaker hold-out);
otherwise validates on the official test set.
Console + out/train.log both receive the progress lines.
EOF
    exit 0
    ;;
esac

require_venv
[ -d "$DATA/clean_trainset_28spk_wav" ] || \
  die "training data missing in $DATA (run ./prepare.sh first)"

mkdir -p "$OUT/checkpoints"
# Start each run with a clean TensorBoard log so the dashboard's loss chart
# reflects this run only.
rm -rf "$OUT/tb"
if [ -f "$OUT/checkpoints/best.pt" ]; then
  backup="$OUT/checkpoints/best.$(date +%Y%m%d-%H%M%S).pt"
  cp "$OUT/checkpoints/best.pt" "$backup"
  echo "backed up previous checkpoint -> $backup"
fi

VAL_ARGS=()
if [ -d "$DATA/clean_val_wav" ] && [ -d "$DATA/noisy_val_wav" ] && \
   [ -n "$(ls -A "$DATA/clean_val_wav" 2>/dev/null || true)" ]; then
  VAL_ARGS=(--clean_val "$DATA/clean_val_wav" --noisy_val "$DATA/noisy_val_wav")
  echo "validation: speaker hold-out ($DATA/clean_val_wav)"
else
  echo "validation: official test set (no hold-out found)"
fi

print_header "training"
"$PY" "$SRC/train.py" \
  --data "$DATA" \
  --out "$OUT" \
  --epochs 40 \
  --patience 8 \
  --workers 4 \
  "${VAL_ARGS[@]}" \
  "$@" 2>&1 | tee "$OUT/train.log"
