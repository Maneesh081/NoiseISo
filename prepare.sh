#!/usr/bin/env bash
# Prepare the real VoiceBank+DEMAND corpus: resample to 16 kHz + validation split.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Prepare the real VoiceBank+DEMAND dataset.

Usage: ./prepare.sh [STAGING_DIR] [--val_speakers p286,p287]

STAGING_DIR holds the original 48 kHz zips/folders (default: data/_raw).
Writes 16 kHz mono sets into data/:
  clean_trainset_28spk_wav, noisy_trainset_28spk_wav
  clean_val_wav, noisy_val_wav        (held-out speakers)
  clean_testset_wav, noisy_testset_wav (official test set, untouched by training)

Environment: VAL_SPEAKERS
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

STAGE="$DATA/_raw"
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  STAGE="$1"
  shift
fi
VAL="${VAL_SPEAKERS:-p286,p287}"

require_venv
[ -d "$STAGE" ] || die "staging folder not found: $STAGE"

print_header "prepare dataset"
echo "source : $STAGE"
echo "dest   : $DATA"
echo "val    : $VAL"
echo "workers: ${WORKERS:-4}"
"$PY" "$SRC/prepare_real_data.py" \
  --src "$STAGE" \
  --dst "$DATA" \
  --val_speakers "$VAL" \
  --workers "${WORKERS:-4}" \
  "$@"
