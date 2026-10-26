#!/usr/bin/env bash
# Generate the synthetic fallback corpus (no download needed).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Generate a synthetic speech-like corpus (fallback when the real set is absent).

Usage: ./synth.sh [--force] [make_synth_data.py flags]

Refuses to run if data/ already contains a dataset unless --force is given,
because it overwrites clean_/noisy_ folders with synthetic audio.
EOF
    exit 0
    ;;
esac

require_venv

FORCE=0
if [ "${1:-}" = "--force" ] || [ "${1:-}" = "-f" ]; then
  FORCE=1
  shift
fi

if [ -d "$DATA/clean_trainset_28spk_wav" ] && [ "$FORCE" -ne 1 ]; then
  die "data/ already contains a dataset; pass --force to overwrite with synthetic audio"
fi

mkdir -p "$OUT"
print_header "synthetic data"
"$PY" "$SRC/make_synth_data.py" --out "$DATA" "$@"
