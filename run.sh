#!/usr/bin/env bash
# Start the web dashboard (FastAPI + uvicorn).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Start the web dashboard.

Usage: ./run.sh [--port N] [--host HOST]

Then open:  http://localhost:7860
(The microphone in the Live tab requires localhost, not the LAN IP.)

Environment: PORT, HOST
EOF
    exit 0
    ;;
esac

require_venv
require_checkpoint

print_header "dashboard"
echo "Open http://localhost:${PORT}  (Ctrl+C to stop)"
exec "$PY" "$SRC/app.py" \
  --checkpoint "$OUT/checkpoints/best.pt" \
  --data "$DATA" \
  --out "$OUT" \
  --test_dir "$TEST_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
