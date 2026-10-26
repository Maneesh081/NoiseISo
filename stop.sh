#!/usr/bin/env bash
# Stop a running dashboard.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Stop the dashboard if it is running.

Usage: ./stop.sh
EOF
    exit 0
    ;;
esac

print_header "stop dashboard"
if pgrep -f "$SRC/app.py" >/dev/null 2>&1; then
  pkill -f "$SRC/app.py" || true
  echo "stopped dashboard (port ${PORT})"
else
  echo "dashboard not running"
fi
