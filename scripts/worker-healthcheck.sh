#!/usr/bin/env sh
set -e
HEARTBEAT="/app/runtime/health/worker_heartbeat.txt"
if [ ! -f "$HEARTBEAT" ]; then
  exit 1
fi
NOW="$(date +%s)"
LAST="$(cat "$HEARTBEAT" | cut -d. -f1)"
AGE=$((NOW - LAST))
[ "$AGE" -lt 45 ]

