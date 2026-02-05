#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${API_LOG_DIR:-$PROJECT_ROOT/logs}"
LATEST_PID_FILE="$LOG_DIR/api_latest.pid"
LATEST_PORT_FILE="$LOG_DIR/api_latest.port"

if [ "${1:-}" != "" ]; then
  PORT="$1"
  PID_FILE="$LOG_DIR/api_${PORT}.pid"
else
  PID_FILE="$LATEST_PID_FILE"
  PORT="$(cat "$LATEST_PORT_FILE" 2>/dev/null || echo "unknown")"
fi

if [ ! -f "$PID_FILE" ]; then
  echo "No API pid file found at $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ ! "$PID" =~ ^[0-9]+$ ]]; then
  echo "Invalid pid in $PID_FILE"
  exit 1
fi

if ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "API process $PID is not running."
  rm -f "$PID_FILE"
  if [ "$PID_FILE" = "$LATEST_PID_FILE" ]; then
    rm -f "$LATEST_PORT_FILE"
  fi
  exit 0
fi

kill "$PID"
for _ in $(seq 1 20); do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

if kill -0 "$PID" >/dev/null 2>&1; then
  kill -9 "$PID"
fi

rm -f "$PID_FILE"
if [ "$PID_FILE" = "$LATEST_PID_FILE" ]; then
  rm -f "$LATEST_PORT_FILE"
fi

echo "Stopped API process $PID (port $PORT)"
