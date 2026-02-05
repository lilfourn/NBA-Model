#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${API_LOG_DIR:-$PROJECT_ROOT/logs}"
LATEST_PID_FILE="$LOG_DIR/api_latest.pid"
LATEST_PORT_FILE="$LOG_DIR/api_latest.port"

resolve_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    echo "$PYTHON_BIN"
    return
  fi
  if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "$PROJECT_ROOT/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "python3 not found; set PYTHON_BIN explicitly." >&2
  exit 1
}

pid_is_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

PYTHON_BIN="$(resolve_python)"

if [ -f "$LATEST_PID_FILE" ]; then
  existing_pid="$(cat "$LATEST_PID_FILE" 2>/dev/null || true)"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && pid_is_running "$existing_pid"; then
    existing_port="$(cat "$LATEST_PORT_FILE" 2>/dev/null || echo "unknown")"
    echo "API already running: pid=$existing_pid port=$existing_port"
    echo "URL: http://127.0.0.1:$existing_port"
    exit 0
  fi
fi

PORT="${API_PORT:-}"
if [ -z "$PORT" ]; then
  PORT="$("$PYTHON_BIN" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi

HOST="${API_HOST:-0.0.0.0}"
WORKERS="${API_WORKERS:-1}"
LOG_FILE="$LOG_DIR/api_${PORT}.log"
PID_FILE="$LOG_DIR/api_${PORT}.pid"

if "$PYTHON_BIN" - <<PY
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    in_use = sock.connect_ex(("127.0.0.1", int("${PORT}"))) == 0
sys.exit(0 if in_use else 1)
PY
then
  echo "Port $PORT is already in use. Set API_PORT to another value."
  exit 1
fi

nohup "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" >"$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" >"$PID_FILE"
echo "$PID" >"$LATEST_PID_FILE"
echo "$PORT" >"$LATEST_PORT_FILE"

for _ in $(seq 1 30); do
  if ! pid_is_running "$PID"; then
    echo "API failed to start. Log tail:"
    tail -n 40 "$LOG_FILE" || true
    exit 1
  fi
  if "$PYTHON_BIN" - <<PY
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    ready = sock.connect_ex(("127.0.0.1", int("${PORT}"))) == 0
sys.exit(0 if ready else 1)
PY
  then
    echo "API running in background: pid=$PID port=$PORT"
    echo "URL: http://127.0.0.1:$PORT"
    echo "Log: $LOG_FILE"
    exit 0
  fi
  sleep 0.5
done

echo "API process started but did not open port $PORT in time. Check log: $LOG_FILE"
exit 1
