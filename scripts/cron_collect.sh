#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
EMAIL_SCRIPT="$PROJECT_ROOT/scripts/send_email.py"
EMAIL_SUBJECT="PrizePicks cron failed"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

tmp_log="$(mktemp)"

git_sha="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
echo "cron_collect start: $(date -u +'%Y-%m-%dT%H:%M:%SZ') git_sha=${git_sha}" >> "$LOG_DIR/cron.log"

if ! command -v docker >/dev/null 2>&1; then
  echo "cron_collect error: docker not found in PATH=$PATH" | tee -a "$LOG_DIR/cron.log" | tee "$tmp_log"
  echo "cron_collect failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

if ! docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python scripts/collect_prizepicks.py 2>&1 | tee -a "$LOG_DIR/cron.log" | tee "$tmp_log"; then
  echo "cron_collect failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

# Log predictions for the newest snapshot so the online ensemble can learn once outcomes arrive.
set +e
calibration_path="$(cd "$PROJECT_ROOT" && ls -1t data/calibration/forecast_calibration_*.json 2>/dev/null | head -n 1)"
if [ -n "${calibration_path:-}" ]; then
  docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
    python scripts/run_top_picks_ensemble.py --calibration "$calibration_path" --log-decisions \
    2>&1 | tee -a "$LOG_DIR/cron.log" | tee -a "$tmp_log"
else
  docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
    python scripts/run_top_picks_ensemble.py --log-decisions \
    2>&1 | tee -a "$LOG_DIR/cron.log" | tee -a "$tmp_log"
fi
score_status=${PIPESTATUS[0]}
set -e

if [ "$score_status" -ne 0 ]; then
  echo "cron_collect failed during scoring/logging: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

rm -f "$tmp_log"
