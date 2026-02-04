#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
EMAIL_SCRIPT="$PROJECT_ROOT/scripts/send_email.py"
EMAIL_SUBJECT="Model training cron failed"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

tmp_log="$(mktemp)"

DATE_FROM="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

today = datetime.now(ZoneInfo("America/Chicago")).date()
end = today - timedelta(days=1)
start = end - timedelta(days=2)
print(start.isoformat())
PY
)"

DATE_TO="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

today = datetime.now(ZoneInfo("America/Chicago")).date()
end = today - timedelta(days=1)
print(end.isoformat())
PY
)"

git_sha="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
echo "cron_train start: $(date -u +'%Y-%m-%dT%H:%M:%SZ') git_sha=${git_sha}" >> "$LOG_DIR/cron_train.log"

if ! command -v docker >/dev/null 2>&1; then
  echo "cron_train error: docker not found in PATH=$PATH" | tee -a "$LOG_DIR/cron_train.log" | tee "$tmp_log"
  echo "cron_train failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

set +e
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python scripts/fetch_nba_stats.py --date-from "$DATE_FROM" --date-to "$DATE_TO" \
  2>&1 | tee -a "$LOG_DIR/cron_train.log" | tee "$tmp_log"
fetch_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python scripts/train_baseline_model.py 2>&1 | tee -a "$LOG_DIR/cron_train.log" | tee -a "$tmp_log"
train_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python scripts/train_nn_model.py 2>&1 | tee -a "$LOG_DIR/cron_train.log" | tee -a "$tmp_log"
nn_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python scripts/train_online_ensemble.py --log-path data/monitoring/prediction_log.csv --out models/ensemble_weights.json \
  2>&1 | tee -a "$LOG_DIR/cron_train.log" | tee -a "$tmp_log"
ensemble_status=${PIPESTATUS[0]}
set -e

if [ "$fetch_status" -ne 0 ] || [ "$train_status" -ne 0 ] || [ "$nn_status" -ne 0 ] || [ "$ensemble_status" -ne 0 ]; then
  echo "cron_train failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

rm -f "$tmp_log"
