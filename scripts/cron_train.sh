#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/cron_train.log"
EMAIL_SCRIPT="$PROJECT_ROOT/scripts/ops/send_email.py"
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
echo "cron_train start: $(date -u +'%Y-%m-%dT%H:%M:%SZ') git_sha=${git_sha}" >> "$LOG_FILE"

if ! command -v docker >/dev/null 2>&1; then
  echo "cron_train error: docker not found in PATH=$PATH" | tee -a "$LOG_FILE" | tee "$tmp_log"
  echo "cron_train failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

build_api_image() {
  : > "$tmp_log"
  set +e
  docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" build api \
    2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"
  local build_status=${PIPESTATUS[0]}
  set -e
  if [ "$build_status" -eq 0 ]; then
    return 0
  fi

  if grep -Eiq "unable to lease content|lease does not exist" "$tmp_log"; then
    echo "cron_train detected buildkit lease error; pruning builder cache and retrying once." >> "$LOG_FILE"
    docker builder prune -f >> "$LOG_FILE" 2>&1 || true
    : > "$tmp_log"
    set +e
    docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" build api \
      2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"
    build_status=${PIPESTATUS[0]}
    set -e
  fi

  return "$build_status"
}

force_build="${CRON_FORCE_BUILD:-0}"
image_id="$(docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" images -q api 2>/dev/null || true)"
build_stamp="$LOG_DIR/.api_image_git_sha_train"
last_built_sha="$(cat "$build_stamp" 2>/dev/null || true)"
need_build=0
if [ "$force_build" = "1" ] || [ -z "${image_id:-}" ]; then
  need_build=1
elif [ -n "${git_sha:-}" ] && [ "$git_sha" != "${last_built_sha:-}" ]; then
  need_build=1
fi

if [ "$need_build" -eq 1 ]; then
  if ! build_api_image; then
    echo "cron_train failed during docker build: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
    if [ -f "$EMAIL_SCRIPT" ]; then
      "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
    fi
    rm -f "$tmp_log"
    exit 1
  fi
  if [ -n "${git_sha:-}" ]; then
    printf "%s" "$git_sha" > "$build_stamp"
  fi
else
  echo "cron_train using existing api image (set CRON_FORCE_BUILD=1 to rebuild)." >> "$LOG_FILE"
fi

: > "$tmp_log"

if ! docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  alembic upgrade head 2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"; then
  echo "cron_train failed during migration: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

: > "$tmp_log"

set +e
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.nba.fetch_nba_stats --date-from "$DATE_FROM" --date-to "$DATE_TO" \
  2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"
fetch_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ops.resolve_projection_outcomes --days-back 30 --decision-lag-hours 3 \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
resolve_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.train_baseline_model 2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
train_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.train_nn_model 2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
nn_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.train_xgb_model 2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
xgb_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.train_lgbm_model 2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
lgbm_status=${PIPESTATUS[0]}

# Generate OOF predictions and train meta-learner (advisory, non-blocking)
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.generate_oof_predictions 2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log" || true
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.train_meta_learner 2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log" || true

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.train_online_ensemble --source db --days-back 90 --log-path data/monitoring/prediction_log.csv --out models/ensemble_weights.json \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
ensemble_status=${PIPESTATUS[0]}

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ops.monitor_model_health --alert-email \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
# Health monitoring is advisory; don't fail the pipeline on it.

# Drift detection (advisory — logs report, exit code 1 = drift detected)
set +e
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ops.check_drift --recent-days 7 --baseline-days 30 --output data/reports/drift_report.json \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
drift_status=$?

# Conditional retune: if drift detected, re-run Optuna tuning (advisory, non-blocking)
if [ "$drift_status" -ne 0 ]; then
  echo "Drift detected — triggering conditional hyperparameter retune..." | tee -a "$LOG_FILE" | tee -a "$tmp_log"
  docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
    python -m scripts.ml.tune_hyperparams --model both --n-trials 30 \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log" || true
  docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
    python -m scripts.ml.tune_nn_hyperparams --n-trials 15 \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log" || true
fi

# Generate visual reports (advisory, non-blocking)
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.ml.generate_reports \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log" || true

set -e

if [ "$fetch_status" -ne 0 ] || [ "$resolve_status" -ne 0 ] || [ "$train_status" -ne 0 ] || [ "$nn_status" -ne 0 ] || [ "$xgb_status" -ne 0 ] || [ "$lgbm_status" -ne 0 ] || [ "$ensemble_status" -ne 0 ]; then
  echo "cron_train failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

rm -f "$tmp_log"
