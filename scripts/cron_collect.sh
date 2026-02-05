#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/cron.log"
EMAIL_SCRIPT="$PROJECT_ROOT/scripts/ops/send_email.py"
EMAIL_SUBJECT="PrizePicks cron failed"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

tmp_log="$(mktemp)"

git_sha="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
echo "cron_collect start: $(date -u +'%Y-%m-%dT%H:%M:%SZ') git_sha=${git_sha}" >> "$LOG_FILE"

if ! command -v docker >/dev/null 2>&1; then
  echo "cron_collect error: docker not found in PATH=$PATH" | tee -a "$LOG_FILE" | tee "$tmp_log"
  echo "cron_collect failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_error.log"
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
    echo "cron_collect detected buildkit lease error; pruning builder cache and retrying once." >> "$LOG_FILE"
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
build_stamp="$LOG_DIR/.api_image_git_sha_collect"
last_built_sha="$(cat "$build_stamp" 2>/dev/null || true)"
need_build=0
if [ "$force_build" = "1" ] || [ -z "${image_id:-}" ]; then
  need_build=1
elif [ -n "${git_sha:-}" ] && [ "$git_sha" != "${last_built_sha:-}" ]; then
  need_build=1
fi

if [ "$need_build" -eq 1 ]; then
  if ! build_api_image; then
    echo "cron_collect failed during docker build: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_error.log"
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
  echo "cron_collect using existing api image (set CRON_FORCE_BUILD=1 to rebuild)." >> "$LOG_FILE"
fi

: > "$tmp_log"

if ! docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  alembic upgrade head 2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"; then
  echo "cron_collect failed during migration: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

: > "$tmp_log"

if ! docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
  python -m scripts.prizepicks.collect_prizepicks 2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"; then
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
    python -m scripts.ml.run_top_picks_ensemble --calibration "$calibration_path" --log-decisions \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
else
  docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" run --rm -T api \
    python -m scripts.ml.run_top_picks_ensemble --log-decisions \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$tmp_log"
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
