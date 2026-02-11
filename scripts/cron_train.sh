#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/cron_train.log"
EMAIL_SCRIPT="$PROJECT_ROOT/scripts/ops/send_email.py"
EMAIL_SUBJECT="Model training cron failed"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

tmp_log="$(mktemp)"

git_sha="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
echo "cron_train start: $(date -u +'%Y-%m-%dT%H:%M:%SZ') git_sha=${git_sha}" >> "$LOG_FILE"

modal_cmd=()
venv_python="$PROJECT_ROOT/.venv/bin/python"
resolved_modal_bin="${MODAL_BIN:-}"
if [ -x "$venv_python" ] && "$venv_python" -c "import modal" >/dev/null 2>&1; then
  modal_cmd=("$venv_python" "-m" "modal")
elif [ -n "$resolved_modal_bin" ] && [ -x "$resolved_modal_bin" ]; then
  modal_cmd=("$resolved_modal_bin")
elif command -v modal >/dev/null 2>&1; then
  modal_cmd=("$(command -v modal)")
else
  for candidate in \
    "$HOME/miniconda3/bin/modal" \
    "$HOME/anaconda3/bin/modal" \
    "$HOME/.local/bin/modal"
  do
    if [ -x "$candidate" ]; then
      modal_cmd=("$candidate")
      break
    fi
  done

  if [ "${#modal_cmd[@]}" -eq 0 ]; then
    shopt -s nullglob
    user_modal_bins=("$HOME"/Library/Python/*/bin/modal)
    shopt -u nullglob
    for candidate in "${user_modal_bins[@]}"; do
      if [ -x "$candidate" ]; then
        modal_cmd=("$candidate")
        break
      fi
    done
  fi
fi

if [ "${#modal_cmd[@]}" -eq 0 ]; then
  echo "cron_train error: modal CLI not found (checked .venv, MODAL_BIN, PATH, and common user install locations)" | tee -a "$LOG_FILE" | tee "$tmp_log"
  echo "cron_train error: PATH=$PATH" | tee -a "$LOG_FILE" | tee -a "$tmp_log"
  echo "cron_train failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

echo "cron_train info: using modal command: ${modal_cmd[*]}" >> "$LOG_FILE"

: > "$tmp_log"

if ! "${modal_cmd[@]}" run "$PROJECT_ROOT/modal_app.py::train_now" \
  2>&1 | tee -a "$LOG_FILE" | tee "$tmp_log"; then
  echo "cron_train failed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/cron_train_error.log"
  if [ -f "$EMAIL_SCRIPT" ]; then
    "$PROJECT_ROOT/.venv/bin/python" "$EMAIL_SCRIPT" --subject "$EMAIL_SUBJECT" --body-file "$tmp_log" || true
  fi
  rm -f "$tmp_log"
  exit 1
fi

rm -f "$tmp_log"
