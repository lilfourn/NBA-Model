from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _parse_iso_date(value: str) -> datetime:
    return datetime.strptime(str(value).strip(), "%Y-%m-%d")


def _latest_nba_fetch_summary(log_path: Path, *, max_age_hours: int) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(max_age_hours)))
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return None
    for line in reversed(lines):
        raw = line.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "run_summary":
            continue
        if str(entry.get("source") or "") != "nba_stats":
            continue
        ts_raw = entry.get("ts")
        try:
            ts = datetime.fromisoformat(str(ts_raw))
        except Exception:  # noqa: BLE001
            ts = None
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts is not None and ts < cutoff:
            return None
        return entry
    return None


def _load_pending_summary(engine, *, date_from: str, date_to: str) -> dict[str, int]:
    frame = pd.read_sql(
        text(
            """
            select
                count(*)::int as pending_rows,
                count(distinct game_id)::int as pending_games
            from projection_predictions
            where actual_value is null
              and game_id is not null
              and (coalesce(decision_time, created_at) at time zone 'America/New_York')::date >= :date_from
              and (coalesce(decision_time, created_at) at time zone 'America/New_York')::date <= :date_to
            """
        ),
        engine,
        params={"date_from": date_from, "date_to": date_to},
    )
    if frame.empty:
        return {"pending_rows": 0, "pending_games": 0}
    row = frame.iloc[0]
    return {
        "pending_rows": int(row.get("pending_rows") or 0),
        "pending_games": int(row.get("pending_games") or 0),
    }


def _load_boxscore_summary(engine, *, date_from: str, date_to: str) -> dict[str, int]:
    frame = pd.read_sql(
        text(
            """
            select
                count(*)::int as stats_rows,
                count(distinct s.game_id)::int as stats_games,
                count(distinct g.id)::int as game_rows
            from nba_games g
            left join nba_player_game_stats s on s.game_id = g.id
            where g.game_date >= :date_from
              and g.game_date <= :date_to
            """
        ),
        engine,
        params={"date_from": date_from, "date_to": date_to},
    )
    if frame.empty:
        return {"stats_rows": 0, "stats_games": 0, "game_rows": 0}
    row = frame.iloc[0]
    return {
        "stats_rows": int(row.get("stats_rows") or 0),
        "stats_games": int(row.get("stats_games") or 0),
        "game_rows": int(row.get("game_rows") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NBA training data freshness before retraining.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--min-pending-games", type=int, default=2)
    parser.add_argument("--min-game-coverage-ratio", type=float, default=0.6)
    parser.add_argument("--min-stats-rows", type=int, default=20)
    parser.add_argument("--max-log-age-hours", type=int, default=18)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when freshness gate fails.",
    )
    args = parser.parse_args()

    # Validate date format early with concrete errors.
    _parse_iso_date(args.date_from)
    _parse_iso_date(args.date_to)

    load_env()
    engine = get_engine(args.database_url)

    pending = _load_pending_summary(engine, date_from=args.date_from, date_to=args.date_to)
    boxscores = _load_boxscore_summary(engine, date_from=args.date_from, date_to=args.date_to)

    pending_games = int(pending["pending_games"])
    stats_games = int(boxscores["stats_games"])
    stats_rows = int(boxscores["stats_rows"])
    coverage_ratio = 1.0 if pending_games <= 0 else float(stats_games) / float(max(1, pending_games))

    log_path = Path(args.log_path or settings.collection_log_path)
    fetch_summary = _latest_nba_fetch_summary(
        log_path,
        max_age_hours=int(args.max_log_age_hours),
    )
    latest_counts = {}
    degraded_empty = False
    if isinstance(fetch_summary, dict):
        counts = fetch_summary.get("counts")
        if isinstance(counts, dict):
            latest_counts = counts
            degraded_empty = bool(int(counts.get("degraded_empty") or 0))

    reasons: list[str] = []
    needs_fresh_data = pending_games >= int(max(1, args.min_pending_games))
    if needs_fresh_data:
        if stats_rows < int(max(1, args.min_stats_rows)):
            reasons.append(
                f"stats_rows={stats_rows} is below min_stats_rows={int(max(1, args.min_stats_rows))}"
            )
        if coverage_ratio < float(args.min_game_coverage_ratio):
            reasons.append(
                "coverage ratio "
                f"{coverage_ratio:.3f} is below min_game_coverage_ratio={float(args.min_game_coverage_ratio):.3f}"
            )
        if degraded_empty:
            reasons.append("latest nba_stats collection run was degraded_empty=1")

    report = {
        "date_from": args.date_from,
        "date_to": args.date_to,
        "pending": pending,
        "boxscores": boxscores,
        "coverage_ratio": round(coverage_ratio, 4),
        "needs_fresh_data": bool(needs_fresh_data),
        "fetch_summary_found": fetch_summary is not None,
        "latest_fetch_counts": latest_counts,
        "degraded_empty": bool(degraded_empty),
        "is_fresh": len(reasons) == 0,
        "reasons": reasons,
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    if args.strict and reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
