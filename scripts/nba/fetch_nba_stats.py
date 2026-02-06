import argparse
import os
import sys
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.logging import log_run_summary, set_log_path  # noqa: E402
from app.clients.nba_stats import fetch_league_game_log  # noqa: E402
from app.clients.nba_stats import reset_nba_stats_client  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.db.nba_loader import load_league_game_logs  # noqa: E402


def current_season(today: date | None = None) -> str:
    now = today or date.today()
    if now.month >= 10:
        start_year = now.year
    else:
        start_year = now.year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def _parse_iso_date(value: str) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _iter_days_inclusive(start_date: date, end_date: date) -> list[str]:
    out: list[str] = []
    cursor = start_date
    while cursor <= end_date:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        game_id = str(row.get("GAME_ID") or row.get("game_id") or "").strip()
        player_id = str(row.get("PLAYER_ID") or row.get("player_id") or "").strip()
        if not game_id or not player_id:
            out.append(row)
            continue
        key = (game_id, player_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return True
    message = str(exc).lower()
    return "timed out" in message or "operation timed out" in message or "curl: (28)" in message


def _fetch_chunked_by_day(
    *,
    season: str,
    season_type: str,
    date_from: str,
    date_to: str,
    fast_fail_consecutive_timeouts: int = 2,
) -> list[dict]:
    start = _parse_iso_date(date_from)
    end = _parse_iso_date(date_to)
    if start is None or end is None:
        raise RuntimeError(f"Invalid date range for day-chunk fallback: {date_from}..{date_to}")
    if start > end:
        raise RuntimeError(f"Invalid date range for day-chunk fallback: {date_from}..{date_to}")

    all_rows: list[dict] = []
    failed_days: list[str] = []
    consecutive_timeouts = 0
    for day in _iter_days_inclusive(start, end):
        reset_nba_stats_client()
        try:
            day_rows = fetch_league_game_log(
                season=season,
                season_type=season_type,
                date_from=day,
                date_to=day,
            )
        except Exception as exc:  # noqa: BLE001
            failed_days.append(day)
            print(f"[nba_stats] day {day}: failed ({exc.__class__.__name__})")
            if _is_timeout_error(exc):
                consecutive_timeouts += 1
            else:
                consecutive_timeouts = 0
            if (
                consecutive_timeouts >= int(max(1, fast_fail_consecutive_timeouts))
                and not all_rows
            ):
                print(
                    "[nba_stats] repeated timeout failures detected; "
                    "stopping day-chunk retries early for faster fallback handling."
                )
                break
            continue
        consecutive_timeouts = 0
        all_rows.extend(day_rows)

    deduped = _dedupe_rows(all_rows)
    if not deduped:
        raise RuntimeError(
            f"NBA stats day-chunk fallback failed for all days in range {date_from}..{date_to}: {failed_days}"
        )
    if failed_days:
        print(
            f"[nba_stats] day-chunk fallback partial success: fetched {len(deduped)} rows "
            f"with failures on {len(failed_days)} day(s)."
        )
    return deduped


def _existing_stats_rows(engine, *, date_from: str, date_to: str) -> int:
    start = _parse_iso_date(date_from)
    end = _parse_iso_date(date_to)
    if start is None or end is None or start > end:
        return 0
    from sqlalchemy import text

    with engine.connect() as conn:
        count = conn.execute(
            text(
                """
                select count(*)
                from nba_player_game_stats s
                join nba_games g on g.id = s.game_id
                where g.game_date >= :date_from
                  and g.game_date <= :date_to
                """
            ),
            {"date_from": start, "date_to": end},
        ).scalar()
    return int(count or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA league game logs and load into DB.")
    parser.add_argument("--season", default=None, help="Season like 2025-26")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--allow-empty-on-failure",
        action="store_true",
        help=(
            "If all NBA upstream requests fail and no existing DB rows are found, continue with zero fetched rows "
            "instead of failing."
        ),
    )
    parser.add_argument(
        "--fast-fail-consecutive-timeouts",
        type=int,
        default=2,
        help="Stop day-level fallback early after N consecutive timeout failures.",
    )
    args = parser.parse_args()

    import time as _time
    _start = _time.monotonic()

    set_log_path(settings.collection_log_path)

    season = args.season or os.getenv("NBA_SEASON") or current_season()
    engine = None
    reused_existing = False
    degraded_empty = False
    allow_empty_on_failure = bool(args.allow_empty_on_failure) or str(
        os.getenv("NBA_STATS_ALLOW_EMPTY_ON_FAILURE", "")
    ).strip().lower() in {"1", "true", "yes", "on"}

    try:
        rows = fetch_league_game_log(
            season=season,
            season_type=args.season_type,
            date_from=args.date_from,
            date_to=args.date_to,
        )
    except Exception as exc:  # noqa: BLE001
        if args.date_from and args.date_to:
            print(
                "[nba_stats] bulk fetch failed "
                f"({exc.__class__.__name__}); retrying with per-day chunk fallback."
            )
            try:
                rows = _fetch_chunked_by_day(
                    season=season,
                    season_type=args.season_type,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    fast_fail_consecutive_timeouts=args.fast_fail_consecutive_timeouts,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                engine = get_engine(args.database_url)
                existing = _existing_stats_rows(engine, date_from=args.date_from, date_to=args.date_to)
                if existing > 0:
                    reused_existing = True
                    rows = []
                    print(
                        "[nba_stats] all day-chunk requests failed; "
                        f"reusing existing {existing} DB boxscore row(s) for {args.date_from}..{args.date_to}."
                    )
                elif allow_empty_on_failure:
                    rows = []
                    degraded_empty = True
                    print(
                        "[nba_stats] all day-chunk requests failed and no existing rows found; "
                        "continuing with empty fetch result due allow-empty-on-failure."
                    )
                else:
                    raise RuntimeError(
                        "NBA stats fetch failed and no existing DB data is available for "
                        f"{args.date_from}..{args.date_to}"
                    ) from fallback_exc
        else:
            raise
    rows = _dedupe_rows(rows)

    if engine is None:
        engine = get_engine(args.database_url)
    counts = load_league_game_logs(rows, engine=engine)

    _elapsed = _time.monotonic() - _start
    log_run_summary(
        "nba_stats",
        duration_seconds=_elapsed,
        counts={
            "rows_fetched": len(rows),
            "reused_existing": int(reused_existing),
            "degraded_empty": int(degraded_empty),
            **counts,
        },
    )

    print(
        {
            "season": season,
            "rows": len(rows),
            "reused_existing": reused_existing,
            "degraded_empty": degraded_empty,
            **counts,
        }
    )


if __name__ == "__main__":
    main()
