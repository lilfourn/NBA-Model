import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.db.nba_loader import load_league_game_logs  # noqa: E402
from app.clients.nba_stats import fetch_league_game_log  # noqa: E402
from scripts.train_baseline_model import load_env  # noqa: E402


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = cursor.replace(day=28) + timedelta(days=4)
        month_end = next_month.replace(day=1) - timedelta(days=1)
        chunk_start = max(start, cursor)
        chunk_end = min(end, month_end)
        ranges.append((chunk_start, chunk_end))
        cursor = month_end + timedelta(days=1)
    return ranges


def _season_dates(season: str) -> tuple[date, date]:
    start_year, end_year = season.split("-")
    start = date(int(start_year), 10, 1)
    end = date(int("20" + end_year), 6, 30)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NBA league game logs for full seasons in monthly chunks.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Season strings like 2023-24 2024-25",
    )
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--date-from", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--max-months", type=int, default=None, help="Limit number of month chunks")
    parser.add_argument(
        "--checkpoint",
        default="data/nba_stats_season_checkpoint.json",
        help="Path to resume checkpoint JSON.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start from the beginning.",
    )
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    checkpoint_path = Path(args.checkpoint)
    if not args.no_resume and checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checkpoint = {}
    else:
        checkpoint = {}

    for season in args.seasons:
        start, end = _season_dates(season)
        if args.date_from:
            start = date.fromisoformat(args.date_from)
        if args.date_to:
            end = date.fromisoformat(args.date_to)
        month_ranges = _month_ranges(start, end)
        last_end_raw = checkpoint.get(season, {}).get("last_end")
        last_end = date.fromisoformat(last_end_raw) if last_end_raw else None
        if last_end:
            month_ranges = [
                (chunk_start, chunk_end)
                for chunk_start, chunk_end in month_ranges
                if chunk_end > last_end
            ]
        if args.max_months:
            month_ranges = month_ranges[: args.max_months]
        for chunk_start, chunk_end in month_ranges:
            rows = fetch_league_game_log(
                season=season,
                season_type=args.season_type,
                date_from=chunk_start.isoformat(),
                date_to=chunk_end.isoformat(),
            )
            print(
                {
                    "season": season,
                    "date_from": chunk_start.isoformat(),
                    "date_to": chunk_end.isoformat(),
                    "rows": len(rows),
                }
            )
            if args.skip_load:
                continue
            load_league_game_logs(rows, engine=engine)
            checkpoint.setdefault(season, {})
            checkpoint[season]["last_end"] = chunk_end.isoformat()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
