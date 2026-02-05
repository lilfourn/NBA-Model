import argparse
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.logging import log_run_summary, set_log_path  # noqa: E402
from app.clients.nba_stats import fetch_league_game_log  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA league game logs and load into DB.")
    parser.add_argument("--season", default=None, help="Season like 2025-26")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    import time as _time
    _start = _time.monotonic()

    set_log_path("logs/collection.jsonl")

    season = args.season or os.getenv("NBA_SEASON") or current_season()

    rows = fetch_league_game_log(
        season=season,
        season_type=args.season_type,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    engine = get_engine(args.database_url)
    counts = load_league_game_logs(rows, engine=engine)

    _elapsed = _time.monotonic() - _start
    log_run_summary("nba_stats", duration_seconds=_elapsed, counts={"rows_fetched": len(rows), **counts})

    print({"season": season, "rows": len(rows), **counts})


if __name__ == "__main__":
    main()
