import argparse
import csv
import sys
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.modeling.db_logs import load_db_game_logs  # noqa: E402
from app.modeling.stat_forecast import ForecastParams, LeaguePriors, StatForecastPredictor  # noqa: E402
from app.modeling.stat_mappings import SPECIAL_STATS, STAT_TYPE_MAP, stat_value  # noqa: E402
from app.modeling.types import Projection  # noqa: E402
from scripts.train_baseline_model import load_env  # noqa: E402


DEFAULT_STAT_TYPES = [
    "Points",
    "Rebounds",
    "Assists",
    "Steals",
    "Blocked Shots",
    "Turnovers",
    "3-PT Made",
    "Pts+Rebs",
    "Pts+Asts",
    "Rebs+Asts",
    "Pts+Rebs+Asts",
    "Blks+Stls",
]


def _projection_for_game(*, player_name: str, game_date, stat_type: str) -> Projection:
    start_time = datetime.combine(game_date, time(12, 0))
    return Projection(
        projection_id=f"backtest::{player_name}::{game_date}::{stat_type}",
        player_id="",
        player_name=player_name,
        stat_type=stat_type,
        line_score=0.0,
        start_time=start_time,
        game_id=None,
        event_type=None,
        projection_type=None,
        trending_count=None,
        is_today=None,
        is_combo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a prequential backtest dataset of (y_true, mu_hat, sigma_hat) for calibration.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--output",
        default="data/calibration/forecast_backtest.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--stat-types",
        nargs="*",
        default=None,
        help="Stat types to include (defaults to common). Use --all-supported for all supported stat_value types.",
    )
    parser.add_argument("--all-supported", action="store_true")
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit total rows written.")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--date-from", default=None, help="Only emit rows for game_date >= YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="Only emit rows for game_date <= YYYY-MM-DD")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    logs = load_db_game_logs(engine)
    logs = [log for log in logs if log.game_date is not None and log.player_name]
    logs.sort(key=lambda entry: entry.game_date)

    date_from = datetime.fromisoformat(args.date_from).date() if args.date_from else None
    date_to = datetime.fromisoformat(args.date_to).date() if args.date_to else None

    if args.all_supported:
        stat_types = sorted(list(STAT_TYPE_MAP.keys()) + list(SPECIAL_STATS.keys()))
    else:
        stat_types = args.stat_types or DEFAULT_STAT_TYPES

    params = ForecastParams()
    priors = LeaguePriors(logs, stat_types=stat_types, minutes_prior=params.minutes_prior)
    predictor = StatForecastPredictor(logs, min_games=args.min_games, league_priors=priors, params=params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stat_type",
                "player_name",
                "game_date",
                "y_true",
                "mu_hat",
                "sigma_hat",
                "n_eff",
            ],
        )
        writer.writeheader()

        for log in logs:
            if date_from and log.game_date and log.game_date < date_from:
                continue
            if date_to and log.game_date and log.game_date > date_to:
                continue
            for stat_type in stat_types:
                y_true = stat_value(stat_type, log.stats)
                if y_true is None:
                    continue
                proj = _projection_for_game(
                    player_name=str(log.player_name),
                    game_date=log.game_date,
                    stat_type=stat_type,
                )
                forecast = predictor.forecast_distribution(proj)
                if forecast is None:
                    continue
                mu_hat, sigma_hat, details = forecast
                if sigma_hat <= 0:
                    continue

                writer.writerow(
                    {
                        "stat_type": stat_type,
                        "player_name": log.player_name,
                        "game_date": log.game_date.isoformat(),
                        "y_true": float(y_true),
                        "mu_hat": float(mu_hat),
                        "sigma_hat": float(sigma_hat),
                        "n_eff": float(details.get("n_eff")) if details else None,
                    }
                )
                written += 1
                if args.max_rows and written >= args.max_rows:
                    print(f"Reached max rows: {written}")
                    print(f"Wrote -> {output_path}")
                    return
                if args.progress_every and written % args.progress_every == 0:
                    print({"written": written, "last_date": log.game_date.isoformat()})

    print(f"Wrote {written} rows -> {output_path}")


if __name__ == "__main__":
    main()
