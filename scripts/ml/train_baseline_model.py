import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import load_training_data  # noqa: E402
from app.ml.train import train_baseline  # noqa: E402


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def report_training_data_state(engine) -> None:
    print("Training data preflight")
    print("=" * 80)
    tables = [
        "snapshots",
        "projection_features",
        "players",
        "games",
        "nba_players",
        "nba_games",
        "nba_player_game_stats",
    ]
    with engine.connect() as conn:
        for table in tables:
            try:
                count = conn.execute(text(f"select count(*) from {table}")).scalar()
            except Exception as exc:  # noqa: BLE001
                count = f"ERR: {exc.__class__.__name__}"
            print(f"{table:24s} {count}")

        try:
            range_row = conn.execute(
                text(
                    "select min((start_time at time zone 'America/New_York')::date), "
                    "max((start_time at time zone 'America/New_York')::date) "
                    "from games"
                )
            ).first()
            if range_row:
                print(f"games.game_date range    {range_row[0]} -> {range_row[1]}")
        except Exception:
            pass

        try:
            range_row = conn.execute(
                text("select min(game_date), max(game_date) from nba_games")
            ).first()
            if range_row:
                print(f"nba_games.game_date      {range_row[0]} -> {range_row[1]}")
        except Exception:
            pass

        try:
            match_row = conn.execute(
                text(
                    """
                    select
                        min(ng.game_date),
                        max(ng.game_date),
                        count(*)
                    from games g
                    left join teams ht on ht.id = g.home_team_id
                    left join teams at on at.id = g.away_team_id
                    join nba_games ng
                        on ng.game_date = (g.start_time at time zone 'America/New_York')::date
                        and ng.home_team_abbreviation = ht.abbreviation
                        and ng.away_team_abbreviation = at.abbreviation
                    """
                )
            ).first()
            if match_row:
                print(
                    "matched games range      "
                    f"{match_row[0]} -> {match_row[1]} (count {match_row[2]})"
                )
        except Exception:
            pass

        try:
            mapped_players = conn.execute(
                text(
                    """
                    select count(*)
                    from players pl
                    join nba_players np on np.name_key = pl.name_key
                    """
                )
            ).scalar()
            print(f"mapped players            {mapped_players}")
        except Exception:
            pass

        try:
            feature_games = conn.execute(
                text(
                    """
                    select count(*)
                    from projection_features pf
                    join games g on g.id = pf.game_id
                    left join teams ht on ht.id = g.home_team_id
                    left join teams at on at.id = g.away_team_id
                    join nba_games ng
                        on ng.game_date = (g.start_time at time zone 'America/New_York')::date
                        and ng.home_team_abbreviation = ht.abbreviation
                        and ng.away_team_abbreviation = at.abbreviation
                    """
                )
            ).scalar()
            print(f"projection_features games {feature_games}")
        except Exception:
            pass

        try:
            df = load_training_data(engine)
            print(f"training_rows             {len(df)}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model from DB.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model-dir", "--models-dir", default="models")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    try:
        result = train_baseline(engine, Path(args.model_dir))
    except RuntimeError as exc:
        message = str(exc)
        if (
            "No training data available" in message
            or "Not enough training data" in message
            or "Not enough class variety" in message
        ):
            print(f"{message} Skipping training.")
            report_training_data_state(engine)
            return
        raise
    print({"rows": result.rows, "metrics": result.metrics, "model_path": result.model_path})


if __name__ == "__main__":
    main()
