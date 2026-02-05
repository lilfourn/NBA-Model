import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.ml.nn.train import train_nn  # noqa: E402
from scripts.ml.train_baseline_model import load_env, report_training_data_state  # noqa: E402


def report_nn_data_state(engine) -> None:
    print("NN data preflight (non-combo)")
    print("=" * 80)
    queries = {
        "non_combo_projections": """
            select count(*)
            from projection_features pf
            join players pl on pl.id = pf.player_id
            where pl.combo is null or pl.combo = false
        """,
        "non_combo_with_game_match": """
            select count(*)
            from projection_features pf
            join players pl on pl.id = pf.player_id
            join games g on g.id = pf.game_id
            left join teams ht on ht.id = g.home_team_id
            left join teams at on at.id = g.away_team_id
            join nba_games ng
              on ng.game_date = (g.start_time at time zone 'America/New_York')::date
             and ng.home_team_abbreviation = ht.abbreviation
             and ng.away_team_abbreviation = at.abbreviation
            where pl.combo is null or pl.combo = false
        """,
        "non_combo_with_stats": """
            select count(*)
            from projection_features pf
            join players pl on pl.id = pf.player_id
            join games g on g.id = pf.game_id
            left join teams ht on ht.id = g.home_team_id
            left join teams at on at.id = g.away_team_id
            join nba_games ng
              on ng.game_date = (g.start_time at time zone 'America/New_York')::date
             and ng.home_team_abbreviation = ht.abbreviation
             and ng.away_team_abbreviation = at.abbreviation
            join nba_player_game_stats s on s.game_id = ng.id
            where pl.combo is null or pl.combo = false
        """,
    }
    with engine.connect() as conn:
        for label, sql in queries.items():
            try:
                count = conn.execute(text(sql)).scalar()
            except Exception as exc:  # noqa: BLE001
                count = f"ERR: {exc.__class__.__name__}"
            print(f"{label:28s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU+attention NN model.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model-dir", "--models-dir", default="models")
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    try:
        result = train_nn(
            engine=engine,
            model_dir=Path(args.model_dir),
            history_len=args.history_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
    except RuntimeError as exc:
        if "No training data available" in str(exc):
            print("No training data available yet. Skipping training.")
            report_training_data_state(engine)
            report_nn_data_state(engine)
            return
        raise
    print({"rows": result.rows, "metrics": result.metrics, "model_path": result.model_path})


if __name__ == "__main__":
    main()
