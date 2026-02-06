from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.tabdl.train import train_tabdl  # noqa: E402
from scripts.ml.train_baseline_model import load_env, report_training_data_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep tabular model from DB.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model-dir", "--models-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pos-weight", action="store_true")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    try:
        result = train_tabdl(
            engine,
            Path(args.model_dir),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            use_pos_weight=not bool(args.no_pos_weight),
            seed=int(args.seed),
        )
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
