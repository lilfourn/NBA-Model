"""Train the stacked meta-learner from OOF predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.meta_learner import train_meta_learner  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train stacked meta-learner.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--oof-path", default="data/oof_predictions.csv")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    result = train_meta_learner(
        oof_path=args.oof_path,
        model_dir=Path(args.models_dir),
        engine=engine,
    )
    print(f"Meta-learner trained: {result.model_path}")
    print(f"  rows={result.rows} metrics={result.metrics}")
    print(f"  coefficients={result.coefficients}")


if __name__ == "__main__":
    main()
