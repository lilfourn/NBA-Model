"""Train LightGBM model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.lgbm.train import train_lightgbm  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LightGBM model.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    result = train_lightgbm(engine, Path(args.models_dir))
    print(f"LightGBM trained: {result.model_path}")
    print(f"  rows={result.rows} metrics={result.metrics}")


if __name__ == "__main__":
    main()
