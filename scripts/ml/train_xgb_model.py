import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.xgb.train import train_xgboost  # noqa: E402
from scripts.ml.train_baseline_model import load_env, report_training_data_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model from DB.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model-dir", "--models-dir", default="models")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    try:
        result = train_xgboost(engine, Path(args.model_dir))
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
