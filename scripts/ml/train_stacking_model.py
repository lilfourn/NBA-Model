"""Train OOF stacking meta-learner and save as joblib artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.ml.train_baseline_model import load_env  # noqa: E402


OOF_TO_EXPERT = {
    "oof_lr": "p_lr",
    "oof_xgb": "p_xgb",
    "oof_lgbm": "p_lgbm",
    "oof_nn": "p_nn",
    "oof_forecast": "p_forecast_cal",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train stacking meta-learner from OOF predictions."
    )
    ap.add_argument("--oof-path", required=True, help="Path to OOF predictions CSV")
    ap.add_argument(
        "--models-dir", required=True, help="Directory to save model artifact"
    )
    ap.add_argument("--upload-db", action="store_true", help="Upload artifact to DB")
    args = ap.parse_args()

    load_env()
    oof_path = Path(args.oof_path)
    if not oof_path.exists():
        raise SystemExit(f"OOF file not found: {oof_path}")

    df = pd.read_csv(oof_path)
    if len(df) < 100:
        raise SystemExit(f"Not enough OOF data ({len(df)} rows).")

    rename = {k: v for k, v in OOF_TO_EXPERT.items() if k in df.columns}
    df = df.rename(columns=rename)

    # p_tabdl has no OOF proxy — set neutral so stacking learns no weight for it
    if "p_tabdl" not in df.columns:
        df["p_tabdl"] = 0.5

    from app.ml.stacking import train_stacking_meta

    result = train_stacking_meta(df)
    print(f"Stacking meta-learner trained: {result.metrics}")
    print(f"Weights: {result.weights}")

    from datetime import datetime, timezone
    import joblib

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"stacking_meta_{ts}.joblib"
    joblib.dump(
        {"model": result.model, "weights": result.weights, "metrics": result.metrics},
        artifact_path,
    )
    print(f"Saved -> {artifact_path}")

    if args.upload_db:
        from app.db.engine import get_engine
        from app.ml.artifact_store import upload_file

        engine = get_engine()
        upload_file(engine, model_name="stacking_meta", file_path=artifact_path)
        print("Uploaded to DB")


if __name__ == "__main__":
    main()
