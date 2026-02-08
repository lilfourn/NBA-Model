"""One-time script to seed current model artifacts into the DB.

Usage:
    python -m scripts.ops.upload_models_to_db [--models-dir models]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload latest model artifacts to DB.")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    load_env()
    from app.db.engine import get_engine  # noqa: E402
    from app.ml.artifact_store import upload_file  # noqa: E402
    from app.ml.artifacts import latest_compatible_joblib_path  # noqa: E402
    from app.ml.nn.infer import latest_compatible_checkpoint  # noqa: E402

    engine = get_engine()
    models_dir = Path(args.models_dir)

    uploads = [
        (
            "baseline_logreg",
            latest_compatible_joblib_path(models_dir, "baseline_logreg_*.joblib"),
        ),
        ("xgb", latest_compatible_joblib_path(models_dir, "xgb_*.joblib")),
        ("lgbm", latest_compatible_joblib_path(models_dir, "lgbm_*.joblib")),
        (
            "nn_gru_attention",
            latest_compatible_checkpoint(models_dir, "nn_gru_attention_*.pt"),
        ),
        (
            "meta_learner",
            latest_compatible_joblib_path(models_dir, "meta_learner_*.joblib"),
        ),
    ]

    # JSON/joblib ensemble + calibration files (not pattern-matched)
    for name, filename in [
        ("ensemble_weights", "ensemble_weights.json"),
        ("thompson_weights", "thompson_weights.json"),
        ("gating_model", "gating_model.joblib"),
        ("hybrid_mixing", "hybrid_mixing.json"),
        ("context_priors", "context_priors.json"),
        ("stat_calibrator", "stat_calibrator.joblib"),
    ]:
        p = models_dir / filename
        if p.exists():
            uploads.append((name, p))

    for model_name, path in uploads:
        if path is None or not Path(path).exists():
            print(f"  SKIP {model_name}: no file found")
            continue
        row_id = upload_file(engine, model_name=model_name, file_path=path)
        size_kb = Path(path).stat().st_size / 1024
        print(f"  OK   {model_name}: {Path(path).name} ({size_kb:.0f} KB) -> {row_id}")

    print("Done.")


if __name__ == "__main__":
    main()
