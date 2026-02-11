"""Fit per-stat-type isotonic calibrators from resolved predictions.

Usage:
    python -m scripts.ml.fit_stat_calibrator [--days-back 45] [--output models/stat_calibrator.joblib]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.stat_calibrator import StatTypeCalibrator  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-stat isotonic calibrators.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--days-back", type=int, default=45)
    parser.add_argument("--output", default="models/stat_calibrator.joblib")
    parser.add_argument(
        "--upload-db",
        action="store_true",
        help="Upload fitted calibrator to DB artifact store.",
    )
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    cal = StatTypeCalibrator.fit_from_db(engine, days_back=args.days_back)
    path = cal.save(args.output)

    meta = cal.meta
    print(f"Stat calibrator -> {path}")
    print(f"  Total rows:    {meta.get('total_rows', 0)}")
    print(f"  Stat types calibrated: {meta.get('n_stat_types_calibrated', 0)}")
    print(f"  Global calibrated:     {meta.get('global_calibrated', False)}")
    print(f"  Input source:          {meta.get('input_source', 'unknown')}")
    degenerate_stats = list(meta.get("degenerate_stats") or [])
    print(f"  Degenerate stat types: {len(degenerate_stats)}")
    for st, info in sorted(meta.get("stat_types", {}).items()):
        status = "OK" if info.get("calibrated") else info.get("reason", "skip")
        print(f"    {st:25s} n={info.get('n', 0):>5d} {status}")

    if args.upload_db:
        try:
            from app.ml.artifact_store import upload_file

            row_id = upload_file(
                engine, model_name="stat_calibrator", file_path=Path(path)
            )
            print(f"  Uploaded to DB -> {row_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: DB upload failed: {exc}")


if __name__ == "__main__":
    main()
