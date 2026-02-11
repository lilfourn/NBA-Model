"""Fit adaptive per-stat selection thresholds from resolved predictions.

Usage:
    python -m scripts.ml.fit_selection_thresholds --days-back 180
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.selection_policy import fit_selection_policy_from_db  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit adaptive selection thresholds.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--days-back", type=int, default=180)
    parser.add_argument("--coverage-floor", type=float, default=0.40)
    parser.add_argument("--target-hit-rate", type=float, default=0.55)
    parser.add_argument("--min-rows-per-stat", type=int, default=200)
    parser.add_argument("--threshold-start", type=float, default=0.55)
    parser.add_argument("--threshold-end", type=float, default=0.72)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--conformal-penalty", type=float, default=0.02)
    parser.add_argument("--output", default="models/selection_policy.json")
    parser.add_argument(
        "--upload-db",
        action="store_true",
        help="Upload fitted policy json to DB artifact store.",
    )
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    policy = fit_selection_policy_from_db(
        engine,
        days_back=args.days_back,
        min_rows_per_stat=args.min_rows_per_stat,
        coverage_floor=args.coverage_floor,
        target_hit_rate=args.target_hit_rate,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        conformal_ambiguous_penalty=args.conformal_penalty,
    )
    output_path = policy.save(args.output)

    print(f"Selection policy -> {output_path}")
    print(f"  version:          {policy.version}")
    print(f"  source rows:      {policy.source_rows}")
    print(f"  global threshold: {policy.global_threshold:.3f}")
    print(f"  stat thresholds:  {len(policy.per_stat_thresholds)}")
    print(f"  conformal penalty:{policy.conformal_ambiguous_penalty:.3f}")

    per_stat = policy.diagnostics.get("per_stat", {})
    for stat_type, meta in sorted(per_stat.items()):
        if meta.get("status") != "trained":
            continue
        threshold = float(meta.get("threshold", policy.global_threshold))
        n_rows = int(meta.get("n", 0))
        print(f"    {stat_type:25s} n={n_rows:>5d} threshold={threshold:.3f}")

    if args.upload_db:
        try:
            from app.ml.artifact_store import upload_file

            row_id = upload_file(
                engine, model_name="selection_policy", file_path=Path(output_path)
            )
            print(f"  Uploaded to DB -> {row_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: DB upload failed: {exc}")


if __name__ == "__main__":
    main()
