from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.db.prediction_logs import resolve_prediction_outcomes  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve logged projection predictions to actual outcomes from NBA box scores."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--days-back",
        type=int,
        default=21,
        help="Only consider unresolved predictions with decision_time in the last N days.",
    )
    parser.add_argument(
        "--decision-lag-hours",
        type=int,
        default=3,
        help="Only resolve predictions older than this lag window (hours).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum unresolved rows to process per run.",
    )
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    result = resolve_prediction_outcomes(
        engine,
        days_back=int(args.days_back),
        decision_lag_hours=int(args.decision_lag_hours),
        limit=int(args.limit),
    )
    print(result)


if __name__ == "__main__":
    main()
