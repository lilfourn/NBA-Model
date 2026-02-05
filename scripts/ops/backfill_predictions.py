"""One-time backfill: read prediction_log.csv and insert missing rows into projection_predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.db.prediction_logs import append_prediction_rows  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    load_env()
    engine = get_engine()

    csv_path = ROOT / "data" / "monitoring" / "prediction_log.csv"
    if not csv_path.exists():
        print(f"No CSV at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"CSV rows: {len(df)}")

    # Get existing projection_ids from DB to avoid duplicates
    existing = pd.read_sql(
        "select projection_id, snapshot_id from projection_predictions",
        engine,
    )
    existing_keys = set(
        zip(existing["projection_id"].astype(str), existing["snapshot_id"].astype(str))
    ) if not existing.empty else set()
    print(f"Existing DB rows: {len(existing_keys)}")

    rows: list[dict] = []
    for _, r in df.iterrows():
        key = (str(r.get("projection_id", "")), str(r.get("snapshot_id", "")))
        if key in existing_keys:
            continue
        row = r.to_dict()
        # Map CSV columns to what append_prediction_rows expects
        if "p_final" in row and "prob_over" not in row:
            row["prob_over"] = row["p_final"]
        if "pick" not in row or pd.isna(row.get("pick")):
            p = row.get("prob_over") or row.get("p_final")
            if p is not None and not pd.isna(p):
                row["pick"] = "OVER" if float(p) >= 0.5 else "UNDER"
        rows.append(row)

    print(f"New rows to insert: {len(rows)}")
    if not rows:
        print("Nothing to backfill.")
        return

    inserted = append_prediction_rows(engine, rows)
    print(f"Inserted {inserted} rows into projection_predictions")


if __name__ == "__main__":
    main()
