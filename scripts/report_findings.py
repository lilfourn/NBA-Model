from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import load_training_data  # noqa: E402
from scripts.train_baseline_model import load_env  # noqa: E402


def _latest_model_runs(engine, *, limit_per_model: int = 1) -> pd.DataFrame:
    query = text(
        """
        with ranked as (
            select
                model_name,
                created_at,
                train_rows,
                metrics,
                params,
                artifact_path,
                row_number() over (partition by model_name order by created_at desc) as rn
            from model_runs
        )
        select
            model_name,
            created_at,
            train_rows,
            metrics,
            params,
            artifact_path
        from ranked
        where rn <= :limit_per_model
        order by created_at desc
        """
    )
    return pd.read_sql(query, engine, params={"limit_per_model": int(limit_per_model)})


def _print_model_runs(df: pd.DataFrame) -> None:
    if df.empty:
        print("No model_runs found.")
        return
    print("Latest model runs (per model_name)")
    print("=" * 80)
    for row in df.sort_values("created_at", ascending=False).itertuples(index=False):
        model_name = getattr(row, "model_name", "")
        created_at = getattr(row, "created_at", "")
        train_rows = getattr(row, "train_rows", "")
        metrics = getattr(row, "metrics", {}) or {}
        params = getattr(row, "params", {}) or {}

        lr = params.get("learning_rate")
        eta = params.get("eta")
        shrink = params.get("shrink_to_uniform") or params.get("shrink")

        extra = []
        if lr is not None:
            extra.append(f"learning_rate={lr}")
        if eta is not None:
            extra.append(f"eta={eta}")
        if shrink is not None:
            extra.append(f"shrink={shrink}")

        extras = f" ({', '.join(extra)})" if extra else ""
        print(f"- {created_at} | {model_name} | train_rows={train_rows}{extras}")
        print(f"  metrics={metrics}")


def _print_training_sanity(df: pd.DataFrame) -> None:
    if df.empty:
        print("Training dataset is empty.")
        return

    print("\nTraining data sanity (baseline dataset)")
    print("=" * 80)

    # Compute labels the same way training does.
    df = df.copy()
    if "actual_value" not in df.columns:
        # load_training_data already includes raw stat cols; training computes actual_value later
        # but for sanity we just re-use app.ml.dataset.compute_actual_value indirectly by re-running the same logic.
        from app.ml.dataset import compute_actual_value  # noqa: WPS433 (local import to keep script fast)

        df["actual_value"] = df.apply(compute_actual_value, axis=1)

    df = df.dropna(subset=["line_score", "actual_value"])
    if df.empty:
        print("No rows with both line_score and actual_value.")
        return

    df["over"] = (df["actual_value"].astype(float) > df["line_score"].astype(float)).astype(int)

    print(f"rows={len(df)}")
    print(f"over_rate={df['over'].mean():.3f}")
    if "snapshot_id" in df.columns and "projection_id" in df.columns:
        uniq = df[["snapshot_id", "projection_id"]].drop_duplicates().shape[0]
        print(f"unique(snapshot_id,projection_id)={uniq}")
    if "projection_id" in df.columns:
        uniq_proj = df["projection_id"].nunique(dropna=True)
        print(f"unique(projection_id)={uniq_proj}")

    # Quick extreme-value checks to catch aggregation bugs.
    thresholds = {
        "points": 80,
        "rebounds": 40,
        "assists": 30,
        "steals": 15,
        "blocks": 15,
        "turnovers": 20,
        "fga": 60,
        "fgm": 40,
        "fg3a": 35,
        "fg3m": 20,
        "fta": 50,
        "ftm": 40,
    }
    bad = []
    for stat, thresh in thresholds.items():
        if "stat_type" not in df.columns:
            break
        mask = df["stat_type"].fillna("").astype(str).str.contains(stat, case=False, regex=False)
        if not mask.any():
            continue
        max_val = float(df.loc[mask, "actual_value"].max())
        count_over = int((df.loc[mask, "actual_value"].astype(float) > float(thresh)).sum())
        if count_over:
            bad.append((stat, thresh, count_over, max_val))

    if bad:
        print("\nWARNING: suspiciously high actual_value rows (possible aggregation bug):")
        for stat, thresh, count_over, max_val in bad:
            print(f"- '{stat}': count>{thresh} = {count_over}, max={max_val}")
    else:
        print("\nNo obvious 'actual_value too high' red flags found vs conservative thresholds.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Report current model accuracy + sanity-check actuals.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--limit-per-model", type=int, default=1)
    ap.add_argument("--skip-training-sanity", action="store_true")
    args = ap.parse_args()

    # Avoid leaking secrets if someone prints env by accident.
    os.environ.pop("DATABASE_URL", None)
    load_env()
    engine = get_engine(args.database_url)

    runs = _latest_model_runs(engine, limit_per_model=int(args.limit_per_model))
    _print_model_runs(runs)

    if not args.skip_training_sanity:
        df = load_training_data(engine)
        _print_training_sanity(df)


if __name__ == "__main__":
    main()
