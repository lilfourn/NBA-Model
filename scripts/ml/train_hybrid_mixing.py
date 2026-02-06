"""Train hybrid ensemble mixing weights (alpha, beta, gamma).

Loads resolved predictions, builds sub-ensemble predictions (Thompson,
Gating, Meta), then grid-searches for optimal mixing via fit_mixing().

Usage:
    python -m scripts.ml.train_hybrid_mixing [--days-back 90] [--output models/hybrid_mixing.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.engine import get_engine  # noqa: E402
from app.ml.artifacts import load_joblib_artifact  # noqa: E402
from app.ml.meta_learner import infer_meta_learner  # noqa: E402
from app.modeling.gating_model import GatingModel, build_context_features  # noqa: E402
from app.modeling.hybrid_ensemble import HybridEnsembleCombiner  # noqa: E402
from app.modeling.online_ensemble import ContextualHedgeEnsembler  # noqa: E402
from app.modeling.thompson_ensemble import ThompsonSamplingEnsembler  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


EXPERTS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]

# Stat types excluded from training (degenerate base rates)
EXCLUDED_STAT_TYPES = {
    "Dunks",
    "Blocked Shots",
    "Blks+Stls",
    "Offensive Rebounds",
    "Personal Fouls",
    "Steals",
}


def _load_resolved(engine, days_back: int) -> pd.DataFrame:
    """Load resolved predictions with expert probabilities."""
    for table in ("vw_resolved_picks_canonical", "projection_predictions"):
        try:
            where = (
                "WHERE 1=1"
                if table == "vw_resolved_picks_canonical"
                else "WHERE outcome IN ('over', 'under') AND over_label IS NOT NULL"
            )
            df = pd.read_sql(
                text(
                    f"""
                    SELECT
                        stat_type, over_label, n_eff,
                        p_forecast_cal, p_nn,
                        coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl,
                        p_lr, p_xgb, p_lgbm,
                        coalesce(details->>'is_live', 'false') as is_live
                    FROM {table}
                    {where}
                      AND coalesce(decision_time, created_at) >= now() - (:days * interval '1 day')
                    ORDER BY coalesce(decision_time, created_at) ASC
                """
                ),
                engine,
                params={"days": int(max(1, days_back))},
            )
            if not df.empty:
                return df
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hybrid ensemble mixing weights."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--days-back", type=int, default=90)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="models/hybrid_mixing.json")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    models_dir = Path(args.models_dir)

    # Load resolved data
    df = _load_resolved(engine, args.days_back)
    if df.empty:
        print("No resolved predictions found.")
        return

    # Filter degenerate stat types
    df = df[~df["stat_type"].isin(EXCLUDED_STAT_TYPES)].copy()
    df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
    df["n_eff"] = pd.to_numeric(df["n_eff"], errors="coerce")
    for col in EXPERTS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["is_live"] = (
        df["is_live"]
        .fillna("false")
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes", "t"})
    )
    df = df.dropna(subset=["over_label", "stat_type"])
    if df.empty:
        print("No valid rows after filtering.")
        return

    print(f"Training hybrid mixing on {len(df)} resolved rows.")

    # Load component models
    ts_path = models_dir / "thompson_weights.json"
    thompson = None
    if ts_path.exists():
        try:
            thompson = ThompsonSamplingEnsembler.load(str(ts_path))
        except Exception as e:
            print(f"Thompson load failed: {e}")

    gating_path = models_dir / "gating_model.joblib"
    gating = None
    if gating_path.exists():
        try:
            gating = GatingModel.load(str(gating_path))
        except Exception as e:
            print(f"Gating load failed: {e}")

    meta_path = None
    meta_candidates = sorted(
        models_dir.glob("meta_learner_*.joblib"), key=lambda p: p.stat().st_mtime
    )
    if meta_candidates:
        meta_path = meta_candidates[-1]

    # Build hybrid combiner
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson,
        gating=gating,
        experts=EXPERTS,
    )

    # Build input lists
    expert_probs_list = []
    ctx_list = []
    ctx_features_list = []
    p_meta_list = []
    labels = []

    for row in df.itertuples(index=False):
        ep = {}
        for col in EXPERTS:
            v = getattr(row, col, None)
            if v is not None and pd.notna(v):
                ep[col] = float(v)
            else:
                ep[col] = None

        # Need at least 2 expert probs
        n_valid = sum(1 for v in ep.values() if v is not None)
        if n_valid < 2:
            continue

        n_eff_val = getattr(row, "n_eff", None)
        if n_eff_val is not None and pd.notna(n_eff_val):
            n_eff_val = float(n_eff_val)
        else:
            n_eff_val = None

        stat_type = str(getattr(row, "stat_type", ""))
        is_live = bool(getattr(row, "is_live", False))
        ctx = (
            stat_type,
            "live" if is_live else "pregame",
            "high" if (n_eff_val or 0) >= 15 else "low",
        )

        # Context features for gating
        avail = {k: v for k, v in ep.items() if v is not None}
        ctx_feat = None
        if avail and gating is not None and gating.is_fitted:
            try:
                ctx_feat = build_context_features(
                    {k: np.array([v]) for k, v in avail.items()},
                    n_eff=np.array([n_eff_val or 0.0]),
                )[0]
            except Exception:  # noqa: BLE001
                ctx_feat = None

        # Meta-learner prediction
        p_meta = None
        if meta_path:
            try:
                p_meta = infer_meta_learner(
                    model_path=str(meta_path), expert_probs=ep, n_eff=n_eff_val
                )
            except Exception:  # noqa: BLE001
                pass

        expert_probs_list.append(ep)
        ctx_list.append(ctx)
        ctx_features_list.append(ctx_feat)
        p_meta_list.append(p_meta if p_meta is not None else float("nan"))
        labels.append(int(getattr(row, "over_label")))

    if len(labels) < 50:
        print(f"Not enough valid rows for hybrid mixing ({len(labels)} < 50).")
        return

    # Convert to numpy
    labels_arr = np.array(labels, dtype=float)
    p_meta_arr = np.array(p_meta_list, dtype=float)

    # Context features: only pass if we have gating
    ctx_features_arr = None
    if gating is not None and gating.is_fitted:
        valid_feats = [f for f in ctx_features_list if f is not None]
        if len(valid_feats) == len(labels):
            ctx_features_arr = np.array(valid_feats)

    print(
        f"Before: alpha={hybrid.alpha:.2f} beta={hybrid.beta:.2f} gamma={hybrid.gamma:.2f}"
    )

    # Fit mixing weights
    hybrid.fit_mixing(
        expert_probs_list=expert_probs_list,
        ctx_list=ctx_list,
        context_features_list=ctx_features_arr,
        p_meta_list=p_meta_arr,
        labels=labels_arr,
    )

    print(
        f"After:  alpha={hybrid.alpha:.2f} beta={hybrid.beta:.2f} gamma={hybrid.gamma:.2f}"
    )

    # Save
    out_path = Path(args.output)
    hybrid.save_mixing(out_path)
    print(f"Saved hybrid mixing weights -> {out_path}")

    # Upload to DB
    try:
        from app.ml.artifact_store import upload_file

        upload_file(engine, model_name="hybrid_mixing", file_path=out_path)
        print("Uploaded hybrid_mixing to DB artifact store.")
    except Exception as e:  # noqa: BLE001
        print(f"DB upload failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
