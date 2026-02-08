"""Full-pipeline backtest: replay resolved predictions through the production scoring path.

Tests the complete pipeline: ensemble predict -> inversion correction -> shrinkage ->
isotonic calibration -> abstain policy. Reports per-tier and per-stat-type metrics.

Usage:
    python -m scripts.ml.backtest_full_pipeline [--days-back 120] [--output data/reports/pipeline_backtest.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.engine import get_engine  # noqa: E402
from app.ml.context_prior import get_context_prior, load_context_priors  # noqa: E402
from app.ml.inversion_corrections import load_inversion_flags  # noqa: E402
from app.ml.stat_calibrator import StatTypeCalibrator  # noqa: E402
from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler  # noqa: E402
from app.services.scoring import (  # noqa: E402
    EXCLUDED_STAT_TYPES,
    MIN_EDGE,
    PICK_THRESHOLD,
    PRIOR_ONLY_STAT_TYPES,
    _compute_edge,
    shrink_probability,
)
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EXPERTS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]


def _logloss(y: int, p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _load_resolved(engine, days_back: int) -> pd.DataFrame:
    for table in ("vw_resolved_picks_canonical", "projection_predictions"):
        try:
            where = (
                "WHERE 1=1"
                if table == "vw_resolved_picks_canonical"
                else "WHERE outcome IN ('over', 'under') AND over_label IS NOT NULL"
            )
            p_expr = "p_final" if table == "vw_resolved_picks_canonical" else "prob_over"
            df = pd.read_sql(
                text(
                    f"""
                    SELECT
                        stat_type, line_score, over_label, n_eff,
                        p_forecast_cal, p_nn,
                        coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl,
                        p_lr, p_xgb, p_lgbm,
                        {p_expr} as p_final_logged,
                        coalesce(details->>'is_live', 'false') as is_live,
                        coalesce(decision_time, created_at) as event_time
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
    parser = argparse.ArgumentParser(description="Full-pipeline backtest.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--days-back", type=int, default=120)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="data/reports/pipeline_backtest.json")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    models_dir = Path(args.models_dir)

    df = _load_resolved(engine, args.days_back)
    if df.empty:
        print("No resolved predictions found.")
        return

    # Prep columns
    for col in EXPERTS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
    df["n_eff"] = pd.to_numeric(df["n_eff"], errors="coerce")
    df["line_score"] = pd.to_numeric(df["line_score"], errors="coerce")
    df["p_final_logged"] = pd.to_numeric(df.get("p_final_logged"), errors="coerce")
    df["is_live"] = (
        df["is_live"]
        .fillna("false")
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes", "t"})
    )
    df = df.dropna(subset=["over_label", "stat_type", "line_score"])

    print(f"Backtesting {len(df)} resolved predictions over {args.days_back} days...")

    # Load models
    ens_path = models_dir / "ensemble_weights.json"
    if ens_path.exists():
        ens = ContextualHedgeEnsembler.load(str(ens_path))
    else:
        ens = ContextualHedgeEnsembler(experts=EXPERTS, eta=0.2, shrink_to_uniform=0.05)

    context_priors = load_context_priors()
    stat_calibrator = StatTypeCalibrator.load()
    inversion_flags = load_inversion_flags()
    if inversion_flags:
        print(f"Inversion corrections active for: {list(inversion_flags.keys())}")

    # Replay each prediction
    results = []
    for row in df.itertuples(index=False):
        stat_type = str(row.stat_type)
        y = int(row.over_label)
        line_score = float(row.line_score)
        n_eff_val = float(row.n_eff) if pd.notna(row.n_eff) else None
        is_live = bool(row.is_live)

        # Exclusions
        if stat_type in EXCLUDED_STAT_TYPES:
            continue
        is_prior_only = stat_type in PRIOR_ONLY_STAT_TYPES

        # Build expert probs
        expert_probs = {}
        for col in EXPERTS:
            v = getattr(row, col, None)
            if v is not None and pd.notna(v):
                expert_probs[col] = float(v)
            else:
                expert_probs[col] = None

        # Apply inversion corrections
        for expert, should_flip in inversion_flags.items():
            if should_flip and expert_probs.get(expert) is not None:
                expert_probs[expert] = 1.0 - expert_probs[expert]

        # Get context prior
        ctx_prior = get_context_prior(
            context_priors, stat_type=stat_type, line_score=line_score
        )

        if is_prior_only:
            p_final = ctx_prior if ctx_prior is not None else 0.5
        else:
            ctx = Context(stat_type=stat_type, is_live=is_live, n_eff=n_eff_val)
            p_raw = float(ens.predict(expert_probs, ctx))
            if not math.isfinite(p_raw):
                continue
            p_final = shrink_probability(
                p_raw, n_eff=n_eff_val, context_prior=ctx_prior
            )
            p_final = stat_calibrator.transform(p_final, stat_type)

        pick = 1 if p_final >= 0.5 else 0
        correct = int(pick == y)
        p_pick = max(p_final, 1.0 - p_final)

        edge = _compute_edge(p_final, expert_probs, None, n_eff=n_eff_val)
        is_actionable = p_pick >= PICK_THRESHOLD
        is_placed = is_actionable and edge >= MIN_EDGE and not is_prior_only

        # Also compute logged p_final accuracy for comparison
        p_logged = getattr(row, "p_final_logged", None)
        logged_correct = None
        if p_logged is not None and pd.notna(p_logged):
            p_logged = float(p_logged)
            logged_pick = 1 if p_logged >= 0.5 else 0
            logged_correct = int(logged_pick == y)

        results.append(
            {
                "stat_type": stat_type,
                "y": y,
                "p_final": p_final,
                "correct": correct,
                "logloss": _logloss(y, p_final),
                "is_actionable": is_actionable,
                "is_placed": is_placed,
                "edge": edge,
                "logged_correct": logged_correct,
            }
        )

    if not results:
        print("No predictions after filtering.")
        return

    rdf = pd.DataFrame(results)
    n_total = len(rdf)

    # Overall metrics
    overall = {
        "n": n_total,
        "accuracy": round(float(rdf["correct"].mean()), 4),
        "logloss": round(float(rdf["logloss"].mean()), 4),
    }

    # Logged comparison
    logged_valid = rdf.dropna(subset=["logged_correct"])
    if not logged_valid.empty:
        overall["logged_accuracy"] = round(
            float(logged_valid["logged_correct"].mean()), 4
        )
        overall["accuracy_delta"] = round(
            overall["accuracy"] - overall["logged_accuracy"], 4
        )

    # Tier metrics
    actionable = rdf[rdf["is_actionable"]]
    placed = rdf[rdf["is_placed"]]
    tiers = {
        "scored": {
            "n": n_total,
            "accuracy": overall["accuracy"],
            "logloss": overall["logloss"],
        },
        "actionable": {
            "n": len(actionable),
            "accuracy": round(float(actionable["correct"].mean()), 4)
            if len(actionable) > 0
            else None,
            "coverage": round(len(actionable) / n_total, 4) if n_total > 0 else 0,
        },
        "placed": {
            "n": len(placed),
            "accuracy": round(float(placed["correct"].mean()), 4)
            if len(placed) > 0
            else None,
            "coverage": round(len(placed) / n_total, 4) if n_total > 0 else 0,
        },
    }

    # Per-stat-type
    per_stat = {}
    for st, grp in rdf.groupby("stat_type"):
        st = str(st)
        per_stat[st] = {
            "n": len(grp),
            "accuracy": round(float(grp["correct"].mean()), 4),
            "logloss": round(float(grp["logloss"].mean()), 4),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_back": args.days_back,
        "overall": overall,
        "tiers": tiers,
        "per_stat_type": per_stat,
        "inversion_corrections": list(inversion_flags.keys()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nPipeline backtest -> {out}")
    print(f"\n{'='*60}")
    print(
        f"  Overall:    n={overall['n']:>5d}  acc={overall['accuracy']:.1%}  ll={overall['logloss']:.4f}"
    )
    if "logged_accuracy" in overall:
        print(
            f"  Logged:     n={len(logged_valid):>5d}  acc={overall['logged_accuracy']:.1%}  delta={overall['accuracy_delta']:+.1%}"
        )
    print(
        f"  Actionable: n={tiers['actionable']['n']:>5d}  acc={tiers['actionable']['accuracy']:.1%}  coverage={tiers['actionable']['coverage']:.1%}"
    )
    print(
        f"  Placed:     n={tiers['placed']['n']:>5d}  acc={tiers['placed']['accuracy']:.1%}  coverage={tiers['placed']['coverage']:.1%}"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
