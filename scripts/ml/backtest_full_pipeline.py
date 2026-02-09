"""Full-pipeline backtest: replay resolved predictions through multiple ensemble strategies.

Compares strategies side-by-side on the same resolved data with per-tier,
per-stat-type, and multi-window metrics.

Usage:
    python -m scripts.ml.backtest_full_pipeline [--days-back 120] [--strategies logit_avg,top_k]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.engine import get_engine  # noqa: E402
from app.ml.context_prior import get_context_prior, load_context_priors  # noqa: E402
from app.ml.ensemble_strategies import (
    build_strategies,
    multi_window_metrics,
)  # noqa: E402
from app.ml.stat_calibrator import StatTypeCalibrator  # noqa: E402
from app.services.scoring import (  # noqa: E402
    ENSEMBLE_EXPERTS,
    EXCLUDED_STAT_TYPES,
    MIN_EDGE,
    PICK_THRESHOLD,
    PRIOR_ONLY_STAT_TYPES,
    _compute_edge,
    shrink_probability,
)
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EXPERTS = list(ENSEMBLE_EXPERTS)


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
            p_expr = (
                "p_final" if table == "vw_resolved_picks_canonical" else "prob_over"
            )
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


def _prep_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    return df.dropna(subset=["over_label", "stat_type", "line_score"])


def _load_stacking_model(models_dir: Path):
    try:
        from app.ml.artifacts import latest_compatible_joblib_path, load_joblib_artifact

        path = latest_compatible_joblib_path(models_dir, "stacking_meta_*.joblib")
        if path and path.exists():
            payload = load_joblib_artifact(str(path))
            return payload.get("model")
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_comparison(
    strategy_results: dict[str, list[dict]],
) -> dict[str, dict]:
    comparison: dict[str, dict] = {}
    for name, results in strategy_results.items():
        if not results:
            comparison[name] = {"n": 0}
            continue

        probs = np.array([r["p_final"] for r in results])
        labels = np.array([r["y"] for r in results], dtype=float)
        windows = multi_window_metrics(probs, labels)

        rdf = pd.DataFrame(results)
        actionable = rdf[rdf["is_actionable"]]
        placed = rdf[rdf["is_placed"]]

        tiers = {
            "scored": {
                "n": len(rdf),
                "accuracy": round(float(rdf["correct"].mean()), 4),
                "logloss": round(float(rdf["logloss"].mean()), 4),
            },
            "actionable": {
                "n": len(actionable),
                "accuracy": (
                    round(float(actionable["correct"].mean()), 4)
                    if len(actionable) > 0
                    else None
                ),
                "coverage": round(len(actionable) / len(rdf), 4) if len(rdf) else 0,
            },
            "placed": {
                "n": len(placed),
                "accuracy": (
                    round(float(placed["correct"].mean()), 4)
                    if len(placed) > 0
                    else None
                ),
                "coverage": round(len(placed) / len(rdf), 4) if len(rdf) else 0,
            },
        }

        per_stat: dict[str, dict] = {}
        for st, grp in rdf.groupby("stat_type"):
            per_stat[str(st)] = {
                "n": len(grp),
                "accuracy": round(float(grp["correct"].mean()), 4),
                "logloss": round(float(grp["logloss"].mean()), 4),
            }

        comparison[name] = {
            "windows": windows,
            "tiers": tiers,
            "per_stat_type": per_stat,
        }
    return comparison


def _best_strategy_per_window(comparison: dict[str, dict]) -> dict[str, str]:
    best: dict[str, str] = {}
    window_keys: set[str] = set()
    for data in comparison.values():
        window_keys.update(data.get("windows", {}).keys())

    for wk in sorted(window_keys):
        best_name, best_acc = "", -1.0
        for name, data in comparison.items():
            acc = data.get("windows", {}).get(wk, {}).get("rolling_accuracy", -1)
            if acc > best_acc:
                best_acc = acc
                best_name = name
        if best_name:
            best[wk] = best_name
    return best


def _print_comparison(comparison: dict[str, dict]) -> None:
    window_keys = ["last_50", "last_100", "last_200", "all_time"]
    header = f"{'Strategy':<20}"
    for wk in window_keys:
        header += f" {wk:>12}"
    print(f"\n{'='*72}")
    print(header)
    print("-" * 72)

    for name, data in sorted(comparison.items()):
        windows = data.get("windows", {})
        row = f"{name:<20}"
        for wk in window_keys:
            m = windows.get(wk, {})
            acc = m.get("rolling_accuracy")
            row += f" {acc:>11.1%}" if acc is not None else f" {'N/A':>11}"
        print(row)
    print("=" * 72)

    print("\nTier breakdown:")
    for name, data in sorted(comparison.items()):
        tiers = data.get("tiers", {})
        scored = tiers.get("scored", {})
        placed = tiers.get("placed", {})
        s_acc = scored.get("accuracy")
        p_acc = placed.get("accuracy")
        p_n = placed.get("n", 0)
        print(
            f"  {name:<20} scored={s_acc:.1%}"
            + (f"  placed={p_acc:.1%} (n={p_n})" if p_acc is not None else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-strategy pipeline backtest.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--days-back", type=int, default=120)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy names (default: all)",
    )
    parser.add_argument("--output", default="data/reports/pipeline_backtest.json")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    models_dir = Path(args.models_dir)

    df = _load_resolved(engine, args.days_back)
    if df.empty:
        print("No resolved predictions found.")
        return
    df = _prep_columns(df)

    print(f"Backtesting {len(df)} resolved predictions over {args.days_back} days...")

    context_priors = load_context_priors()
    stat_calibrator = StatTypeCalibrator.load()

    stacking_model = _load_stacking_model(models_dir)
    strategies = build_strategies(stacking_model=stacking_model, experts=EXPERTS)

    if args.strategies:
        selected = {s.strip() for s in args.strategies.split(",")}
        strategies = {k: v for k, v in strategies.items() if k in selected}
        if not strategies:
            print(f"No matching strategies for: {args.strategies}")
            return

    print(f"Strategies: {', '.join(strategies.keys())}")

    strategy_results: dict[str, list[dict]] = {name: [] for name in strategies}

    for row in df.itertuples(index=False):
        stat_type = str(row.stat_type)
        y = int(row.over_label)
        line_score = float(row.line_score)
        n_eff_val = float(row.n_eff) if pd.notna(row.n_eff) else None

        if stat_type in EXCLUDED_STAT_TYPES:
            continue
        is_prior_only = stat_type in PRIOR_ONLY_STAT_TYPES

        expert_probs: dict[str, float | None] = {}
        for col in EXPERTS:
            v = getattr(row, col, None)
            if v is not None and pd.notna(v):
                expert_probs[col] = float(v)
            else:
                expert_probs[col] = None

        ctx_prior = get_context_prior(
            context_priors, stat_type=stat_type, line_score=line_score
        )

        for name, strategy in strategies.items():
            if is_prior_only:
                p_final = ctx_prior if ctx_prior is not None else 0.5
            else:
                strategy._current_stat_type = stat_type if hasattr(strategy, "_current_stat_type") else None  # type: ignore[union-attr]
                p_raw = strategy.combine(expert_probs)
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

            strategy_results[name].append(
                {
                    "stat_type": stat_type,
                    "y": y,
                    "p_final": p_final,
                    "correct": correct,
                    "logloss": _logloss(y, p_final),
                    "is_actionable": is_actionable,
                    "is_placed": is_placed,
                    "edge": edge,
                }
            )

        # Update online strategies with outcome
        for strategy in strategies.values():
            strategy.update(expert_probs, y, stat_type=stat_type)

    comparison = _build_comparison(strategy_results)
    best_per_window = _best_strategy_per_window(comparison)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_back": args.days_back,
        "strategies": comparison,
        "best_strategy_per_window": best_per_window,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nBacktest report -> {out}")

    _print_comparison(comparison)

    print(f"\nBest strategy per window:")
    for wk, name in sorted(best_per_window.items()):
        print(f"  {wk}: {name}")


if __name__ == "__main__":
    main()
