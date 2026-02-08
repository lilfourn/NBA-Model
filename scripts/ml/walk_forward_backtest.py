"""Walk-forward backtesting framework for the full ML pipeline.

Trains models on sliding windows, predicts on held-out periods,
and outputs comprehensive metrics (ROI, hit rate, calibration, drawdown, Sharpe).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import load_training_data  # noqa: E402
from app.ml.stat_mappings import stat_value_from_row  # noqa: E402
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    if "is_combo" in df.columns:
        df = df[df["is_combo"].fillna(False) == False]  # noqa: E712
    if "player_name" in df.columns:
        df = df[~df["player_name"].fillna("").astype(str).str.contains("+", regex=False)]

    df["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in df.itertuples(index=False)
    ]
    df = df.dropna(subset=["line_score", "actual_value", "fetched_at"])
    if "minutes_to_start" in df.columns:
        df = df[df["minutes_to_start"].fillna(0) >= 0]
    if "is_live" in df.columns:
        df = df[df["is_live"].fillna(False) == False]  # noqa: E712
    if "in_game" in df.columns:
        df = df[df["in_game"].fillna(False) == False]  # noqa: E712

    df["over"] = (df["actual_value"] > df["line_score"]).astype(int)
    df["_sort_ts"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    df = df.dropna(subset=["_sort_ts"]).sort_values("_sort_ts")

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown").astype(str)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype(float)
    if "trending_count" in df.columns:
        df["trending_count"] = np.log1p(df["trending_count"].clip(lower=0.0))

    cat_dummies = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, dtype=float)
    X = pd.concat([cat_dummies, df[NUMERIC_COLS]], axis=1)
    y = df["over"]
    return X, y, df


def _train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
) -> np.ndarray:
    """Train a model and return predicted probabilities on X_test."""
    if model_type == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
            eval_metric="logloss", use_label_encoder=False,
            verbosity=0, random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.5, reg_lambda=2.0, num_leaves=31,
            verbosity=-1, random_state=42,
        )
        model.fit(X_train, y_train)
    elif model_type == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ])
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.predict_proba(X_test)[:, 1].astype(np.float32)


def _ensemble_probs(prob_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Simple average ensemble of available model probabilities."""
    stacked = np.column_stack(list(prob_dict.values()))
    return stacked.mean(axis=1)


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Compute calibration curve bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        result.append({
            "bin_start": round(float(bins[i]), 2),
            "bin_end": round(float(bins[i + 1]), 2),
            "n": n,
            "mean_predicted": round(float(y_prob[mask].mean()), 4),
            "mean_actual": round(float(y_true[mask].mean()), 4),
        })
    return result


def _max_drawdown(hits: np.ndarray) -> int:
    """Max consecutive losses."""
    max_dd = 0
    current = 0
    for h in hits:
        if h == 0:
            current += 1
            max_dd = max(max_dd, current)
        else:
            current = 0
    return max_dd


def _sharpe_ratio(daily_pnl: pd.Series) -> float:
    """Annualized Sharpe ratio from daily P&L."""
    if daily_pnl.std() == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * math.sqrt(252))


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest for the full ML pipeline.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--train-days", type=int, default=60, help="Training window size in days.")
    ap.add_argument("--test-days", type=int, default=14, help="Test window size in days.")
    ap.add_argument("--slide-days", type=int, default=7, help="Slide forward by this many days.")
    ap.add_argument("--models", nargs="*", default=["xgb", "lgbm", "lr"])
    ap.add_argument("--min-confidence", type=float, default=0.52, help="Only count picks with prob > this.")
    ap.add_argument("--output", default="data/reports/backtest_results.json")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    df = load_training_data(engine)
    if df.empty:
        raise SystemExit("No training data.")
    X, y, df_used = _prepare_features(df)
    if len(df_used) < 200:
        raise SystemExit(f"Not enough data for backtesting ({len(df_used)} rows).")

    ts = df_used["_sort_ts"]
    min_date = ts.min()
    max_date = ts.max()
    train_delta = timedelta(days=args.train_days)
    test_delta = timedelta(days=args.test_days)
    slide_delta = timedelta(days=args.slide_days)

    print(f"Data range: {min_date.date()} to {max_date.date()}")
    print(f"Window: train={args.train_days}d, test={args.test_days}d, slide={args.slide_days}d")
    print(f"Models: {args.models}")

    all_predictions = []
    window_start = min_date
    fold = 0

    while window_start + train_delta + test_delta <= max_date:
        train_end = window_start + train_delta
        test_end = train_end + test_delta

        train_mask = (ts >= window_start) & (ts < train_end)
        test_mask = (ts >= train_end) & (ts < test_end)

        X_train_fold = X.loc[train_mask]
        y_train_fold = y.loc[train_mask]
        X_test_fold = X.loc[test_mask]
        y_test_fold = y.loc[test_mask]
        df_test_fold = df_used.loc[test_mask]

        if len(X_train_fold) < 50 or len(X_test_fold) < 10:
            window_start += slide_delta
            continue

        fold += 1
        probs: dict[str, np.ndarray] = {}
        for model_name in args.models:
            try:
                p = _train_and_predict(
                    X_train_fold, y_train_fold, X_test_fold, y_test_fold, model_name
                )
                probs[model_name] = p
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: {model_name} failed in fold {fold}: {exc}")

        if not probs:
            window_start += slide_delta
            continue

        ens_probs = _ensemble_probs(probs)
        picks = ens_probs >= 0.5
        pick_probs = np.where(picks, ens_probs, 1.0 - ens_probs)
        confident_mask = pick_probs >= args.min_confidence

        for i, idx in enumerate(y_test_fold.index):
            pick_over = bool(picks[i])
            actual_over = bool(y_test_fold.loc[idx])
            hit = pick_over == actual_over
            all_predictions.append({
                "fold": fold,
                "date": str(df_test_fold.loc[idx, "_sort_ts"].date()) if pd.notna(df_test_fold.loc[idx, "_sort_ts"]) else None,
                "stat_type": str(df_test_fold.loc[idx, "stat_type"]) if "stat_type" in df_test_fold.columns else None,
                "prob": float(ens_probs[i]),
                "pick": "OVER" if pick_over else "UNDER",
                "actual_over": actual_over,
                "hit": hit,
                "confident": bool(confident_mask[i]),
            })

        n_test = len(y_test_fold)
        conf_hits = sum(1 for p in all_predictions[-n_test:] if p["confident"] and p["hit"])
        conf_total = sum(1 for p in all_predictions[-n_test:] if p["confident"])
        conf_rate = conf_hits / conf_total if conf_total else 0
        print(f"  Fold {fold}: train={len(X_train_fold)} test={n_test} "
              f"confident={conf_total} hit_rate={conf_rate:.1%} "
              f"[{train_end.date()} -> {test_end.date()}]")

        window_start += slide_delta

    if not all_predictions:
        raise SystemExit("No predictions generated. Check data coverage.")

    # Compute overall metrics
    pred_df = pd.DataFrame(all_predictions)
    conf_df = pred_df[pred_df["confident"]]

    hits_all = pred_df["hit"].to_numpy()
    hits_conf = conf_df["hit"].to_numpy() if len(conf_df) else np.array([])
    probs_all = pred_df["prob"].to_numpy()
    actuals_all = pred_df["actual_over"].astype(int).to_numpy()

    overall_hit_rate = float(hits_all.mean())
    confident_hit_rate = float(hits_conf.mean()) if len(hits_conf) else None
    overall_logloss = float(log_loss(actuals_all, probs_all)) if len(actuals_all) > 1 else None
    overall_auc = float(roc_auc_score(actuals_all, probs_all)) if len(np.unique(actuals_all)) > 1 else None

    # Max drawdown
    max_dd = _max_drawdown(hits_conf.astype(int)) if len(hits_conf) else 0

    # Sharpe (unit bets: +1 for win, -1 for loss)
    if len(conf_df) and "date" in conf_df.columns:
        conf_df = conf_df.copy()
        conf_df["pnl"] = conf_df["hit"].astype(int) * 2 - 1
        daily_pnl = conf_df.groupby("date")["pnl"].sum()
        sharpe = _sharpe_ratio(daily_pnl)
    else:
        sharpe = 0.0

    # ROI (unit bets on confident picks)
    if len(hits_conf):
        roi = float((hits_conf.sum() * 2 - len(hits_conf)) / len(hits_conf))
    else:
        roi = 0.0

    # Binomial test: is hit rate significantly > 50%?
    if len(hits_conf) > 0:
        binom_p = float(sp_stats.binom_test(int(hits_conf.sum()), len(hits_conf), 0.5, alternative="greater"))
    else:
        binom_p = 1.0

    # Calibration
    calibration = _calibration_bins(actuals_all, probs_all)

    # Per-stat-type breakdown
    stat_breakdown = {}
    if "stat_type" in pred_df.columns:
        for st, group in pred_df.groupby("stat_type"):
            conf_g = group[group["confident"]]
            stat_breakdown[str(st)] = {
                "total": len(group),
                "confident": len(conf_g),
                "hit_rate": round(float(conf_g["hit"].mean()), 4) if len(conf_g) else None,
            }

    report = {
        "folds": fold,
        "total_predictions": len(pred_df),
        "confident_predictions": len(conf_df),
        "overall_hit_rate": round(overall_hit_rate, 4),
        "confident_hit_rate": round(confident_hit_rate, 4) if confident_hit_rate else None,
        "overall_logloss": round(overall_logloss, 4) if overall_logloss else None,
        "overall_auc": round(overall_auc, 4) if overall_auc else None,
        "roi_unit_bets": round(roi, 4),
        "max_drawdown_consecutive_losses": max_dd,
        "sharpe_ratio": round(sharpe, 2),
        "binomial_p_value": round(binom_p, 6),
        "significant_at_5pct": binom_p < 0.05,
        "calibration_bins": calibration,
        "stat_type_breakdown": stat_breakdown,
        "config": {
            "train_days": args.train_days,
            "test_days": args.test_days,
            "slide_days": args.slide_days,
            "models": args.models,
            "min_confidence": args.min_confidence,
        },
    }

    print(f"\n{'='*60}")
    print(f"Walk-Forward Backtest Results ({fold} folds)")
    print(f"{'='*60}")
    print(f"Total predictions: {len(pred_df)}")
    print(f"Confident predictions (>{args.min_confidence:.0%}): {len(conf_df)}")
    print(f"Overall hit rate: {overall_hit_rate:.1%}")
    print(f"Confident hit rate: {confident_hit_rate:.1%}" if confident_hit_rate else "Confident hit rate: N/A")
    print(f"ROI (unit bets): {roi:+.1%}")
    print(f"Max drawdown: {max_dd} consecutive losses")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Binomial p-value: {binom_p:.4f} {'*' if binom_p < 0.05 else ''}")
    if overall_auc:
        print(f"AUC: {overall_auc:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {output}")


if __name__ == "__main__":
    main()
