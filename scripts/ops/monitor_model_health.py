"""Model health monitoring: rolling accuracy, log loss, weight collapse, alerts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.selection_policy import SelectionPolicy  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]
ROLLING_WINDOW = 50
METRICS_VERSION = (
    "2.1.0"  # Bump when metric logic changes (push/void, thresholds, etc.)
)
ACCURACY_ALERT_THRESHOLD = 0.48
LOGLOSS_ALERT_THRESHOLD = 0.75
WEIGHT_COLLAPSE_THRESHOLD = 0.80
MIN_ALERT_EXPERT_WEIGHT = 0.03
TOP_STAT_SHARE_ALERT_THRESHOLD = 0.70
MIN_PUBLISHABLE_FOR_CONCENTRATION_ALERT = 50
MIN_REQUIRED_EXPERT_COVERAGE = 0.90
REQUIRED_EXPERTS = tuple(EXPERT_COLS)
MIN_PUBLISHABLE_SHARE = 0.15


def _logloss(y: int, p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


CANONICAL_VIEW = "vw_resolved_picks_canonical"

# Fallback SQL used when the canonical view does not exist yet (pre-migration).
_FALLBACK_SQL = """
select
    id::text as prediction_id,
    prob_over as p_final,
    p_raw,
    p_forecast_cal, p_nn, coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl, p_lr, p_xgb, p_lgbm,
    cast(details->>'conformal_set_size' as integer) as conformal_set_size,
    cast(details->>'selection_threshold' as float) as selection_threshold,
    over_label, is_correct, outcome, stat_type, rank_score, n_eff,
    coalesce(decision_time, resolved_at, created_at) as event_time,
    coalesce(decision_time, created_at) as decision_time,
    created_at
from projection_predictions
where over_label is not null
  and actual_value is not null
  and outcome in ('over', 'under')
  and coalesce(decision_time, resolved_at, created_at) >= now() - (:days_back * interval '1 day')
order by coalesce(decision_time, resolved_at, created_at) asc, created_at asc, id asc
"""


def _load_resolved_predictions(engine, *, days_back: int = 90) -> pd.DataFrame:
    """Load resolved predictions from canonical view (with fallback to raw table)."""
    from sqlalchemy import text as sa_text

    canonical_sql = f"""
select
    id::text as prediction_id,
    p_final, p_raw,
    p_forecast_cal, p_nn, p_tabdl, p_lr, p_xgb, p_lgbm,
    cast(details->>'conformal_set_size' as integer) as conformal_set_size,
    cast(details->>'selection_threshold' as float) as selection_threshold,
    over_label, is_correct, outcome, stat_type, rank_score, n_eff,
    coalesce(decision_time, resolved_at, created_at) as event_time,
    decision_time, created_at
from {CANONICAL_VIEW}
where coalesce(decision_time, resolved_at, created_at) >= now() - (:days_back * interval '1 day')
order by coalesce(decision_time, resolved_at, created_at) asc, created_at asc, id asc
"""
    try:
        df = pd.read_sql(
            sa_text(canonical_sql),
            engine,
            params={"days_back": int(max(1, days_back))},
        )
    except Exception:  # noqa: BLE001
        # Fallback if view doesn't exist yet
        df = pd.read_sql(
            sa_text(_FALLBACK_SQL),
            engine,
            params={"days_back": int(max(1, days_back))},
        )
    for col in EXPERT_COLS + [
        "p_final",
        "p_raw",
        "n_eff",
        "rank_score",
        "conformal_set_size",
        "selection_threshold",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_correct" in df.columns:
        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
    return df


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _load_latest_collect_predictions(engine) -> pd.DataFrame:
    from sqlalchemy import text as sa_text

    query = sa_text(
        """
        with latest_snapshot as (
            select snapshot_id
            from projection_predictions
            where snapshot_id is not null
            group by snapshot_id
            order by max(coalesce(decision_time, created_at)) desc
            limit 1
        ),
        latest as (
            select distinct on (pp.projection_id)
                pp.snapshot_id::text as snapshot_id,
                pp.projection_id::text as projection_id,
                pp.stat_type,
                pp.p_forecast_cal,
                pp.p_nn,
                coalesce(pp.p_tabdl::text, pp.details->>'p_tabdl') as p_tabdl,
                pp.p_lr,
                pp.p_xgb,
                pp.p_lgbm,
                pp.details,
                coalesce(pp.decision_time, pp.created_at) as scored_at
            from projection_predictions pp
            join latest_snapshot ls on ls.snapshot_id = pp.snapshot_id
            order by
                pp.projection_id,
                coalesce(pp.decision_time, pp.created_at) desc,
                pp.created_at desc,
                pp.id desc
        )
        select * from latest
        """
    )
    return pd.read_sql(query, engine)


def _compute_latest_collect_metrics(engine) -> dict:
    try:
        latest = _load_latest_collect_predictions(engine)
    except Exception:  # noqa: BLE001
        latest = pd.DataFrame()
    if latest.empty:
        return {
            "snapshot_id": None,
            "total_scored": 0,
            "publishable_count": 0,
            "publishable_by_stat": {},
            "top_stat_share": 0.0,
            "expert_coverage": {},
        }

    details_series = latest.get("details")
    if details_series is None:
        details_series = pd.Series([{}] * len(latest), index=latest.index, dtype=object)

    def _extract_is_publishable(raw: object) -> bool:
        if isinstance(raw, dict):
            return _coerce_bool(raw.get("is_publishable"))
        if isinstance(raw, str):
            text_value = raw.strip()
            if not text_value:
                return False
            try:
                parsed = json.loads(text_value)
            except json.JSONDecodeError:
                return False
            if isinstance(parsed, dict):
                return _coerce_bool(parsed.get("is_publishable"))
        return False

    publishable_mask = details_series.apply(_extract_is_publishable).astype(bool)
    publishable = latest[publishable_mask]
    publishable_by_stat = (
        publishable["stat_type"].fillna("unknown").astype(str).value_counts().to_dict()
        if not publishable.empty and "stat_type" in publishable.columns
        else {}
    )
    publishable_count = int(publishable_mask.sum())
    top_stat_share = (
        float(max(publishable_by_stat.values()) / publishable_count)
        if publishable_count > 0 and publishable_by_stat
        else 0.0
    )

    expert_coverage: dict[str, float] = {}
    for col in EXPERT_COLS:
        if col not in latest.columns:
            expert_coverage[col] = 0.0
            continue
        series = pd.to_numeric(latest[col], errors="coerce")
        expert_coverage[col] = float(series.notna().mean()) if len(series) else 0.0

    snapshot_id = None
    if "snapshot_id" in latest.columns and not latest["snapshot_id"].dropna().empty:
        snapshot_id = str(latest["snapshot_id"].dropna().iloc[0])

    return {
        "snapshot_id": snapshot_id,
        "total_scored": int(len(latest)),
        "publishable_count": publishable_count,
        "publishable_by_stat": {
            str(key): int(value) for key, value in publishable_by_stat.items()
        },
        "top_stat_share": round(top_stat_share, 4),
        "expert_coverage": {
            str(key): round(float(value), 4) for key, value in expert_coverage.items()
        },
    }


def _load_recent_collect_summaries(engine, *, limit: int = 5) -> list[dict]:
    from sqlalchemy import text as sa_text

    query = sa_text(
        """
        with snapshots as (
            select
                snapshot_id,
                max(coalesce(decision_time, created_at)) as scored_at
            from projection_predictions
            where snapshot_id is not null
            group by snapshot_id
            order by scored_at desc
            limit :limit
        ),
        latest as (
            select
                pp.snapshot_id,
                pp.projection_id,
                pp.details,
                row_number() over (
                    partition by pp.snapshot_id, pp.projection_id
                    order by
                        coalesce(pp.decision_time, pp.created_at) desc,
                        pp.created_at desc,
                        pp.id desc
                ) as rn
            from projection_predictions pp
            join snapshots s on s.snapshot_id = pp.snapshot_id
        )
        select
            s.snapshot_id::text as snapshot_id,
            s.scored_at,
            count(l.projection_id) as total_scored,
            sum(
                case
                    when lower(coalesce(l.details->>'is_publishable', '')) in ('1', 'true', 'yes', 'y')
                    then 1 else 0
                end
            ) as publishable_count
        from snapshots s
        left join latest l
            on l.snapshot_id = s.snapshot_id
           and l.rn = 1
        group by s.snapshot_id, s.scored_at
        order by s.scored_at desc
        """
    )
    try:
        df = pd.read_sql(query, engine, params={"limit": int(max(1, limit))})
    except Exception:  # noqa: BLE001
        return []
    if df.empty:
        return []
    out: list[dict] = []
    for row in df.itertuples(index=False):
        total = int(row.total_scored or 0)
        publishable = int(row.publishable_count or 0)
        ratio = (publishable / total) if total > 0 else 0.0
        scored_at = pd.to_datetime(row.scored_at, utc=True, errors="coerce")
        out.append(
            {
                "snapshot_id": str(row.snapshot_id),
                "scored_at": scored_at.isoformat() if pd.notna(scored_at) else None,
                "total_scored": total,
                "publishable_count": publishable,
                "publishable_ratio": round(float(ratio), 4),
            }
        )
    return out


def _load_calibrator_flat_stat_types(
    path: str = "models/stat_calibrator.joblib", *, engine=None
) -> list[str]:
    try:
        calibrator_path = path
        if engine is not None:
            try:
                from app.ml.artifact_store import load_latest_artifact_as_file

                db_path = load_latest_artifact_as_file(
                    engine, "stat_calibrator", suffix=".joblib"
                )
                if db_path:
                    calibrator_path = str(db_path)
            except Exception:  # noqa: BLE001
                pass

        from app.ml.stat_calibrator import StatTypeCalibrator

        cal = StatTypeCalibrator.load(calibrator_path)
        meta = cal.meta or {}
        stats = meta.get("degenerate_stats") or []
        if isinstance(stats, list):
            return sorted({str(st) for st in stats if st})
    except Exception:  # noqa: BLE001
        return []
    return []


def _load_ensemble_mean_weights(weights_path: str) -> dict[str, float]:
    path = Path(weights_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    weights = data.get("weights") or data.get("global_weights") or {}
    if (
        isinstance(weights, dict)
        and weights
        and all(isinstance(v, dict) for v in weights.values())
    ):
        nested_weights = weights
        all_w: dict[str, list[float]] = {}
        for bucket_weights in nested_weights.values():
            for expert, value in bucket_weights.items():
                all_w.setdefault(str(expert), []).append(float(value))
        weights = {expert: float(np.mean(values)) for expert, values in all_w.items()}
    if not weights:
        buckets = data.get("context_weights", {})
        if buckets:
            all_w: dict[str, list[float]] = {}
            for bucket_weights in buckets.values():
                if isinstance(bucket_weights, dict):
                    for k, v in bucket_weights.items():
                        all_w.setdefault(k, []).append(float(v))
            weights = {k: float(np.mean(vs)) for k, vs in all_w.items()}

    out: dict[str, float] = {}
    for expert, value in (weights or {}).items():
        try:
            out[str(expert)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _inversion_test(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute accuracy/logloss for p and 1-p to detect direction mismatch."""
    picks_normal = (probs >= 0.5).astype(int)
    picks_inv = (probs < 0.5).astype(int)  # equivalent to (1-p >= 0.5)
    acc_normal = float((picks_normal == labels).mean())
    acc_inv = float((picks_inv == labels).mean())
    inv_probs = 1.0 - probs
    ll_normal = float(
        np.mean([_logloss(int(y), float(p)) for y, p in zip(labels, probs)])
    )
    ll_inv = float(
        np.mean([_logloss(int(y), float(p)) for y, p in zip(labels, inv_probs)])
    )
    return {
        "accuracy": round(acc_normal, 4),
        "accuracy_inverted": round(acc_inv, 4),
        "logloss": round(ll_normal, 4),
        "logloss_inverted": round(ll_inv, 4),
        "inversion_improves_accuracy": bool(acc_inv > acc_normal + 0.02),
        "inversion_improves_logloss": bool(ll_inv < ll_normal - 0.02),
    }


def _confusion_matrix(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute confusion matrix counts at the 0.5 threshold."""
    picks = (probs >= 0.5).astype(int)
    tp = int(((picks == 1) & (labels == 1)).sum())
    fp = int(((picks == 1) & (labels == 0)).sum())
    tn = int(((picks == 0) & (labels == 0)).sum())
    fn = int(((picks == 0) & (labels == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _check_rolling_metrics(
    joined: pd.DataFrame,
    *,
    ensemble_weights: dict[str, float] | None = None,
    min_alert_weight: float = MIN_ALERT_EXPERT_WEIGHT,
) -> dict[str, list[dict]]:
    """Check rolling accuracy and logloss for each expert + ensemble."""
    alerts: list[dict] = []
    suppressed_alerts: list[dict] = []
    metrics: dict[str, dict] = {}

    # Compute base rate from all valid labels
    valid_labels = joined["over_label"].dropna()
    base_rate = round(float(valid_labels.mean()), 4) if len(valid_labels) > 0 else None

    # Include p_raw in analysis if available
    extra_cols = ["p_final"]
    if "p_raw" in joined.columns and joined["p_raw"].notna().any():
        extra_cols.append("p_raw")

    for expert in EXPERT_COLS + extra_cols:
        if expert not in joined.columns:
            continue
        weight_map = ensemble_weights or {}
        has_weight_map = bool(weight_map)
        if expert in EXPERT_COLS:
            default_weight = 0.0 if has_weight_map else 1.0
            expert_weight = float(weight_map.get(expert, default_weight))
            alert_eligible = expert_weight >= float(min_alert_weight)
            if has_weight_map and expert not in weight_map:
                suppression_reason = "expert not present in active ensemble"
            else:
                suppression_reason = (
                    f"ensemble_weight={expert_weight:.6f} "
                    f"< min_alert_weight={float(min_alert_weight):.3f}"
                )
        else:
            expert_weight = 1.0
            alert_eligible = True
            suppression_reason = ""
        valid = joined.dropna(subset=[expert, "over_label"])
        if len(valid) < ROLLING_WINDOW:
            continue

        recent = valid.tail(ROLLING_WINDOW)
        probs = recent[expert].to_numpy(dtype=float)
        labels = recent["over_label"].to_numpy(dtype=int)

        if expert == "p_final" and "is_correct" in recent.columns:
            is_correct = pd.to_numeric(recent["is_correct"], errors="coerce").dropna()
            if len(is_correct) == len(recent):
                accuracy = float(is_correct.mean())
            else:
                picks = (probs >= 0.5).astype(int)
                accuracy = float((picks == labels).mean())
        else:
            picks = (probs >= 0.5).astype(int)
            accuracy = float((picks == labels).mean())
        avg_ll = float(
            np.mean([_logloss(int(y), float(p)) for y, p in zip(labels, probs)])
        )

        # Inversion test and confusion matrix
        inv_test = _inversion_test(probs, labels)
        cm = _confusion_matrix(probs, labels)
        window_base_rate = round(float(labels.mean()), 4)

        metrics[expert] = {
            "rolling_accuracy": round(accuracy, 4),
            "rolling_logloss": round(avg_ll, 4),
            "n": int(len(recent)),
            "ensemble_weight": round(expert_weight, 6),
            "alert_eligible": bool(alert_eligible),
            "base_rate": window_base_rate,
            "inversion_test": inv_test,
            "confusion_matrix": cm,
        }

        if accuracy < ACCURACY_ALERT_THRESHOLD:
            payload = {
                "type": "low_accuracy",
                "expert": expert,
                "value": round(accuracy, 4),
                "threshold": ACCURACY_ALERT_THRESHOLD,
                "message": f"{expert} rolling {ROLLING_WINDOW}-bet accuracy {accuracy:.1%} < {ACCURACY_ALERT_THRESHOLD:.0%}",
            }
            if alert_eligible:
                alerts.append(payload)
            else:
                payload["suppression_reason"] = suppression_reason
                suppressed_alerts.append(payload)
        if avg_ll > LOGLOSS_ALERT_THRESHOLD:
            payload = {
                "type": "high_logloss",
                "expert": expert,
                "value": round(avg_ll, 4),
                "threshold": LOGLOSS_ALERT_THRESHOLD,
                "message": f"{expert} rolling logloss {avg_ll:.3f} > {LOGLOSS_ALERT_THRESHOLD}",
            }
            if alert_eligible:
                alerts.append(payload)
            else:
                payload["suppression_reason"] = suppression_reason
                suppressed_alerts.append(payload)

    return {
        "metrics": metrics,
        "base_rate": base_rate,
        "alerts": alerts,
        "suppressed_alerts": suppressed_alerts,
    }


def _check_ensemble_weights(weights_path: str) -> list[dict]:
    """Check if any expert dominates the ensemble weights."""
    alerts = []
    weights = _load_ensemble_mean_weights(weights_path)

    if not weights:
        return alerts

    total = sum(weights.values())
    if total <= 0:
        return alerts

    for expert, w in weights.items():
        ratio = w / total
        if ratio > WEIGHT_COLLAPSE_THRESHOLD:
            alerts.append(
                {
                    "type": "weight_collapse",
                    "expert": expert,
                    "value": round(ratio, 4),
                    "threshold": WEIGHT_COLLAPSE_THRESHOLD,
                    "message": f"{expert} has {ratio:.0%} of ensemble weight (collapse risk)",
                }
            )

    return alerts


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean((p - y)^2).  Lower is better."""
    return float(np.mean((probs - labels) ** 2))


def _expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 5
) -> dict:
    """Compute ECE and return a calibration table with n_bins bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict] = []
    total_ece = 0.0
    total_n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        n = len(bin_probs)
        if n == 0:
            table.append(
                {
                    "bin": f"[{lo:.2f},{hi:.2f})",
                    "n": 0,
                    "mean_pred": None,
                    "mean_actual": None,
                }
            )
            continue
        mean_pred = float(np.mean(bin_probs))
        mean_actual = float(np.mean(bin_labels))
        table.append(
            {
                "bin": f"[{lo:.2f},{hi:.2f})"
                if i < n_bins - 1
                else f"[{lo:.2f},{hi:.2f}]",
                "n": int(n),
                "mean_pred": round(mean_pred, 4),
                "mean_actual": round(mean_actual, 4),
            }
        )
        total_ece += abs(mean_pred - mean_actual) * (n / total_n)

    return {
        "ece": round(total_ece, 4),
        "n_bins": n_bins,
        "calibration_table": table,
    }


def _compute_calibration_diagnostics(joined: pd.DataFrame) -> dict:
    """Compute calibration diagnostics overall and per stat_type."""
    if (
        joined.empty
        or "p_final" not in joined.columns
        or "over_label" not in joined.columns
    ):
        return {}

    valid = joined.dropna(subset=["p_final", "over_label"]).copy()
    if len(valid) < 10:
        return {}

    probs = valid["p_final"].to_numpy(dtype=float)
    labels = valid["over_label"].to_numpy(dtype=int)

    overall_brier = round(_brier_score(probs, labels), 4)
    overall_ece = _expected_calibration_error(probs, labels)

    result: dict = {
        "overall": {
            "brier_score": overall_brier,
            **overall_ece,
        },
    }

    # Per stat_type breakdown
    if "stat_type" in valid.columns:
        per_stat: dict[str, dict] = {}
        for st, group in valid.groupby("stat_type"):
            if len(group) < 10:
                continue
            st_probs = group["p_final"].to_numpy(dtype=float)
            st_labels = group["over_label"].to_numpy(dtype=int)
            per_stat[str(st)] = {
                "n": int(len(group)),
                "brier_score": round(_brier_score(st_probs, st_labels), 4),
                **_expected_calibration_error(st_probs, st_labels),
            }
        result["per_stat_type"] = per_stat

    return result


def _compute_tier_metrics(joined: pd.DataFrame) -> dict:
    """Compute accuracy metrics for scored vs actionable pick tiers.

    Tiers:
      - scored: all resolved rows with p_final
      - actionable: rows passing adaptive per-stat selection policy threshold
    """
    if (
        joined.empty
        or "p_final" not in joined.columns
        or "over_label" not in joined.columns
    ):
        return {}

    valid = joined.dropna(subset=["p_final", "over_label"]).copy()
    if valid.empty:
        return {}

    probs = valid["p_final"].to_numpy(dtype=float)
    labels = valid["over_label"].to_numpy(dtype=int)
    picks = (probs >= 0.5).astype(int)
    p_pick = np.maximum(probs, 1.0 - probs)

    scored_correct = (picks == labels).astype(float)
    scored_acc = float(scored_correct.mean())
    scored_n = int(len(valid))

    policy = SelectionPolicy.load()
    stat_types = (
        valid["stat_type"].astype(str)
        if "stat_type" in valid.columns
        else pd.Series(["unknown"] * len(valid), index=valid.index, dtype=str)
    )
    conformal_sizes = (
        valid["conformal_set_size"]
        if "conformal_set_size" in valid.columns
        else pd.Series([None] * len(valid), index=valid.index, dtype=object)
    )
    thresholds = np.array(
        [
            float(policy.threshold_for(str(st), int(cs) if pd.notna(cs) else None))
            for st, cs in zip(stat_types.tolist(), conformal_sizes.tolist())
        ],
        dtype=float,
    )
    if "selection_threshold" in valid.columns:
        explicit = pd.to_numeric(valid["selection_threshold"], errors="coerce").to_numpy(dtype=float)
        thresholds = np.where(np.isfinite(explicit), explicit, thresholds)

    actionable_mask = p_pick >= thresholds
    actionable_n = int(actionable_mask.sum())
    if actionable_n > 0:
        actionable_acc = float(scored_correct[actionable_mask].mean())
    else:
        actionable_acc = None

    coverage = round(actionable_n / scored_n, 4) if scored_n > 0 else 0.0

    # Placed tier: actionable AND conformal_set_size != 2 (if available)
    placed_mask = actionable_mask.copy()
    if "conformal_set_size" in valid.columns:
        conf_ok = valid["conformal_set_size"].fillna(1).to_numpy() != 2
        placed_mask = placed_mask & conf_ok
    placed_n = int(placed_mask.sum())
    placed_acc = float(scored_correct[placed_mask].mean()) if placed_n > 0 else None
    placed_coverage = round(placed_n / scored_n, 4) if scored_n > 0 else 0.0

    # Direction slice for actionable picks.
    actionable_pick_dirs = np.where(probs >= 0.5, "OVER", "UNDER")
    actionable_by_direction: dict[str, dict[str, float | int | None]] = {}
    for direction in ("OVER", "UNDER"):
        mask = actionable_mask & (actionable_pick_dirs == direction)
        n_dir = int(mask.sum())
        actionable_by_direction[direction] = {
            "n": n_dir,
            "accuracy": round(float(scored_correct[mask].mean()), 4) if n_dir > 0 else None,
            "coverage": round(n_dir / scored_n, 4) if scored_n > 0 else 0.0,
        }

    # Confidence deciles over p_pick (for calibration/slice diagnostics).
    confidence_deciles: list[dict[str, float | int | str]] = []
    if scored_n > 0:
        bins = np.linspace(0.5, 1.0, 11)
        decile_labels = [
            "50-55",
            "55-60",
            "60-65",
            "65-70",
            "70-75",
            "75-80",
            "80-85",
            "85-90",
            "90-95",
            "95-100",
        ]
        binned = pd.cut(p_pick, bins=bins, include_lowest=True, right=True, labels=decile_labels)
        diag_df = pd.DataFrame(
            {
                "decile": binned.astype(str),
                "p_pick": p_pick,
                "correct": scored_correct,
            }
        )
        for decile, grp in diag_df.groupby("decile", observed=False):
            if decile == "nan" or len(grp) == 0:
                continue
            confidence_deciles.append(
                {
                    "decile": str(decile),
                    "n": int(len(grp)),
                    "coverage": round(len(grp) / scored_n, 4),
                    "accuracy": round(float(grp["correct"].mean()), 4),
                    "mean_p_pick": round(float(grp["p_pick"].mean()), 4),
                }
            )

    return {
        "scored": {"n": scored_n, "accuracy": round(scored_acc, 4)},
        "actionable": {
            "n": actionable_n,
            "accuracy": round(actionable_acc, 4)
            if actionable_acc is not None
            else None,
            "threshold": round(float(policy.global_threshold), 4),
            "policy_version": policy.version,
        },
        "placed": {
            "n": placed_n,
            "accuracy": round(placed_acc, 4) if placed_acc is not None else None,
        },
        "coverage": coverage,
        "placed_coverage": placed_coverage,
        "actionable_by_direction": actionable_by_direction,
        "confidence_deciles": confidence_deciles,
    }


def _latest_event_time_iso(joined: pd.DataFrame) -> str | None:
    if joined.empty or "event_time" not in joined.columns:
        return None
    event_times = pd.to_datetime(
        joined["event_time"], errors="coerce", utc=True
    ).dropna()
    if event_times.empty:
        return None
    return event_times.max().isoformat()


def build_health_report(
    engine,
    *,
    days_back: int = 90,
    ensemble_weights_path: str = "models/ensemble_weights.json",
    min_alert_weight: float = MIN_ALERT_EXPERT_WEIGHT,
) -> dict:
    joined = _load_resolved_predictions(engine, days_back=int(max(1, days_back)))
    all_alerts: list[dict] = []
    suppressed_alerts: list[dict] = []
    expert_metrics: dict[str, dict] = {}

    ensemble_weights = _load_ensemble_mean_weights(ensemble_weights_path)
    latest_collect = _compute_latest_collect_metrics(engine)
    recent_collects = _load_recent_collect_summaries(engine, limit=5)
    calibrator_flat_stat_types = _load_calibrator_flat_stat_types(engine=engine)

    base_rate = None
    tier_metrics: dict = {}
    calibration_diagnostics: dict = {}
    if not joined.empty:
        result = _check_rolling_metrics(
            joined,
            ensemble_weights=ensemble_weights,
            min_alert_weight=float(min_alert_weight),
        )
        all_alerts.extend(result["alerts"])
        suppressed_alerts = result.get("suppressed_alerts", [])
        expert_metrics = result["metrics"]
        base_rate = result.get("base_rate")
        tier_metrics = _compute_tier_metrics(joined)
        calibration_diagnostics = _compute_calibration_diagnostics(joined)

    weight_alerts = _check_ensemble_weights(ensemble_weights_path)
    all_alerts.extend(weight_alerts)

    top_stat_share = float(latest_collect.get("top_stat_share") or 0.0)
    publishable_count = int(latest_collect.get("publishable_count") or 0)
    if top_stat_share > TOP_STAT_SHARE_ALERT_THRESHOLD:
        payload = {
            "type": "stat_concentration",
            "value": round(top_stat_share, 4),
            "threshold": TOP_STAT_SHARE_ALERT_THRESHOLD,
            "message": (
                f"Top publishable stat share {top_stat_share:.1%} exceeds "
                f"{TOP_STAT_SHARE_ALERT_THRESHOLD:.0%}"
            ),
        }
        if publishable_count >= MIN_PUBLISHABLE_FOR_CONCENTRATION_ALERT:
            all_alerts.append(payload)
        else:
            payload["suppression_reason"] = (
                "low_slate_exception: publishable_count="
                f"{publishable_count} < {MIN_PUBLISHABLE_FOR_CONCENTRATION_ALERT}"
            )
            suppressed_alerts.append(payload)

    expert_coverage = latest_collect.get("expert_coverage") or {}
    for expert in REQUIRED_EXPERTS:
        coverage = float(expert_coverage.get(expert, 0.0))
        if coverage < MIN_REQUIRED_EXPERT_COVERAGE:
            all_alerts.append(
                {
                    "type": "expert_coverage_low",
                    "expert": str(expert),
                    "value": round(coverage, 4),
                    "threshold": MIN_REQUIRED_EXPERT_COVERAGE,
                    "message": (
                        f"{expert} coverage {coverage:.1%} < "
                        f"{MIN_REQUIRED_EXPERT_COVERAGE:.0%} on latest collect"
                    ),
                }
            )

    if len(recent_collects) >= 2:
        low_publishable = [
            row
            for row in recent_collects[:2]
            if int(row.get("total_scored") or 0) > 0
            and float(row.get("publishable_ratio") or 0.0) < MIN_PUBLISHABLE_SHARE
        ]
        if len(low_publishable) == 2:
            all_alerts.append(
                {
                    "type": "low_publishable_ratio",
                    "value": [row.get("publishable_ratio") for row in low_publishable],
                    "threshold": MIN_PUBLISHABLE_SHARE,
                    "message": (
                        "Publishable ratio below "
                        f"{MIN_PUBLISHABLE_SHARE:.0%} for two consecutive collects"
                    ),
                }
            )

    if calibrator_flat_stat_types:
        all_alerts.append(
            {
                "type": "calibrator_flat_stats",
                "value": calibrator_flat_stat_types,
                "message": (
                    "Per-stat calibrator marked degenerate stats: "
                    + ", ".join(calibrator_flat_stat_types[:8])
                ),
            }
        )

    return {
        "metrics_version": METRICS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "live_db",
        "days_back": int(max(1, days_back)),
        "time_column": "coalesce(decision_time, resolved_at, created_at)",
        "latest_event_time": _latest_event_time_iso(joined),
        "prediction_log_rows": int(len(joined)),
        "outcome_rows": int(len(joined)) if not joined.empty else 0,
        "rolling_window": ROLLING_WINDOW,
        "base_rate": base_rate,
        "tier_metrics": tier_metrics,
        "calibration_diagnostics": calibration_diagnostics,
        "latest_collect": latest_collect,
        "recent_collects": recent_collects,
        "publishable_by_stat": latest_collect.get("publishable_by_stat", {}),
        "top_stat_share": latest_collect.get("top_stat_share", 0.0),
        "expert_coverage": latest_collect.get("expert_coverage", {}),
        "calibrator_flat_stat_types": calibrator_flat_stat_types,
        "expert_metrics": expert_metrics,
        "alerts": all_alerts,
        "suppressed_alerts": suppressed_alerts,
        "alert_count": len(all_alerts),
        "suppressed_alert_count": len(suppressed_alerts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Monitor model health from prediction log."
    )
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--days-back", type=int, default=90)
    ap.add_argument("--ensemble-weights", default="models/ensemble_weights.json")
    ap.add_argument("--output", default="data/reports/model_health.json")
    ap.add_argument(
        "--alert-email", action="store_true", help="Send email if alerts fire."
    )
    ap.add_argument(
        "--upload-db",
        action="store_true",
        help="Upload health report to DB artifact store.",
    )
    args = ap.parse_args()

    load_env()

    engine = get_engine(args.database_url)
    report = build_health_report(
        engine,
        days_back=int(args.days_back),
        ensemble_weights_path=str(args.ensemble_weights),
        min_alert_weight=MIN_ALERT_EXPERT_WEIGHT,
    )
    expert_metrics = report.get("expert_metrics", {})
    all_alerts = report.get("alerts", [])
    suppressed_alerts = report.get("suppressed_alerts", [])

    base_rate = report.get("base_rate")
    if base_rate is not None:
        print(f"Base rate (mean over_label): {base_rate:.4f}")

    tier = report.get("tier_metrics", {})
    if tier:
        scored = tier.get("scored", {})
        actionable = tier.get("actionable", {})
        coverage = tier.get("coverage", 0)
        print("\nTier breakdown:")
        print(
            f"  Scored:     n={scored.get('n', 0)} acc={scored.get('accuracy', 'N/A')}"
        )
        act_acc = actionable.get("accuracy")
        act_acc_str = f"{act_acc:.1%}" if act_acc is not None else "N/A"
        print(
            f"  Actionable: n={actionable.get('n', 0)} acc={act_acc_str} "
            f"(threshold={actionable.get('threshold', 'N/A')})"
        )
        print(f"  Coverage:   {coverage:.1%}")

    latest_collect = report.get("latest_collect", {})
    if latest_collect:
        total_scored = int(latest_collect.get("total_scored") or 0)
        publishable_count = int(latest_collect.get("publishable_count") or 0)
        top_stat_share = float(latest_collect.get("top_stat_share") or 0.0)
        print("\nLatest collect telemetry:")
        print(f"  Scored rows:      {total_scored}")
        print(f"  Publishable rows: {publishable_count}")
        print(f"  Top stat share:   {top_stat_share:.1%}")

    if expert_metrics:
        print(f"\nRolling {ROLLING_WINDOW}-bet metrics:")
        for expert, m in expert_metrics.items():
            status = "OK"
            if not bool(m.get("alert_eligible", True)):
                status = "INFO"
            elif m["rolling_accuracy"] < ACCURACY_ALERT_THRESHOLD:
                status = "ALERT"
            elif m["rolling_logloss"] > LOGLOSS_ALERT_THRESHOLD:
                status = "WARN"
            weight_text = ""
            if expert in EXPERT_COLS:
                weight_text = f" w={float(m.get('ensemble_weight', 0.0)):.3f}"
            print(
                f"  {expert:<20} acc={m['rolling_accuracy']:.1%} ll={m['rolling_logloss']:.3f}"
                f"{weight_text} [{status}]"
            )
            # Print inversion test results
            inv = m.get("inversion_test")
            if inv:
                inv_flag = (
                    " *** INVERSION DETECTED ***"
                    if inv.get("inversion_improves_accuracy")
                    else ""
                )
                print(
                    f"    inversion: acc(p)={inv['accuracy']:.1%} acc(1-p)={inv['accuracy_inverted']:.1%} "
                    f"ll(p)={inv['logloss']:.3f} ll(1-p)={inv['logloss_inverted']:.3f}{inv_flag}"
                )
            cm = m.get("confusion_matrix")
            if cm:
                print(
                    f"    confusion: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']} "
                    f"base_rate={m.get('base_rate', 'N/A')}"
                )
    else:
        print("No outcomes available yet for rolling metrics.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nHealth report -> {output}")

    if args.upload_db:
        try:
            from app.ml.artifact_store import upload_file

            row_id = upload_file(
                engine, model_name="model_health", file_path=output
            )
            print(f"  Uploaded to DB -> {row_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: DB upload failed: {exc}")

    if all_alerts:
        print(f"\n{'!'*50}")
        print(f"  {len(all_alerts)} ALERT(S) FIRED:")
        for a in all_alerts:
            print(f"  - {a['message']}")
        print(f"{'!'*50}")

        if args.alert_email:
            email_script = ROOT / "scripts" / "ops" / "send_email.py"
            if email_script.exists():
                import subprocess

                body = "\n".join(a["message"] for a in all_alerts)
                body_file = output.parent / "health_alert_body.txt"
                body_file.write_text(body, encoding="utf-8")
                subprocess.run(
                    [
                        sys.executable,
                        str(email_script),
                        "--subject",
                        "Model Health Alert",
                        "--body-file",
                        str(body_file),
                    ],
                    check=False,
                )
    else:
        print("\nAll models healthy.")
    if suppressed_alerts:
        print(f"Suppressed alerts (audit only): {len(suppressed_alerts)}")


if __name__ == "__main__":
    main()
