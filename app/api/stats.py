from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from sqlalchemy import text

from app.core.config import settings
from app.db.engine import get_async_engine, get_engine
from app.ml.selection_policy import SelectionPolicy

router = APIRouter(prefix="/api/stats", tags=["stats"])

ENSEMBLE_WEIGHTS_PATH = Path(settings.ensemble_weights_path)
CALIBRATION_DIR = Path("data/calibration")
EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]


@router.get("/training-history")
async def training_history() -> dict:
    engine = get_async_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                select id, created_at, model_name, train_rows, metrics, params
                from model_runs
                order by created_at asc
                """
            )
        )
        rows = result.all()

    runs = []
    for r in rows:
        metrics = r.metrics or {}
        runs.append(
            {
                "id": str(r.id),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "model_name": r.model_name,
                "train_rows": r.train_rows,
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "logloss": metrics.get("logloss"),
                "conformal_q_hat": metrics.get("conformal_q_hat"),
            }
        )
    return {"runs": runs}


@router.get("/expert-comparison")
async def expert_comparison() -> dict:
    engine = get_async_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                select distinct on (model_name)
                    id, created_at, model_name, train_rows, metrics
                from model_runs
                order by model_name, created_at desc
                """
            )
        )
        rows = result.all()

    experts = []
    for r in rows:
        metrics = r.metrics or {}
        experts.append(
            {
                "model_name": r.model_name,
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "logloss": metrics.get("logloss"),
                "conformal_q_hat": metrics.get("conformal_q_hat"),
                "train_rows": r.train_rows,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )
    return {"experts": experts}


CANONICAL_VIEW = "vw_resolved_picks_canonical"

_CANONICAL_SQL = f"""
select
    id::text as prediction_id,
    decision_time, created_at,
    over_label, outcome, is_correct,
    p_final, p_raw,
    p_forecast_cal, p_nn, p_tabdl, p_lr, p_xgb, p_lgbm,
    cast(details->>'conformal_set_size' as integer) as conformal_set_size,
    cast(details->>'selection_threshold' as float) as selection_threshold,
    stat_type, rank_score, n_eff
from {CANONICAL_VIEW}
order by coalesce(decision_time, created_at) asc, created_at asc, id asc
"""

_FALLBACK_SQL = """
select
    id::text as prediction_id,
    coalesce(decision_time, created_at) as decision_time,
    created_at,
    over_label, outcome, is_correct,
    prob_over as p_final, p_raw,
    p_forecast_cal, p_nn,
    coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl,
    cast(details->>'conformal_set_size' as integer) as conformal_set_size,
    cast(details->>'selection_threshold' as float) as selection_threshold,
    p_lr, p_xgb, p_lgbm,
    stat_type, rank_score, n_eff
from projection_predictions
order by coalesce(decision_time, created_at) asc, created_at asc, id asc
"""


def _load_prediction_records_sync() -> pd.DataFrame | None:
    try:
        engine = get_engine()
        try:
            df = pd.read_sql(text(_CANONICAL_SQL), engine)
        except Exception:  # noqa: BLE001
            df = pd.read_sql(text(_FALLBACK_SQL), engine)
        if not df.empty:
            return df
    except Exception:  # noqa: BLE001
        pass
    return None


def _load_model_health_sync(*, days_back: int, min_alert_weight: float) -> dict:
    from scripts.ops.monitor_model_health import build_health_report

    engine = get_engine()
    return build_health_report(
        engine,
        days_back=int(max(1, days_back)),
        ensemble_weights_path=str(ENSEMBLE_WEIGHTS_PATH),
        min_alert_weight=float(max(0.0, min(1.0, min_alert_weight))),
    )


def _selection_thresholds_for_rows(scored: pd.DataFrame) -> pd.Series:
    if scored.empty:
        return pd.Series(dtype=float)
    policy = SelectionPolicy.load()
    stat_types = scored.get("stat_type", pd.Series([""] * len(scored))).astype(str).tolist()
    conformal = pd.to_numeric(scored.get("conformal_set_size"), errors="coerce")
    thresholds = [
        float(
            policy.threshold_for(
                stat_type,
                int(conf) if conf is not None and pd.notna(conf) else None,
            )
        )
        for stat_type, conf in zip(stat_types, conformal.tolist())
    ]
    explicit = pd.to_numeric(scored.get("selection_threshold"), errors="coerce")
    if explicit is not None and len(explicit) == len(thresholds):
        thresholds = [
            float(exp) if exp is not None and pd.notna(exp) else float(th)
            for th, exp in zip(thresholds, explicit.tolist())
        ]
    return pd.Series(thresholds, index=scored.index, dtype=float)


@router.get("/hit-rate")
async def hit_rate(window: int = Query(50, ge=5, le=500)) -> dict:
    df = await asyncio.to_thread(_load_prediction_records_sync)
    if df is None:
        return {
            "total_predictions": 0,
            "total_resolved": 0,
            "total_scored": 0,
            "overall_hit_rate": None,
            "rolling": [],
        }

    if "over_label" not in df.columns:
        return {
            "total_predictions": len(df),
            "total_resolved": 0,
            "total_scored": 0,
            "overall_hit_rate": None,
            "rolling": [],
        }

    df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
    total_predictions = len(df)

    if "outcome" in df.columns:
        resolved = df[df["outcome"].isin(["over", "under"])].copy()
    else:
        resolved = df.copy()
    resolved = resolved.dropna(subset=["over_label"]).copy()

    # p_final threshold: pick OVER if p_final >= 0.5
    if "p_final" in df.columns:
        resolved["p_final"] = pd.to_numeric(resolved["p_final"], errors="coerce")
        resolved["ensemble_correct"] = (
            ((resolved["p_final"] >= 0.5) & (resolved["over_label"] == 1))
            | ((resolved["p_final"] < 0.5) & (resolved["over_label"] == 0))
        ).astype(float)
        resolved.loc[resolved["p_final"].isna(), "ensemble_correct"] = float("nan")
        if "is_correct" in resolved.columns:
            resolved["is_correct"] = pd.to_numeric(
                resolved["is_correct"], errors="coerce"
            )
            mask = resolved["is_correct"].notna() & resolved["p_final"].notna()
            resolved.loc[mask, "ensemble_correct"] = resolved.loc[
                mask, "is_correct"
            ].astype(float)
    else:
        resolved["ensemble_correct"] = float("nan")

    for col in EXPERT_COLS:
        if col in resolved.columns:
            resolved[col] = pd.to_numeric(resolved[col], errors="coerce")
            resolved[f"{col}_correct"] = (
                ((resolved[col] >= 0.5) & (resolved["over_label"] == 1))
                | ((resolved[col] < 0.5) & (resolved["over_label"] == 0))
            ).astype(float)
            resolved.loc[resolved[col].isna(), f"{col}_correct"] = float("nan")
        else:
            resolved[f"{col}_correct"] = float("nan")
    total_resolved = len(resolved)
    overall_hit_rate = (
        float(resolved["ensemble_correct"].mean())
        if total_resolved > 0 and "ensemble_correct" in resolved.columns
        else None
    )

    scored = resolved.dropna(subset=["ensemble_correct"]).copy()
    if len(scored) < 2:
        return {
            "total_predictions": total_predictions,
            "total_resolved": total_resolved,
            "total_scored": int(len(scored)),
            "overall_hit_rate": overall_hit_rate,
            "rolling": [],
        }

    scored = scored.reset_index(drop=True)

    rolling_data = []
    ens_rolling = (
        scored["ensemble_correct"]
        .rolling(window, min_periods=max(5, window // 4))
        .mean()
    )

    expert_rolling = {}
    for col in EXPERT_COLS:
        c = f"{col}_correct"
        if c in scored.columns:
            expert_rolling[col] = (
                scored[c].rolling(window, min_periods=max(5, window // 4)).mean()
            )

    date_col = None
    for candidate in ["decision_time", "created_at"]:
        if candidate in scored.columns:
            date_col = candidate
            break

    for i in range(len(scored)):
        if pd.isna(ens_rolling.iloc[i]):
            continue
        point: dict = {
            "index": i,
            "ensemble_hit_rate": _safe_float(ens_rolling.iloc[i]),
        }
        for col in EXPERT_COLS:
            val = expert_rolling.get(col)
            point[f"{col}_hit_rate"] = (
                _safe_float(val.iloc[i]) if val is not None else None
            )
        if date_col:
            point["date"] = (
                str(scored[date_col].iloc[i])
                if pd.notna(scored[date_col].iloc[i])
                else None
            )
        rolling_data.append(point)

    # Downsample if too many points
    if len(rolling_data) > 200:
        step = max(1, len(rolling_data) // 200)
        rolling_data = rolling_data[::step]

    # Tier metrics: scored vs actionable
    import numpy as _np

    p_arr = (
        scored["p_final"].to_numpy(dtype=float) if "p_final" in scored.columns else None
    )
    published_hit_rate = None
    published_n = 0
    coverage = 0.0
    actionable_threshold = None
    if p_arr is not None and len(p_arr) > 0:
        p_pick = _np.maximum(p_arr, 1.0 - p_arr)
        threshold_series = _selection_thresholds_for_rows(scored)
        if threshold_series.empty:
            threshold_series = pd.Series([0.60] * len(scored), index=scored.index, dtype=float)
        actionable_threshold = _safe_float(float(threshold_series.mean()))
        act_mask = p_pick >= threshold_series.to_numpy(dtype=float)
        published_n = int(act_mask.sum())
        if published_n > 0:
            published_hit_rate = _safe_float(
                float(scored.loc[act_mask, "ensemble_correct"].mean())
            )
        coverage = round(published_n / len(scored), 4)

    # Placed tier: actionable AND conformal_set_size != 2 (tightest filter)
    placed_hit_rate = None
    placed_n = 0
    if p_arr is not None and len(p_arr) > 0:
        placed_mask = act_mask.copy()
        if "conformal_set_size" in scored.columns:
            conf_ok = scored["conformal_set_size"].fillna(1).to_numpy() != 2
            placed_mask = placed_mask & conf_ok
        placed_n = int(placed_mask.sum())
        if placed_n > 0:
            placed_hit_rate = _safe_float(
                float(scored.loc[placed_mask, "ensemble_correct"].mean())
            )

    return {
        "total_predictions": total_predictions,
        "total_resolved": total_resolved,
        "total_scored": int(len(scored)),
        "overall_hit_rate": _safe_float(overall_hit_rate),
        "published_hit_rate": published_hit_rate,
        "published_n": published_n,
        "placed_hit_rate": placed_hit_rate,
        "placed_n": placed_n,
        "coverage": coverage,
        "actionable_threshold": actionable_threshold,
        "rolling": rolling_data,
    }


@router.get("/model-health")
async def model_health(
    days_back: int = Query(90, ge=7, le=365),
    min_alert_weight: float = Query(0.03, ge=0.0, le=1.0),
) -> dict:
    """Return live model-health metrics computed directly from DB state."""
    try:
        return await asyncio.to_thread(
            _load_model_health_sync,
            days_back=days_back,
            min_alert_weight=min_alert_weight,
        )
    except Exception as exc:
        return {"error": str(exc), "experts": [], "alerts": []}


@router.get("/calibration")
async def calibration() -> dict:
    if not CALIBRATION_DIR.exists():
        return {"stat_types": []}

    report_files = sorted(
        CALIBRATION_DIR.glob("*_report*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not report_files:
        return {"stat_types": []}

    df = pd.read_csv(report_files[0])
    if df.empty:
        return {"stat_types": []}

    result = []
    for _, row in df.iterrows():
        result.append(
            {
                "stat_type": str(row.get("stat_type", "")),
                "train_rows": (
                    int(row["train_rows"]) if pd.notna(row.get("train_rows")) else None
                ),
                "val_rows": (
                    int(row["val_rows"]) if pd.notna(row.get("val_rows")) else None
                ),
                "nll_before": _safe_float(row.get("nll_before")),
                "nll_after": _safe_float(row.get("nll_after_param")),
                "crps_before": _safe_float(row.get("crps_before")),
                "crps_after": _safe_float(row.get("crps_after_param")),
                "cov90_before": _safe_float(row.get("cov90_before")),
                "cov90_after": _safe_float(row.get("cov90_after_param")),
                "pit_ks_before": _safe_float(row.get("pit_ks_before")),
                "pit_ks_after": _safe_float(row.get("pit_ks_after_param")),
            }
        )
    return {"stat_types": result}


@router.get("/ensemble-weights")
async def ensemble_weights() -> dict:
    if not ENSEMBLE_WEIGHTS_PATH.exists():
        return {"experts": [], "contexts": []}

    state = json.loads(ENSEMBLE_WEIGHTS_PATH.read_text(encoding="utf-8"))
    experts = state.get("experts", [])
    weights_map = state.get("weights", {})

    contexts = []
    for ctx_key_str, weights in weights_map.items():
        try:
            parts = json.loads(ctx_key_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parts, list) or len(parts) != 3:
            continue
        contexts.append(
            {
                "context_key": ctx_key_str,
                "stat_type": parts[0],
                "regime": parts[1],
                "neff_bucket": parts[2],
                "weights": {e: _safe_float(weights.get(e, 0.0)) for e in experts},
            }
        )

    # Sort by stat_type for readability
    contexts.sort(key=lambda c: (c["stat_type"], c["regime"], c["neff_bucket"]))
    return {"experts": experts, "contexts": contexts}


@router.get("/confidence-dist")
async def confidence_dist(bins: int = Query(20, ge=5, le=50)) -> dict:
    df = await asyncio.to_thread(_load_prediction_records_sync)
    if df is None:
        return {"bins": []}

    if "p_final" not in df.columns:
        return {"bins": []}

    df["p_final"] = pd.to_numeric(df["p_final"], errors="coerce")
    df = df.dropna(subset=["p_final"])
    if df.empty:
        return {"bins": []}

    # Use confidence (distance from 0.5) for the histogram
    df["confidence"] = df["p_final"].apply(lambda p: max(p, 1.0 - p))

    has_outcomes = "over_label" in df.columns
    if has_outcomes:
        df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
        if "outcome" in df.columns:
            df = df[df["outcome"].isin(["over", "under"]) | df["outcome"].isna()].copy()
        df["hit"] = ((df["p_final"] >= 0.5) & (df["over_label"] == 1)) | (
            (df["p_final"] < 0.5) & (df["over_label"] == 0)
        )
        if "is_correct" in df.columns:
            df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
            df.loc[df["is_correct"].notna(), "hit"] = df.loc[
                df["is_correct"].notna(), "is_correct"
            ].astype(bool)

    bin_edges = [0.5 + i * (0.5 / bins) for i in range(bins + 1)]
    result = []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (df["confidence"] >= lo) & (
            df["confidence"] < hi if i < bins - 1 else df["confidence"] <= hi
        )
        subset = df[mask]
        entry: dict = {
            "range_start": round(lo, 4),
            "range_end": round(hi, 4),
            "count": int(len(subset)),
        }
        if has_outcomes:
            resolved = subset.dropna(subset=["over_label"])
            entry["hits"] = int(resolved["hit"].sum()) if not resolved.empty else 0
            entry["misses"] = (
                int(len(resolved) - resolved["hit"].sum()) if not resolved.empty else 0
            )
            entry["hit_rate"] = (
                _safe_float(resolved["hit"].mean()) if not resolved.empty else None
            )
        else:
            entry["hits"] = None
            entry["misses"] = None
            entry["hit_rate"] = None
        result.append(entry)

    return {"bins": result}


@router.get("/per-stat-performance")
async def per_stat_performance(days_back: int = Query(120, ge=7, le=365)) -> dict:
    df = await asyncio.to_thread(_load_prediction_records_sync)
    if df is None or df.empty or "p_final" not in df.columns or "over_label" not in df.columns:
        return {"days_back": days_back, "stats": []}

    frame = df.copy()
    for col in ["decision_time", "created_at"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    if "decision_time" in frame.columns:
        recent_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=int(days_back))
        frame = frame[frame["decision_time"].fillna(frame.get("created_at")) >= recent_cutoff]
    if frame.empty:
        return {"days_back": days_back, "stats": []}

    frame["p_final"] = pd.to_numeric(frame["p_final"], errors="coerce")
    frame["over_label"] = pd.to_numeric(frame["over_label"], errors="coerce")
    frame = frame.dropna(subset=["p_final", "over_label", "stat_type"]).copy()
    if frame.empty:
        return {"days_back": days_back, "stats": []}

    frame["pick"] = (frame["p_final"] >= 0.5).astype(int)
    frame["correct"] = (frame["pick"] == frame["over_label"].astype(int)).astype(float)
    frame["p_pick"] = frame["p_final"].apply(lambda p: max(float(p), 1.0 - float(p)))
    frame["selection_threshold"] = _selection_thresholds_for_rows(frame)
    frame["is_actionable"] = frame["p_pick"] >= frame["selection_threshold"]
    if "conformal_set_size" in frame.columns:
        conf_ok = pd.to_numeric(frame["conformal_set_size"], errors="coerce").fillna(1).to_numpy() != 2
        frame["is_placed"] = frame["is_actionable"] & conf_ok
    else:
        frame["is_placed"] = frame["is_actionable"]

    rows: list[dict] = []
    for stat_type, grp in frame.groupby("stat_type"):
        actionable = grp[grp["is_actionable"]]
        placed = grp[grp["is_placed"]]
        rows.append(
            {
                "stat_type": str(stat_type),
                "n_scored": int(len(grp)),
                "accuracy_scored": _safe_float(float(grp["correct"].mean())),
                "n_actionable": int(len(actionable)),
                "accuracy_actionable": _safe_float(float(actionable["correct"].mean())) if len(actionable) > 0 else None,
                "actionable_coverage": _safe_float(float(len(actionable) / len(grp))) if len(grp) > 0 else 0.0,
                "n_placed": int(len(placed)),
                "accuracy_placed": _safe_float(float(placed["correct"].mean())) if len(placed) > 0 else None,
                "placed_coverage": _safe_float(float(len(placed) / len(grp))) if len(grp) > 0 else 0.0,
                "mean_p_pick": _safe_float(float(grp["p_pick"].mean())),
                "mean_threshold": _safe_float(float(grp["selection_threshold"].mean())),
            }
        )

    rows.sort(key=lambda x: (x.get("n_actionable", 0), x.get("n_scored", 0)), reverse=True)
    return {"days_back": days_back, "stats": rows}


@router.get("/coverage-frontier")
async def coverage_frontier(days_back: int = Query(120, ge=7, le=365)) -> dict:
    df = await asyncio.to_thread(_load_prediction_records_sync)
    if df is None or df.empty or "p_final" not in df.columns or "over_label" not in df.columns:
        return {"days_back": days_back, "points": []}

    frame = df.copy()
    for col in ["decision_time", "created_at"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    if "decision_time" in frame.columns:
        recent_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=int(days_back))
        frame = frame[frame["decision_time"].fillna(frame.get("created_at")) >= recent_cutoff]
    if frame.empty:
        return {"days_back": days_back, "points": []}

    frame["p_final"] = pd.to_numeric(frame["p_final"], errors="coerce")
    frame["over_label"] = pd.to_numeric(frame["over_label"], errors="coerce")
    frame = frame.dropna(subset=["p_final", "over_label"]).copy()
    if frame.empty:
        return {"days_back": days_back, "points": []}

    probs = frame["p_final"].to_numpy(dtype=float)
    labels = frame["over_label"].to_numpy(dtype=int)
    picks = (probs >= 0.5).astype(int)
    correct = (picks == labels).astype(float)
    p_pick = pd.Series(np.maximum(probs, 1.0 - probs), dtype=float)
    conformal = pd.to_numeric(frame.get("conformal_set_size"), errors="coerce").fillna(1).to_numpy()
    n_total = len(frame)

    points: list[dict] = []
    for threshold in np.arange(0.50, 0.76, 0.01):
        threshold = round(float(threshold), 2)
        actionable = p_pick.to_numpy(dtype=float) >= threshold
        n_actionable = int(actionable.sum())
        if n_actionable == 0:
            points.append(
                {
                    "threshold": threshold,
                    "n_actionable": 0,
                    "coverage": 0.0,
                    "accuracy_actionable": None,
                    "n_placed": 0,
                    "placed_coverage": 0.0,
                    "accuracy_placed": None,
                }
            )
            continue
        placed_mask = actionable & (conformal != 2)
        n_placed = int(placed_mask.sum())
        points.append(
            {
                "threshold": threshold,
                "n_actionable": n_actionable,
                "coverage": _safe_float(float(n_actionable / n_total)),
                "accuracy_actionable": _safe_float(float(correct[actionable].mean())),
                "n_placed": n_placed,
                "placed_coverage": _safe_float(float(n_placed / n_total)),
                "accuracy_placed": _safe_float(float(correct[placed_mask].mean())) if n_placed > 0 else None,
            }
        )

    return {"days_back": days_back, "points": points}


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


WEIGHT_HISTORY_PATH = Path("data/reports/weight_history.jsonl")
DRIFT_REPORT_PATH = Path("data/reports/drift_report.json")


@router.get("/weight-history")
async def weight_history(limit: int = Query(100, ge=1, le=1000)) -> dict:
    """Return ensemble weight evolution over time."""
    if not WEIGHT_HISTORY_PATH.exists():
        return {"entries": []}

    entries = []
    for line in WEIGHT_HISTORY_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Return most recent entries
    return {"entries": entries[-limit:]}


@router.get("/drift-report")
async def drift_report() -> dict:
    """Return latest drift detection results."""
    if not DRIFT_REPORT_PATH.exists():
        return {"checks": [], "any_drift": False}

    try:
        data = json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError:
        return {"checks": [], "any_drift": False}


@router.get("/mixing-weights")
async def mixing_weights() -> dict:
    """Return current hybrid ensemble mixing weights."""
    entries = []
    if WEIGHT_HISTORY_PATH.exists():
        for line in WEIGHT_HISTORY_PATH.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return {"mixing": {"thompson": 0.34, "gating": 0.33, "meta_learner": 0.33}}

    latest = entries[-1]
    return {
        "mixing": latest.get(
            "mixing", {"thompson": 0.34, "gating": 0.33, "meta_learner": 0.33}
        ),
        "hedge_avg": latest.get("hedge_avg", {}),
        "thompson_avg": latest.get("thompson_avg", {}),
        "timestamp": latest.get("timestamp"),
        "n_updates": latest.get("n_updates", 0),
    }
