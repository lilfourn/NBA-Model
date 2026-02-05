from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query
from sqlalchemy import text

from app.db.engine import get_async_engine, get_engine

router = APIRouter(prefix="/api/stats", tags=["stats"])

ENSEMBLE_WEIGHTS_PATH = Path("models/ensemble_weights.json")
CALIBRATION_DIR = Path("data/calibration")
EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_lr", "p_xgb", "p_lgbm"]


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


def _load_prediction_records_sync() -> pd.DataFrame | None:
    try:
        engine = get_engine()
        df = pd.read_sql(
            text(
                """
                select
                    coalesce(decision_time, created_at) as decision_time,
                    created_at,
                    over_label,
                    prob_over as p_final,
                    p_forecast_cal,
                    p_nn,
                    p_lr,
                    p_xgb,
                    p_lgbm
                from projection_predictions
                order by coalesce(decision_time, created_at) asc
                """
            ),
            engine,
        )
        if not df.empty:
            return df
    except Exception:  # noqa: BLE001
        pass
    return None


@router.get("/hit-rate")
async def hit_rate(window: int = Query(50, ge=5, le=500)) -> dict:
    df = await asyncio.to_thread(_load_prediction_records_sync)
    if df is None:
        return {"total_predictions": 0, "total_resolved": 0, "overall_hit_rate": None, "rolling": []}

    if "over_label" not in df.columns:
        return {"total_predictions": len(df), "total_resolved": 0, "overall_hit_rate": None, "rolling": []}

    df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
    total_predictions = len(df)

    # p_final threshold: pick OVER if p_final >= 0.5
    if "p_final" in df.columns:
        df["p_final"] = pd.to_numeric(df["p_final"], errors="coerce")
        df["ensemble_correct"] = (
            ((df["p_final"] >= 0.5) & (df["over_label"] == 1))
            | ((df["p_final"] < 0.5) & (df["over_label"] == 0))
        ).astype(float)
        df.loc[df["over_label"].isna() | df["p_final"].isna(), "ensemble_correct"] = float("nan")
    else:
        df["ensemble_correct"] = float("nan")

    for col in EXPERT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_correct"] = (
                ((df[col] >= 0.5) & (df["over_label"] == 1))
                | ((df[col] < 0.5) & (df["over_label"] == 0))
            ).astype(float)
            df.loc[df["over_label"].isna() | df[col].isna(), f"{col}_correct"] = float("nan")
        else:
            df[f"{col}_correct"] = float("nan")

    resolved = df.dropna(subset=["over_label"])
    total_resolved = len(resolved)
    overall_hit_rate = float(resolved["ensemble_correct"].mean()) if total_resolved > 0 and "ensemble_correct" in resolved.columns else None

    if total_resolved < 2:
        return {
            "total_predictions": total_predictions,
            "total_resolved": total_resolved,
            "overall_hit_rate": overall_hit_rate,
            "rolling": [],
        }

    resolved = resolved.reset_index(drop=True)

    rolling_data = []
    ens_rolling = resolved["ensemble_correct"].rolling(window, min_periods=max(5, window // 4)).mean()

    expert_rolling = {}
    for col in EXPERT_COLS:
        c = f"{col}_correct"
        if c in resolved.columns:
            expert_rolling[col] = resolved[c].rolling(window, min_periods=max(5, window // 4)).mean()

    date_col = None
    for candidate in ["decision_time", "created_at"]:
        if candidate in resolved.columns:
            date_col = candidate
            break

    for i in range(len(resolved)):
        if pd.isna(ens_rolling.iloc[i]):
            continue
        point: dict = {"index": i, "ensemble_hit_rate": _safe_float(ens_rolling.iloc[i])}
        for col in EXPERT_COLS:
            val = expert_rolling.get(col)
            point[f"{col}_hit_rate"] = _safe_float(val.iloc[i]) if val is not None else None
        if date_col:
            point["date"] = str(resolved[date_col].iloc[i]) if pd.notna(resolved[date_col].iloc[i]) else None
        rolling_data.append(point)

    # Downsample if too many points
    if len(rolling_data) > 200:
        step = max(1, len(rolling_data) // 200)
        rolling_data = rolling_data[::step]

    return {
        "total_predictions": total_predictions,
        "total_resolved": total_resolved,
        "overall_hit_rate": _safe_float(overall_hit_rate),
        "rolling": rolling_data,
    }


@router.get("/calibration")
async def calibration() -> dict:
    if not CALIBRATION_DIR.exists():
        return {"stat_types": []}

    report_files = sorted(CALIBRATION_DIR.glob("*_report*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
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
                "train_rows": int(row["train_rows"]) if pd.notna(row.get("train_rows")) else None,
                "val_rows": int(row["val_rows"]) if pd.notna(row.get("val_rows")) else None,
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
        df["hit"] = (
            ((df["p_final"] >= 0.5) & (df["over_label"] == 1))
            | ((df["p_final"] < 0.5) & (df["over_label"] == 0))
        )

    bin_edges = [0.5 + i * (0.5 / bins) for i in range(bins + 1)]
    result = []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi if i < bins - 1 else df["confidence"] <= hi)
        subset = df[mask]
        entry: dict = {
            "range_start": round(lo, 4),
            "range_end": round(hi, 4),
            "count": int(len(subset)),
        }
        if has_outcomes:
            resolved = subset.dropna(subset=["over_label"])
            entry["hits"] = int(resolved["hit"].sum()) if not resolved.empty else 0
            entry["misses"] = int(len(resolved) - resolved["hit"].sum()) if not resolved.empty else 0
            entry["hit_rate"] = _safe_float(resolved["hit"].mean()) if not resolved.empty else None
        else:
            entry["hits"] = None
            entry["misses"] = None
            entry["hit_rate"] = None
        result.append(entry)

    return {"bins": result}


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
