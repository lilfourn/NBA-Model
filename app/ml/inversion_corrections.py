"""Load and apply expert inversion corrections from model health report.

When an expert is consistently inverted (1-p would be more accurate than p),
the scoring pipeline flips its probability before passing to the ensemble.

Usage:
    flags = load_inversion_flags()          # from local model_health.json
    flags = load_inversion_flags(engine)    # from DB artifact

    for expert, should_flip in flags.items():
        if should_flip and expert_probs.get(expert) is not None:
            expert_probs[expert] = 1.0 - expert_probs[expert]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

# Default local path for model_health.json
_DEFAULT_HEALTH_PATH = Path(
    os.environ.get("HEALTH_REPORT_PATH", "data/reports/model_health.json")
)

# Experts that should never be flipped (p_final is derived, not an input)
_SKIP_EXPERTS = {"p_final"}
MIN_INVERSION_SAMPLE = int(os.environ.get("INVERSION_MIN_SAMPLE", "300"))
MIN_INVERSION_ACC_GAIN = float(os.environ.get("INVERSION_MIN_ACC_GAIN", "0.03"))
MIN_INVERSION_LOGLOSS_GAIN = float(
    os.environ.get("INVERSION_MIN_LOGLOSS_GAIN", "0.02")
)
DEFAULT_FORECAST_STAT_DAYS_BACK = int(
    os.environ.get("FORECAST_STAT_INVERSION_DAYS_BACK", "180")
)
MIN_FORECAST_STAT_SAMPLE = int(
    os.environ.get("FORECAST_STAT_INVERSION_MIN_SAMPLE", "250")
)
MIN_FORECAST_STAT_ACC_GAIN = float(
    os.environ.get("FORECAST_STAT_INVERSION_MIN_ACC_GAIN", "0.05")
)
MIN_FORECAST_STAT_LOGLOSS_GAIN = float(
    os.environ.get("FORECAST_STAT_INVERSION_MIN_LOGLOSS_GAIN", "0.02")
)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _passes_inversion_gate(
    *,
    n: int,
    accuracy: float | None,
    accuracy_inverted: float | None,
    logloss: float | None,
    logloss_inverted: float | None,
    min_n: int,
    min_accuracy_gain: float,
    min_logloss_gain: float,
) -> bool:
    if int(n) < int(min_n):
        return False
    if (
        accuracy is None
        or accuracy_inverted is None
        or logloss is None
        or logloss_inverted is None
    ):
        return False
    acc_gain = float(accuracy_inverted - accuracy)
    ll_gain = float(logloss - logloss_inverted)
    return acc_gain >= float(min_accuracy_gain) and ll_gain >= float(min_logloss_gain)


def _binary_logloss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(labels, dtype=float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def load_inversion_flags(
    engine_or_path: Any = None,
    *,
    min_n: int = MIN_INVERSION_SAMPLE,
    min_accuracy_gain: float = MIN_INVERSION_ACC_GAIN,
    min_logloss_gain: float = MIN_INVERSION_LOGLOSS_GAIN,
) -> dict[str, bool]:
    """Load inversion flags from model health report.

    Returns dict mapping expert name -> True if the expert should be inverted.
    Uses strict gates to avoid noise-driven flips from very small windows.

    Args:
        engine_or_path: One of:
            - None: load from default local path
            - str/Path: load from explicit file path
            - SQLAlchemy Engine: load from DB artifact store
    """
    report = _load_report(engine_or_path)
    if not report:
        return {}

    expert_metrics = report.get("expert_metrics", {})
    flags: dict[str, bool] = {}

    for expert, data in expert_metrics.items():
        if expert in _SKIP_EXPERTS:
            continue
        if not isinstance(data, dict):
            continue
        inv = data.get("inversion_test", {})
        if not isinstance(inv, dict):
            continue
        n = int(_safe_float(data.get("n")) or _safe_float(inv.get("n")) or 0)
        accuracy = _safe_float(inv.get("accuracy"))
        accuracy_inverted = _safe_float(inv.get("accuracy_inverted"))
        logloss = _safe_float(inv.get("logloss"))
        logloss_inverted = _safe_float(inv.get("logloss_inverted"))
        if _passes_inversion_gate(
            n=n,
            accuracy=accuracy,
            accuracy_inverted=accuracy_inverted,
            logloss=logloss,
            logloss_inverted=logloss_inverted,
            min_n=min_n,
            min_accuracy_gain=min_accuracy_gain,
            min_logloss_gain=min_logloss_gain,
        ):
            flags[expert] = True

    return flags


def load_forecast_stat_inversion_flags(
    engine_or_path: Any = None,
    *,
    days_back: int = DEFAULT_FORECAST_STAT_DAYS_BACK,
    min_samples: int = MIN_FORECAST_STAT_SAMPLE,
    min_accuracy_gain: float = MIN_FORECAST_STAT_ACC_GAIN,
    min_logloss_gain: float = MIN_FORECAST_STAT_LOGLOSS_GAIN,
) -> dict[str, bool]:
    """Load per-stat inversion flags for forecast probabilities.

    For stat types with strong and stable evidence that ``1 - p_forecast_cal``
    outperforms ``p_forecast_cal``, return ``{stat_type: True}``.
    """
    # Preferred source: DB resolved rows.
    if engine_or_path is not None and hasattr(engine_or_path, "connect"):
        try:
            query = text(
                """
                select
                    stat_type,
                    over_label,
                    p_forecast_cal
                from projection_predictions
                where outcome in ('over', 'under')
                  and over_label is not null
                  and actual_value is not null
                  and p_forecast_cal is not null
                  and coalesce(decision_time, created_at) >= now() - (:days * interval '1 day')
                """
            )
            df = pd.read_sql(query, engine_or_path, params={"days": int(max(1, days_back))})
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()
        if not df.empty:
            df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
            df["p_forecast_cal"] = pd.to_numeric(df["p_forecast_cal"], errors="coerce")
            df["stat_type"] = df["stat_type"].astype(str)
            df = df.dropna(subset=["stat_type", "over_label", "p_forecast_cal"])
            if not df.empty:
                flags: dict[str, bool] = {}
                for stat_type, group in df.groupby("stat_type"):
                    n = int(len(group))
                    if n < int(min_samples):
                        continue
                    p = group["p_forecast_cal"].to_numpy(dtype=float)
                    y = group["over_label"].to_numpy(dtype=float)
                    acc = float(((p >= 0.5).astype(int) == y).mean())
                    acc_inv = float((((1.0 - p) >= 0.5).astype(int) == y).mean())
                    ll = _binary_logloss(p, y)
                    ll_inv = _binary_logloss(1.0 - p, y)
                    if _passes_inversion_gate(
                        n=n,
                        accuracy=acc,
                        accuracy_inverted=acc_inv,
                        logloss=ll,
                        logloss_inverted=ll_inv,
                        min_n=int(min_samples),
                        min_accuracy_gain=float(min_accuracy_gain),
                        min_logloss_gain=float(min_logloss_gain),
                    ):
                        flags[str(stat_type)] = True
                return flags

    # Fallback source: health report if available.
    report = _load_report(engine_or_path)
    if not report:
        return {}
    payload = report.get("forecast_stat_inversion_flags") or {}
    if isinstance(payload, dict):
        return {str(stat): bool(flag) for stat, flag in payload.items() if bool(flag)}
    return {}


def _load_report(engine_or_path: Any) -> dict | None:
    """Load model_health.json from local file, explicit path, or DB."""
    # Explicit path
    if isinstance(engine_or_path, (str, Path)):
        return _load_from_file(Path(engine_or_path))

    # SQLAlchemy engine — try DB artifact
    if engine_or_path is not None and hasattr(engine_or_path, "connect"):
        try:
            from app.ml.artifact_store import load_latest_artifact

            data = load_latest_artifact(engine_or_path, "model_health")
            if data:
                return json.loads(data.decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
        # Fall through to local file

    # Default local file
    return _load_from_file(_DEFAULT_HEALTH_PATH)


def _load_from_file(path: Path) -> dict | None:
    """Load JSON report from a local file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
