"""Per-stat-type isotonic recalibration of ensemble probabilities.

Fits sklearn IsotonicRegression per stat type on recent resolved predictions.
Only calibrates stat types with sufficient sample size; others pass through.

Usage:
    from app.ml.stat_calibrator import StatTypeCalibrator

    cal = StatTypeCalibrator.fit_from_db(engine, days_back=45)
    cal.save()
    # ... later at inference time ...
    cal = StatTypeCalibrator.load()
    p_calibrated = cal.transform(p_final=0.62, stat_type="Points")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import text

MIN_SAMPLES_PER_STAT = 100  # Minimum resolved rows before we calibrate
CALIBRATOR_PATH = (
    Path(os.environ.get("MODELS_DIR", "models")) / "stat_calibrator.joblib"
)
CALIBRATOR_VERSION = "2.1.0"

# Stat types excluded from calibration (degenerate base rates)
EXCLUDED_STAT_TYPES: set[str] = {
    "Dunks",
    "Blocked Shots",
    "Blks+Stls",
    "Offensive Rebounds",
    "Personal Fouls",
    "Steals",
}


class StatTypeCalibrator:
    """Isotonic recalibration per stat type."""

    def __init__(
        self,
        calibrators: dict[str, IsotonicRegression] | None = None,
        global_calibrator: IsotonicRegression | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.calibrators: dict[str, IsotonicRegression] = calibrators or {}
        self.global_calibrator: IsotonicRegression | None = global_calibrator
        self.meta: dict[str, Any] = meta or {}

    def transform(self, p_final: float, stat_type: str) -> float:
        """Apply isotonic recalibration. Falls back to global or uncalibrated."""
        cal = self.calibrators.get(stat_type)
        if cal is None:
            cal = self.global_calibrator
        if cal is None:
            return p_final
        try:
            result = float(cal.predict([p_final])[0])
            # Clamp to avoid extreme values
            return max(0.01, min(0.99, result))
        except Exception:  # noqa: BLE001
            return p_final

    def save(self, path: Path | str | None = None) -> Path:
        out = Path(path or CALIBRATOR_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibrators": self.calibrators,
            "global_calibrator": self.global_calibrator,
            "meta": self.meta,
            "version": CALIBRATOR_VERSION,
        }
        joblib.dump(payload, out)
        return out

    @classmethod
    def load(cls, path: Path | str | None = None) -> StatTypeCalibrator:
        p = Path(path or CALIBRATOR_PATH)
        if not p.exists():
            return cls()  # Empty calibrator (pass-through)
        try:
            payload = joblib.load(p)
            meta = payload.get("meta", {}) or {}
            if "calibrator_version" not in meta:
                meta["calibrator_version"] = str(payload.get("version") or "legacy")
            return cls(
                calibrators=payload.get("calibrators", {}),
                global_calibrator=payload.get("global_calibrator"),
                meta=meta,
            )
        except Exception:  # noqa: BLE001
            return cls()

    @classmethod
    def fit_from_dataframe(cls, df: pd.DataFrame) -> StatTypeCalibrator:
        """Fit calibrators from a DataFrame with p_final, over_label, stat_type columns."""
        required = {"p_final", "over_label", "stat_type"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame missing columns: {required - set(df.columns)}")

        valid = df.dropna(subset=["p_final", "over_label"]).copy()
        valid = valid[~valid["stat_type"].isin(EXCLUDED_STAT_TYPES)]
        if valid.empty:
            return cls()

        probs = valid["p_final"].to_numpy(dtype=float)
        labels = valid["over_label"].to_numpy(dtype=int)

        # Fit global calibrator
        global_cal = None
        if len(valid) >= MIN_SAMPLES_PER_STAT:
            global_cal = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip"
            )
            global_cal.fit(probs, labels)

        # Fit per-stat-type calibrators
        calibrators: dict[str, IsotonicRegression] = {}
        stat_meta: dict[str, dict[str, Any]] = {}
        for st, group in valid.groupby("stat_type"):
            st = str(st)
            n = len(group)
            if n < MIN_SAMPLES_PER_STAT:
                stat_meta[st] = {
                    "n": n,
                    "calibrated": False,
                    "reason": "insufficient_samples",
                }
                continue
            st_probs = group["p_final"].to_numpy(dtype=float)
            st_labels = group["over_label"].to_numpy(dtype=int)
            # Need at least 2 unique labels
            if len(np.unique(st_labels)) < 2:
                stat_meta[st] = {"n": n, "calibrated": False, "reason": "single_class"}
                continue
            cal = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            cal.fit(st_probs, st_labels)
            calibrators[st] = cal
            stat_meta[st] = {"n": n, "calibrated": True}

        meta = {
            "calibrator_version": CALIBRATOR_VERSION,
            "total_rows": len(valid),
            "n_stat_types_calibrated": len(calibrators),
            "global_calibrated": global_cal is not None,
            "stat_types": stat_meta,
            "min_samples_per_stat": MIN_SAMPLES_PER_STAT,
        }
        return cls(calibrators=calibrators, global_calibrator=global_cal, meta=meta)

    @classmethod
    def fit_from_db(cls, engine, *, days_back: int = 45) -> StatTypeCalibrator:
        """Fit calibrators from resolved predictions in the database."""
        # Try canonical view first, fallback to raw table
        for table in ("vw_resolved_picks_canonical", "projection_predictions"):
            try:
                where = (
                    "WHERE 1=1"
                    if table == "vw_resolved_picks_canonical"
                    else "WHERE outcome IN ('over', 'under') AND over_label IS NOT NULL AND actual_value IS NOT NULL"
                )
                p_expr = "p_final" if table == "vw_resolved_picks_canonical" else "prob_over"
                df = pd.read_sql(
                    text(
                        f"""
                        SELECT
                            {p_expr} AS p_final,
                            over_label,
                            stat_type
                        FROM {table}
                        {where}
                          AND coalesce(decision_time, created_at) >= now() - (:days * interval '1 day')
                    """
                    ),
                    engine,
                    params={"days": int(max(1, days_back))},
                )
                if not df.empty:
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            return cls()

        if df.empty:
            return cls()

        df["p_final"] = pd.to_numeric(df["p_final"], errors="coerce")
        df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
        return cls.fit_from_dataframe(df)
