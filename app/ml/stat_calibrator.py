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
CALIBRATOR_VERSION = "2.2.0"
MIN_UNIQUE_OUTPUTS = 4
MIN_OUTPUT_RANGE = 0.04
PROBE_GRID = np.linspace(0.35, 0.65, 7, dtype=float)
MIDBAND_PROBE = np.array([0.45, 0.50, 0.55], dtype=float)
MAX_MIDBAND_SHIFT = 0.12

# Stat types excluded from calibration (degenerate base rates)
EXCLUDED_STAT_TYPES: set[str] = {
    "Dunks",
    "Blocked Shots",
    "Blks+Stls",
    "Offensive Rebounds",
    "Personal Fouls",
    "Steals",
}


def _calibrator_shape(cal: Any) -> tuple[int, float, float]:
    outputs = np.asarray(cal.predict(PROBE_GRID), dtype=float)
    unique_outputs = int(len(np.unique(np.round(outputs, 6))))
    output_range = float(np.max(outputs) - np.min(outputs))
    midband_outputs = np.asarray(cal.predict(MIDBAND_PROBE), dtype=float)
    max_midband_shift = float(np.max(np.abs(midband_outputs - MIDBAND_PROBE)))
    return unique_outputs, output_range, max_midband_shift


def _degenerate_reason(
    unique_outputs: int, output_range: float, max_midband_shift: float
) -> str | None:
    if unique_outputs < MIN_UNIQUE_OUTPUTS:
        return "degenerate_low_unique_outputs"
    if output_range < MIN_OUTPUT_RANGE:
        return "degenerate_low_output_range"
    if max_midband_shift > MAX_MIDBAND_SHIFT:
        return "degenerate_excessive_midband_shift"
    return None


def _auto_detect_degenerate_calibrators(
    calibrators: dict[str, Any],
) -> tuple[set[str], dict[str, str]]:
    degenerate_stats: set[str] = set()
    degenerate_reason: dict[str, str] = {}
    for stat_type, cal in (calibrators or {}).items():
        st = str(stat_type)
        try:
            unique_outputs, output_range, max_midband_shift = _calibrator_shape(cal)
        except Exception:  # noqa: BLE001
            degenerate_stats.add(st)
            degenerate_reason[st] = "degenerate_predict_error"
            continue
        reason = _degenerate_reason(
            unique_outputs, output_range, max_midband_shift
        )
        if reason is not None:
            degenerate_stats.add(st)
            degenerate_reason[st] = reason
    return degenerate_stats, degenerate_reason


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
        result, _, _ = self.transform_with_info(p_final, stat_type)
        return result

    def transform_with_info(
        self, p_final: float, stat_type: str
    ) -> tuple[float, str, str]:
        """Apply isotonic recalibration with source/mode diagnostics."""
        degenerate_stats = set(self.meta.get("degenerate_stats") or [])
        cal: IsotonicRegression | None
        source = "identity"
        mode = "none"
        if stat_type in degenerate_stats:
            cal = self.global_calibrator
            if cal is not None:
                source = "global"
                mode = "degenerate_fallback"
            else:
                cal = None
                mode = "degenerate_identity"
        else:
            cal = self.calibrators.get(stat_type)
            if cal is not None:
                source = "per_stat"
                mode = "active"
            else:
                cal = self.global_calibrator
                if cal is not None:
                    source = "global"
                    mode = "fallback_global"
        if cal is None:
            return p_final, source, mode
        try:
            result = float(cal.predict([p_final])[0])
            # Clamp to avoid extreme values
            return max(0.01, min(0.99, result)), source, mode
        except Exception:  # noqa: BLE001
            return p_final, "identity", "error_fallback"

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
            calibrators = payload.get("calibrators", {}) or {}
            meta = payload.get("meta", {}) or {}
            if "calibrator_version" not in meta:
                meta["calibrator_version"] = str(payload.get("version") or "legacy")
            existing_degenerate = set(meta.get("degenerate_stats") or [])
            auto_stats, auto_reasons = _auto_detect_degenerate_calibrators(calibrators)
            newly_detected = sorted(auto_stats - existing_degenerate)
            if auto_stats:
                merged_degenerate = sorted(existing_degenerate | auto_stats)
                meta["degenerate_stats"] = merged_degenerate
                merged_reasons = dict(meta.get("degenerate_reason") or {})
                for stat_type, reason in auto_reasons.items():
                    merged_reasons.setdefault(str(stat_type), str(reason))
                meta["degenerate_reason"] = merged_reasons
            if newly_detected:
                meta["auto_detected_degenerate_stats"] = newly_detected
            return cls(
                calibrators=calibrators,
                global_calibrator=payload.get("global_calibrator"),
                meta=meta,
            )
        except Exception:  # noqa: BLE001
            return cls()

    @classmethod
    def fit_from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        input_col: str = "p_final",
        input_source: str = "p_final",
        fit_window_days: int | None = None,
    ) -> StatTypeCalibrator:
        """Fit calibrators from a DataFrame with probability, over_label, stat_type columns."""
        required = {input_col, "over_label", "stat_type"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame missing columns: {required - set(df.columns)}")

        valid = df.dropna(subset=[input_col, "over_label"]).copy()
        valid = valid[~valid["stat_type"].isin(EXCLUDED_STAT_TYPES)]
        if valid.empty:
            return cls()

        probs = valid[input_col].to_numpy(dtype=float)
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
        degenerate_stats: list[str] = []
        degenerate_reason: dict[str, str] = {}
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
            st_probs = group[input_col].to_numpy(dtype=float)
            st_labels = group["over_label"].to_numpy(dtype=int)
            # Need at least 2 unique labels
            if len(np.unique(st_labels)) < 2:
                stat_meta[st] = {"n": n, "calibrated": False, "reason": "single_class"}
                continue
            cal = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            cal.fit(st_probs, st_labels)
            try:
                unique_outputs, output_range, max_midband_shift = _calibrator_shape(cal)
            except Exception:  # noqa: BLE001
                stat_meta[st] = {
                    "n": n,
                    "calibrated": False,
                    "reason": "degenerate_predict_error",
                    "unique_outputs": 0,
                    "output_range": 0.0,
                    "max_midband_shift": 0.0,
                }
                degenerate_stats.append(st)
                degenerate_reason[st] = "degenerate_predict_error"
                continue
            reason = _degenerate_reason(
                unique_outputs, output_range, max_midband_shift
            )
            if reason is not None:
                stat_meta[st] = {
                    "n": n,
                    "calibrated": False,
                    "reason": reason,
                    "unique_outputs": unique_outputs,
                    "output_range": round(output_range, 6),
                    "max_midband_shift": round(max_midband_shift, 6),
                }
                degenerate_stats.append(st)
                degenerate_reason[st] = reason
                continue

            calibrators[st] = cal
            stat_meta[st] = {
                "n": n,
                "calibrated": True,
                "unique_outputs": unique_outputs,
                "output_range": round(output_range, 6),
                "max_midband_shift": round(max_midband_shift, 6),
            }

        meta = {
            "calibrator_version": CALIBRATOR_VERSION,
            "total_rows": len(valid),
            "n_stat_types_calibrated": len(calibrators),
            "global_calibrated": global_cal is not None,
            "stat_types": stat_meta,
            "min_samples_per_stat": MIN_SAMPLES_PER_STAT,
            "input_source": str(input_source),
            "degenerate_stats": sorted(degenerate_stats),
            "degenerate_reason": degenerate_reason,
            "fit_window_days": int(fit_window_days) if fit_window_days is not None else None,
            "min_unique_outputs": MIN_UNIQUE_OUTPUTS,
            "min_output_range": MIN_OUTPUT_RANGE,
            "max_midband_shift": MAX_MIDBAND_SHIFT,
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
                df = pd.read_sql(
                    text(
                        f"""
                        SELECT
                            details->>'p_pre_cal' AS p_pre_cal,
                            p_raw,
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

        df["p_pre_cal"] = pd.to_numeric(df.get("p_pre_cal"), errors="coerce")
        df["p_raw"] = pd.to_numeric(df.get("p_raw"), errors="coerce")
        df["p_input"] = df["p_pre_cal"].where(df["p_pre_cal"].notna(), df["p_raw"])
        df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
        df = df.dropna(subset=["p_input", "over_label", "stat_type"]).copy()
        if df.empty:
            return cls()
        return cls.fit_from_dataframe(
            df,
            input_col="p_input",
            input_source="details.p_pre_cal_then_p_raw",
            fit_window_days=int(max(1, days_back)),
        )
