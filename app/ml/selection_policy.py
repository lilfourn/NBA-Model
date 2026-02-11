"""Adaptive selection policy for publishable picks.

Learns per-stat confidence thresholds that maximize hit rate while enforcing
coverage constraints. Includes deterministic conformal ambiguity penalty.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

DEFAULT_POLICY_VERSION = "selective_v1"
DEFAULT_POLICY_PATH = Path(os.environ.get("MODELS_DIR", "models")) / "selection_policy.json"


@dataclass
class SelectionPolicy:
    version: str
    fitted_at: str
    source_rows: int
    days_back: int
    global_threshold: float
    per_stat_thresholds: dict[str, float]
    conformal_ambiguous_penalty: float
    min_rows_per_stat: int
    coverage_floor: float
    target_hit_rate: float
    threshold_grid: list[float]
    diagnostics: dict[str, Any]

    def threshold_for(self, stat_type: str, conformal_set_size: int | None = None) -> float:
        base = float(self.per_stat_thresholds.get(stat_type, self.global_threshold))
        if conformal_set_size == 2:
            base += float(self.conformal_ambiguous_penalty)
        return float(max(0.5, min(0.99, round(base, 4))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fitted_at": self.fitted_at,
            "source_rows": int(self.source_rows),
            "days_back": int(self.days_back),
            "global_threshold": float(self.global_threshold),
            "per_stat_thresholds": {
                str(k): float(v) for k, v in self.per_stat_thresholds.items()
            },
            "conformal_ambiguous_penalty": float(self.conformal_ambiguous_penalty),
            "min_rows_per_stat": int(self.min_rows_per_stat),
            "coverage_floor": float(self.coverage_floor),
            "target_hit_rate": float(self.target_hit_rate),
            "threshold_grid": [float(v) for v in self.threshold_grid],
            "diagnostics": dict(self.diagnostics or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SelectionPolicy":
        return cls(
            version=str(payload.get("version") or DEFAULT_POLICY_VERSION),
            fitted_at=str(payload.get("fitted_at") or datetime.now(timezone.utc).isoformat()),
            source_rows=int(payload.get("source_rows") or 0),
            days_back=int(payload.get("days_back") or 0),
            global_threshold=float(payload.get("global_threshold") or 0.60),
            per_stat_thresholds={
                str(k): float(v)
                for k, v in (payload.get("per_stat_thresholds") or {}).items()
            },
            conformal_ambiguous_penalty=float(
                payload.get("conformal_ambiguous_penalty") or 0.02
            ),
            min_rows_per_stat=int(payload.get("min_rows_per_stat") or 200),
            coverage_floor=float(payload.get("coverage_floor") or 0.40),
            target_hit_rate=float(payload.get("target_hit_rate") or 0.55),
            threshold_grid=[
                float(v) for v in (payload.get("threshold_grid") or np.arange(0.55, 0.73, 0.01))
            ],
            diagnostics=dict(payload.get("diagnostics") or {}),
        )

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path or DEFAULT_POLICY_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SelectionPolicy":
        p = Path(path or DEFAULT_POLICY_PATH)
        if not p.exists():
            return default_selection_policy()
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return default_selection_policy()
        return cls.from_dict(payload)


def default_selection_policy() -> SelectionPolicy:
    return SelectionPolicy(
        version=DEFAULT_POLICY_VERSION,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        source_rows=0,
        days_back=0,
        global_threshold=0.60,
        per_stat_thresholds={},
        conformal_ambiguous_penalty=0.02,
        min_rows_per_stat=200,
        coverage_floor=0.40,
        target_hit_rate=0.55,
        threshold_grid=[round(v, 2) for v in np.arange(0.55, 0.73, 0.01)],
        diagnostics={"reason": "default_policy"},
    )


def _coerce_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"p_final", "over_label", "stat_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Selection policy input missing columns: {sorted(missing)}")

    frame = df.copy()
    frame["p_final"] = pd.to_numeric(frame["p_final"], errors="coerce")
    frame["over_label"] = pd.to_numeric(frame["over_label"], errors="coerce")
    frame = frame.dropna(subset=["p_final", "over_label", "stat_type"])
    if frame.empty:
        return frame

    frame["stat_type"] = frame["stat_type"].astype(str)
    frame["over_label"] = frame["over_label"].astype(int)
    frame["pick"] = (frame["p_final"] >= 0.5).astype(int)
    frame["is_correct"] = (frame["pick"] == frame["over_label"]).astype(int)
    frame["p_pick"] = np.maximum(frame["p_final"], 1.0 - frame["p_final"])

    if "conformal_set_size" in frame.columns:
        frame["conformal_set_size"] = pd.to_numeric(
            frame["conformal_set_size"], errors="coerce"
        )
    else:
        frame["conformal_set_size"] = np.nan

    return frame


def _threshold_metrics(
    frame: pd.DataFrame,
    threshold: float,
    *,
    conformal_penalty: float,
) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "threshold": float(threshold),
            "selected_n": 0,
            "coverage": 0.0,
            "accuracy": None,
        }

    required = np.full(len(frame), float(threshold), dtype=float)
    conformal = frame["conformal_set_size"].to_numpy(dtype=float)
    required = required + np.where(conformal == 2.0, float(conformal_penalty), 0.0)

    selected = frame["p_pick"].to_numpy(dtype=float) >= required
    selected_n = int(selected.sum())
    coverage = float(selected_n / len(frame))
    if selected_n == 0:
        accuracy = None
    else:
        accuracy = float(frame.loc[selected, "is_correct"].mean())
    return {
        "threshold": float(threshold),
        "selected_n": selected_n,
        "coverage": round(coverage, 4),
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
    }


def _choose_threshold(
    frame: pd.DataFrame,
    *,
    thresholds: list[float],
    conformal_penalty: float,
    coverage_floor: float,
    target_hit_rate: float,
) -> tuple[float, list[dict[str, float | int | None]]]:
    metrics = [
        _threshold_metrics(frame, th, conformal_penalty=conformal_penalty)
        for th in thresholds
    ]

    feasible = [m for m in metrics if float(m["coverage"] or 0.0) >= coverage_floor and m["accuracy"] is not None]
    good = [m for m in feasible if float(m["accuracy"] or 0.0) >= target_hit_rate]
    pool = good or feasible

    if not pool:
        fallback = min(thresholds) if thresholds else 0.60
        return float(fallback), metrics

    pool.sort(
        key=lambda m: (
            float(m["accuracy"] or 0.0),
            float(m["coverage"] or 0.0),
            -float(m["threshold"] or 0.0),
        ),
        reverse=True,
    )
    return float(pool[0]["threshold"]), metrics


def fit_selection_policy(
    df: pd.DataFrame,
    *,
    days_back: int,
    min_rows_per_stat: int = 200,
    coverage_floor: float = 0.40,
    target_hit_rate: float = 0.55,
    threshold_start: float = 0.55,
    threshold_end: float = 0.72,
    threshold_step: float = 0.01,
    conformal_ambiguous_penalty: float = 0.02,
) -> SelectionPolicy:
    frame = _coerce_training_frame(df)
    grid = [round(float(v), 2) for v in np.arange(threshold_start, threshold_end + 1e-9, threshold_step)]
    if not grid:
        grid = [0.60]

    if frame.empty:
        policy = default_selection_policy()
        policy.days_back = int(max(0, days_back))
        policy.min_rows_per_stat = int(min_rows_per_stat)
        policy.coverage_floor = float(coverage_floor)
        policy.target_hit_rate = float(target_hit_rate)
        policy.conformal_ambiguous_penalty = float(conformal_ambiguous_penalty)
        policy.threshold_grid = grid
        policy.diagnostics = {"reason": "no_rows"}
        return policy

    global_threshold, global_grid = _choose_threshold(
        frame,
        thresholds=grid,
        conformal_penalty=conformal_ambiguous_penalty,
        coverage_floor=coverage_floor,
        target_hit_rate=target_hit_rate,
    )

    per_stat: dict[str, float] = {}
    per_stat_diag: dict[str, Any] = {}
    for stat_type, group in frame.groupby("stat_type"):
        n_rows = int(len(group))
        if n_rows < int(min_rows_per_stat):
            per_stat_diag[str(stat_type)] = {
                "n": n_rows,
                "status": "fallback_global",
                "reason": "insufficient_rows",
            }
            continue

        stat_threshold, stat_grid = _choose_threshold(
            group,
            thresholds=grid,
            conformal_penalty=conformal_ambiguous_penalty,
            coverage_floor=coverage_floor,
            target_hit_rate=target_hit_rate,
        )
        per_stat[str(stat_type)] = float(stat_threshold)
        per_stat_diag[str(stat_type)] = {
            "n": n_rows,
            "status": "trained",
            "threshold": round(float(stat_threshold), 4),
            "grid_metrics": stat_grid,
        }

    diagnostics = {
        "global_threshold": round(float(global_threshold), 4),
        "global_grid_metrics": global_grid,
        "per_stat": per_stat_diag,
        "trained_stat_types": int(len(per_stat)),
        "total_stat_types": int(frame["stat_type"].nunique()),
    }

    return SelectionPolicy(
        version=DEFAULT_POLICY_VERSION,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        source_rows=int(len(frame)),
        days_back=int(max(1, days_back)),
        global_threshold=float(global_threshold),
        per_stat_thresholds=per_stat,
        conformal_ambiguous_penalty=float(conformal_ambiguous_penalty),
        min_rows_per_stat=int(min_rows_per_stat),
        coverage_floor=float(coverage_floor),
        target_hit_rate=float(target_hit_rate),
        threshold_grid=grid,
        diagnostics=diagnostics,
    )


def fit_selection_policy_from_db(
    engine,
    *,
    days_back: int = 180,
    min_rows_per_stat: int = 200,
    coverage_floor: float = 0.40,
    target_hit_rate: float = 0.55,
    threshold_start: float = 0.55,
    threshold_end: float = 0.72,
    threshold_step: float = 0.01,
    conformal_ambiguous_penalty: float = 0.02,
) -> SelectionPolicy:
    queries = [
        text(
            """
            select
                p_final,
                over_label,
                stat_type,
                cast(details->>'conformal_set_size' as integer) as conformal_set_size
            from vw_resolved_picks_canonical
            where coalesce(decision_time, created_at) >= now() - (:days * interval '1 day')
            """
        ),
        text(
            """
            select
                prob_over as p_final,
                over_label,
                stat_type,
                cast(details->>'conformal_set_size' as integer) as conformal_set_size
            from projection_predictions
            where outcome in ('over', 'under')
              and over_label is not null
              and actual_value is not null
              and coalesce(decision_time, created_at) >= now() - (:days * interval '1 day')
            """
        ),
    ]

    df = pd.DataFrame()
    for query in queries:
        try:
            df = pd.read_sql(query, engine, params={"days": int(max(1, days_back))})
        except Exception:  # noqa: BLE001
            continue
        if not df.empty:
            break

    return fit_selection_policy(
        df,
        days_back=days_back,
        min_rows_per_stat=min_rows_per_stat,
        coverage_floor=coverage_floor,
        target_hit_rate=target_hit_rate,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
        conformal_ambiguous_penalty=conformal_ambiguous_penalty,
    )
