"""Context-aware prior probabilities for probability shrinkage.

Instead of shrinking toward a hard-coded constant (0.42), we compute
empirical OVER base rates bucketed by (stat_type, line_bucket) from
recent resolved data.  This adapts to the actual base rate of each
stat type and line region.

Usage:
    from app.ml.context_prior import load_context_priors, get_context_prior

    priors = load_context_priors()  # or from cached JSON
    prior = get_context_prior(priors, stat_type="PTS", line_score=24.5)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

PRIORS_PATH = Path(os.environ.get("MODELS_DIR", "models")) / "context_priors.json"

# Global fallback when no resolved data is available
GLOBAL_FALLBACK_PRIOR = 0.42

# Minimum rows per bucket before falling back to parent level
MIN_BUCKET_ROWS = 30

# Number of quantile-based line buckets per stat type
N_LINE_BUCKETS = 3  # low / mid / high


def compute_context_priors_from_db(engine, *, days_back: int = 90) -> dict[str, Any]:
    """Query recent resolved data and compute context priors.

    Returns a dict structure:
    {
        "global_prior": float,
        "stat_type_priors": {"PTS": float, "REB": float, ...},
        "bucket_priors": {"PTS__low": float, "PTS__mid": float, ...},
        "bucket_edges": {"PTS": [edge1, edge2], ...},
        "meta": {"days_back": int, "total_rows": int, ...},
    }
    """
    from sqlalchemy import text as sa_text

    import pandas as pd

    sql = sa_text(
        """
        select stat_type, line_score, over_label
        from projection_predictions
        where over_label is not null
          and actual_value is not null
          and outcome in ('over', 'under')
          and coalesce(decision_time, resolved_at, created_at)
              >= now() - (:days_back * interval '1 day')
    """
    )
    try:
        df = pd.read_sql(sql, engine, params={"days_back": int(max(1, days_back))})
    except Exception:  # noqa: BLE001
        return _empty_priors()

    if df.empty:
        return _empty_priors()

    df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
    df["line_score"] = pd.to_numeric(df["line_score"], errors="coerce")
    df = df.dropna(subset=["over_label", "stat_type"])

    global_prior = round(float(df["over_label"].mean()), 4)
    stat_type_priors: dict[str, float] = {}
    bucket_priors: dict[str, float] = {}
    bucket_edges: dict[str, list[float]] = {}

    for st, group in df.groupby("stat_type"):
        st = str(st)
        n = len(group)
        if n >= MIN_BUCKET_ROWS:
            stat_type_priors[st] = round(float(group["over_label"].mean()), 4)
        else:
            stat_type_priors[st] = global_prior

        # Compute line-score quantile edges for this stat type
        valid_lines = group["line_score"].dropna()
        if len(valid_lines) >= MIN_BUCKET_ROWS * N_LINE_BUCKETS:
            quantiles = np.linspace(0, 1, N_LINE_BUCKETS + 1)[1:-1]
            edges = [round(float(np.quantile(valid_lines, q)), 2) for q in quantiles]
            bucket_edges[st] = edges

            # Assign bucket labels
            bins = [-math.inf] + edges + [math.inf]
            labels = _bucket_labels(N_LINE_BUCKETS)
            group = group.copy()
            group["bucket"] = pd.cut(
                group["line_score"], bins=bins, labels=labels, include_lowest=True
            )
            for label in labels:
                bucket_group = group[group["bucket"] == label]
                key = f"{st}__{label}"
                if len(bucket_group) >= MIN_BUCKET_ROWS:
                    bucket_priors[key] = round(
                        float(bucket_group["over_label"].mean()), 4
                    )
                else:
                    bucket_priors[key] = stat_type_priors[st]
        else:
            # Not enough data for line buckets; use stat-type prior for all
            for label in _bucket_labels(N_LINE_BUCKETS):
                bucket_priors[f"{st}__{label}"] = stat_type_priors[st]

    return {
        "global_prior": global_prior,
        "stat_type_priors": stat_type_priors,
        "bucket_priors": bucket_priors,
        "bucket_edges": bucket_edges,
        "meta": {
            "days_back": days_back,
            "total_rows": len(df),
            "n_stat_types": len(stat_type_priors),
        },
    }


def _bucket_labels(n: int) -> list[str]:
    if n == 3:
        return ["low", "mid", "high"]
    return [f"q{i}" for i in range(n)]


def _empty_priors() -> dict[str, Any]:
    return {
        "global_prior": GLOBAL_FALLBACK_PRIOR,
        "stat_type_priors": {},
        "bucket_priors": {},
        "bucket_edges": {},
        "meta": {"days_back": 0, "total_rows": 0, "n_stat_types": 0},
    }


def save_context_priors(priors: dict[str, Any], path: Path | str | None = None) -> Path:
    """Save context priors to JSON file."""
    out = Path(path or PRIORS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=2), encoding="utf-8")
    return out


def load_context_priors(path: Path | str | None = None) -> dict[str, Any]:
    """Load context priors from JSON file."""
    p = Path(path or PRIORS_PATH)
    if not p.exists():
        return _empty_priors()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return _empty_priors()


def get_context_prior(
    priors: dict[str, Any],
    stat_type: str | None = None,
    line_score: float | None = None,
) -> float:
    """Look up the context-aware prior for a given stat_type and line_score.

    Falls back: bucket -> stat_type -> global.
    """
    global_prior = priors.get("global_prior", GLOBAL_FALLBACK_PRIOR)

    if not stat_type:
        return global_prior

    stat_type_prior = priors.get("stat_type_priors", {}).get(stat_type, global_prior)

    if line_score is None or line_score != line_score:  # NaN check
        return stat_type_prior

    # Determine bucket
    edges = priors.get("bucket_edges", {}).get(stat_type)
    if not edges:
        return stat_type_prior

    labels = _bucket_labels(len(edges) + 1)
    bucket_label = labels[-1]  # default to highest bucket
    for i, edge in enumerate(edges):
        if line_score <= edge:
            bucket_label = labels[i]
            break

    key = f"{stat_type}__{bucket_label}"
    return priors.get("bucket_priors", {}).get(key, stat_type_prior)
