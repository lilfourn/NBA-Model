from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.selection_policy import (
    SelectionPolicy,
    default_selection_policy,
    fit_selection_policy,
)


def _make_rows(
    *,
    stat_type: str,
    n: int,
    p_final: float,
    accuracy: float,
    conformal_set_size: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    n_correct = int(round(n * accuracy))
    pick = 1 if p_final >= 0.5 else 0
    for i in range(n):
        is_correct = i < n_correct
        over_label = pick if is_correct else 1 - pick
        rows.append(
            {
                "stat_type": stat_type,
                "p_final": p_final,
                "over_label": over_label,
                "conformal_set_size": conformal_set_size,
            }
        )
    return rows


def test_default_policy_has_expected_defaults() -> None:
    policy = default_selection_policy()
    assert policy.global_threshold == 0.60
    assert policy.conformal_ambiguous_penalty == 0.02
    assert policy.threshold_for("Points") == 0.60


def test_fit_selection_policy_trains_per_stat_and_global() -> None:
    rows = []
    rows.extend(_make_rows(stat_type="Points", n=300, p_final=0.63, accuracy=0.62))
    rows.extend(_make_rows(stat_type="Assists", n=120, p_final=0.61, accuracy=0.58))
    df = pd.DataFrame(rows)

    policy = fit_selection_policy(
        df,
        days_back=180,
        min_rows_per_stat=200,
        coverage_floor=0.4,
        target_hit_rate=0.55,
    )

    # Points should be trained; Assists should fall back to global threshold.
    assert "Points" in policy.per_stat_thresholds
    assert "Assists" not in policy.per_stat_thresholds
    assert 0.55 <= policy.global_threshold <= 0.72


def test_threshold_for_applies_conformal_penalty() -> None:
    policy = SelectionPolicy(
        version="v",
        fitted_at="now",
        source_rows=10,
        days_back=1,
        global_threshold=0.6,
        per_stat_thresholds={"Points": 0.58},
        conformal_ambiguous_penalty=0.02,
        min_rows_per_stat=200,
        coverage_floor=0.4,
        target_hit_rate=0.55,
        threshold_grid=[0.55, 0.56],
        diagnostics={},
    )

    assert policy.threshold_for("Points", conformal_set_size=1) == 0.58
    assert policy.threshold_for("Points", conformal_set_size=2) == 0.60
    assert policy.threshold_for("Rebounds", conformal_set_size=2) == 0.62


def test_fit_selection_policy_handles_empty_input() -> None:
    df = pd.DataFrame(columns=["stat_type", "p_final", "over_label", "conformal_set_size"])
    policy = fit_selection_policy(df, days_back=180)
    assert policy.source_rows == 0
    assert policy.per_stat_thresholds == {}
    assert policy.global_threshold == 0.60


def test_threshold_grid_respected() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "stat_type": ["Points"] * 260,
            "p_final": rng.uniform(0.55, 0.7, size=260),
            "over_label": rng.integers(0, 2, size=260),
            "conformal_set_size": [1] * 260,
        }
    )

    policy = fit_selection_policy(
        df,
        days_back=180,
        min_rows_per_stat=200,
        threshold_start=0.57,
        threshold_end=0.61,
        threshold_step=0.02,
    )

    assert set(policy.threshold_grid) == {0.57, 0.59, 0.61}
    assert policy.global_threshold in policy.threshold_grid
    if "Points" in policy.per_stat_thresholds:
        assert policy.per_stat_thresholds["Points"] in policy.threshold_grid
