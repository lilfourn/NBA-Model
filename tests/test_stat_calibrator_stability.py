from __future__ import annotations

import pandas as pd

from app.ml.stat_calibrator import StatTypeCalibrator


def _build_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Degenerate per-stat calibrator: constant probability input.
    for idx in range(220):
        rows.append(
            {
                "stat_type": "Assists",
                "p_final": 0.55,
                "over_label": 1 if idx % 2 == 0 else 0,
            }
        )

    # Non-degenerate per-stat calibrator with clear probability spread.
    rates = {
        0.40: 0.20,
        0.45: 0.30,
        0.50: 0.40,
        0.55: 0.60,
        0.60: 0.70,
        0.65: 0.80,
    }
    for prob, over_rate in rates.items():
        positives = int(round(40 * over_rate))
        for idx in range(40):
            rows.append(
                {
                    "stat_type": "Points",
                    "p_final": prob,
                    "over_label": 1 if idx < positives else 0,
                }
            )

    return pd.DataFrame(rows)


def test_degenerate_stat_falls_back_to_global_or_identity() -> None:
    cal = StatTypeCalibrator.fit_from_dataframe(_build_frame())

    assert "Assists" in set(cal.meta.get("degenerate_stats") or [])
    assert "Assists" not in cal.calibrators

    transformed, source, mode = cal.transform_with_info(0.55, "Assists")
    assert 0.01 <= transformed <= 0.99
    assert source in {"global", "identity"}
    assert mode in {"degenerate_fallback", "degenerate_identity"}


def test_non_degenerate_stat_keeps_per_stat_calibrator_active() -> None:
    cal = StatTypeCalibrator.fit_from_dataframe(_build_frame())

    assert "Points" in cal.calibrators
    low, low_source, low_mode = cal.transform_with_info(0.45, "Points")
    high, high_source, high_mode = cal.transform_with_info(0.65, "Points")

    assert low_source == "per_stat"
    assert high_source == "per_stat"
    assert low_mode == "active"
    assert high_mode == "active"
    assert high > low
