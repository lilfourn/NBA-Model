from __future__ import annotations

import pandas as pd

from app.ml.stat_calibrator import CALIBRATOR_VERSION, StatTypeCalibrator


def test_fit_stat_calibrator_emits_versioned_meta() -> None:
    rows = []
    for _ in range(140):
        rows.append({"stat_type": "Points", "p_final": 0.62, "over_label": 1})
    for _ in range(120):
        rows.append({"stat_type": "Points", "p_final": 0.52, "over_label": 0})

    df = pd.DataFrame(rows)
    cal = StatTypeCalibrator.fit_from_dataframe(df)

    assert cal.meta.get("calibrator_version") == CALIBRATOR_VERSION
    assert cal.meta.get("global_calibrated") is True


def test_transform_falls_back_global_for_unknown_stat() -> None:
    rows = []
    for _ in range(130):
        rows.append({"stat_type": "Points", "p_final": 0.66, "over_label": 1})
    for _ in range(130):
        rows.append({"stat_type": "Points", "p_final": 0.54, "over_label": 0})

    df = pd.DataFrame(rows)
    cal = StatTypeCalibrator.fit_from_dataframe(df)

    raw = 0.61
    transformed = cal.transform(raw, "UnknownStat")
    assert 0.01 <= transformed <= 0.99


def test_load_legacy_payload_populates_version(tmp_path) -> None:
    # Save a normal calibrator first.
    rows = []
    for _ in range(130):
        rows.append({"stat_type": "Points", "p_final": 0.65, "over_label": 1})
    for _ in range(130):
        rows.append({"stat_type": "Points", "p_final": 0.55, "over_label": 0})

    cal = StatTypeCalibrator.fit_from_dataframe(pd.DataFrame(rows))
    out = tmp_path / "stat_cal.joblib"
    cal.save(out)

    loaded = StatTypeCalibrator.load(out)
    assert loaded.meta.get("calibrator_version") in {CALIBRATOR_VERSION, "legacy", "2.1.0"}


def test_transform_is_stable_for_invalid_calibrator() -> None:
    cal = StatTypeCalibrator()
    p = cal.transform(0.73, "Points")
    assert p == 0.73
