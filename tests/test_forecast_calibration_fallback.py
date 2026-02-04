import numpy as np

from app.modeling.forecast_calibration import ForecastDistributionCalibrator, StatTypeCalibrator


def _identity_calibrator() -> StatTypeCalibrator:
    # Identity PIT map so p_under == raw normal CDF (after optional param inflation).
    return StatTypeCalibrator(
        params={"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0},
        pit_x=np.array([0.0, 1.0], dtype=np.float32),
        pit_y=np.array([0.0, 1.0], dtype=np.float32),
    )


def test_get_with_fallback_prefers_per_stat() -> None:
    cal = ForecastDistributionCalibrator(
        {"Points": _identity_calibrator(), "__global__": _identity_calibrator()}
    )
    picked, status = cal.get_with_fallback("Points")
    assert picked is not None
    assert status == "per_stat"


def test_get_with_fallback_uses_global() -> None:
    cal = ForecastDistributionCalibrator({"__global__": _identity_calibrator()})
    picked, status = cal.get_with_fallback("Unknown Stat")
    assert picked is not None
    assert status == "fallback_global"


def test_get_with_fallback_raw_when_missing() -> None:
    cal = ForecastDistributionCalibrator({"Points": _identity_calibrator()})
    picked, status = cal.get_with_fallback("Assists")
    assert picked is None
    assert status == "raw"

