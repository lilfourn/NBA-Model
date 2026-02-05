"""Tests for per-expert probability calibration (isotonic + Platt)."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.calibration import (
    CalibratedExpert,
    PlattCalibrator,
    best_calibrator,
    load_calibrator,
)


def test_fit_and_transform_monotonic() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=200)
    labels = (probs + rng.normal(0, 0.1, size=200) > 0.5).astype(float)
    cal = CalibratedExpert.fit(probs, labels)
    transformed = cal.transform(probs)
    # Isotonic should be monotonically non-decreasing
    sorted_idx = np.argsort(probs)
    sorted_out = transformed[sorted_idx]
    diffs = np.diff(sorted_out)
    assert np.all(diffs >= -1e-6), "Isotonic output should be monotonic"


def test_fit_requires_min_samples() -> None:
    with pytest.raises(ValueError, match="Need >= 20"):
        CalibratedExpert.fit(np.array([0.5] * 10), np.array([1.0] * 10))


def test_roundtrip_to_dict_from_dict() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=100)
    labels = (probs > 0.5).astype(float)
    cal = CalibratedExpert.fit(probs, labels)
    original = cal.transform(probs)

    data = cal.to_dict()
    restored = CalibratedExpert.from_dict(data)
    roundtripped = restored.transform(probs)

    np.testing.assert_allclose(original, roundtripped, atol=1e-6)


def test_transform_clips_to_bounds() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=100)
    labels = (probs > 0.5).astype(float)
    cal = CalibratedExpert.fit(probs, labels)
    extreme = cal.transform(np.array([0.0, 1.0]))
    assert extreme[0] >= 0.01
    assert extreme[1] <= 0.99


def test_n_cal_tracked() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=50)
    labels = (probs > 0.5).astype(float)
    cal = CalibratedExpert.fit(probs, labels)
    assert cal.n_cal == 50
    data = cal.to_dict()
    assert data["n_cal"] == 50


# --- PlattCalibrator tests ---


def test_platt_fit_and_transform() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=200)
    labels = (probs + rng.normal(0, 0.1, size=200) > 0.5).astype(float)
    platt = PlattCalibrator.fit(probs, labels)
    transformed = platt.transform(probs)
    assert transformed.shape == probs.shape
    assert np.all(transformed >= 0.01)
    assert np.all(transformed <= 0.99)


def test_platt_requires_min_samples() -> None:
    with pytest.raises(ValueError, match="Need >= 20"):
        PlattCalibrator.fit(np.array([0.5] * 10), np.array([1.0] * 10))


def test_platt_roundtrip_to_dict_from_dict() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=100)
    labels = (probs > 0.5).astype(float)
    platt = PlattCalibrator.fit(probs, labels)
    original = platt.transform(probs)

    data = platt.to_dict()
    assert data["type"] == "platt"
    restored = PlattCalibrator.from_dict(data)
    roundtripped = restored.transform(probs)

    np.testing.assert_allclose(original, roundtripped, atol=1e-6)


def test_platt_n_cal_tracked() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=75)
    labels = (probs > 0.5).astype(float)
    platt = PlattCalibrator.fit(probs, labels)
    assert platt.n_cal == 75


# --- best_calibrator tests ---


def test_best_calibrator_returns_valid_type() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=200)
    labels = (probs + rng.normal(0, 0.1, size=200) > 0.5).astype(float)
    cal = best_calibrator(probs, labels)
    assert isinstance(cal, (CalibratedExpert, PlattCalibrator))
    transformed = cal.transform(probs)
    assert transformed.shape == probs.shape
    assert np.all(transformed >= 0.01)
    assert np.all(transformed <= 0.99)


def test_best_calibrator_serializes_correctly() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=200)
    labels = (probs + rng.normal(0, 0.1, size=200) > 0.5).astype(float)
    cal = best_calibrator(probs, labels)
    data = cal.to_dict()
    restored = load_calibrator(data)
    assert type(restored) == type(cal)
    np.testing.assert_allclose(
        cal.transform(probs), restored.transform(probs), atol=1e-6
    )


# --- load_calibrator tests ---


def test_load_calibrator_isotonic() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=100)
    labels = (probs > 0.5).astype(float)
    cal = CalibratedExpert.fit(probs, labels)
    data = cal.to_dict()
    loaded = load_calibrator(data)
    assert isinstance(loaded, CalibratedExpert)


def test_load_calibrator_platt() -> None:
    data = {"type": "platt", "a": 1.5, "b": -0.3, "n_cal": 100}
    loaded = load_calibrator(data)
    assert isinstance(loaded, PlattCalibrator)
    assert loaded.a == 1.5
    assert loaded.b == -0.3
