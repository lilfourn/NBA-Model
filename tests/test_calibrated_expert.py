"""Tests for per-expert isotonic calibration."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.calibration import CalibratedExpert


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
