"""Tests for drift detection module."""
from __future__ import annotations

import numpy as np

from app.ml.drift_detection import (
    check_calibration_drift,
    check_distribution_drift,
    check_performance_drift,
    compute_ece,
    compute_psi,
    run_all_drift_checks,
)


# --- compute_psi tests ---


def test_psi_identical_distributions() -> None:
    rng = np.random.RandomState(42)
    data = rng.normal(0, 1, size=500)
    psi = compute_psi(data, data)
    assert psi < 0.05  # near-zero for identical


def test_psi_shifted_distribution() -> None:
    rng = np.random.RandomState(42)
    baseline = rng.normal(0, 1, size=500)
    shifted = rng.normal(2, 1, size=500)  # large shift
    psi = compute_psi(baseline, shifted)
    assert psi > 0.20  # significant shift


def test_psi_small_samples_returns_zero() -> None:
    assert compute_psi(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0


def test_psi_non_negative() -> None:
    rng = np.random.RandomState(42)
    a = rng.uniform(0, 1, 200)
    b = rng.uniform(0.2, 0.8, 200)
    assert compute_psi(a, b) >= 0.0


# --- compute_ece tests ---


def test_ece_perfect_calibration() -> None:
    # If probs perfectly match label frequencies, ECE should be low
    rng = np.random.RandomState(42)
    probs = rng.uniform(0, 1, 1000)
    labels = (rng.uniform(0, 1, 1000) < probs).astype(float)
    ece = compute_ece(probs, labels)
    assert ece < 0.10  # should be well-calibrated


def test_ece_poor_calibration() -> None:
    # All probs = 0.9 but only 10% are actually 1
    probs = np.full(200, 0.9)
    labels = np.zeros(200)
    labels[:20] = 1.0
    ece = compute_ece(probs, labels)
    assert ece > 0.50  # very poorly calibrated


def test_ece_small_samples_returns_zero() -> None:
    assert compute_ece(np.array([0.5] * 5), np.array([1.0] * 5)) == 0.0


# --- check_performance_drift tests ---


def test_performance_drift_detected() -> None:
    baseline = np.ones(100)  # 100% accuracy
    recent = np.concatenate([np.ones(50), np.zeros(50)])  # 50% accuracy
    result = check_performance_drift(recent, baseline)
    assert result.is_drifted
    assert result.check_type == "performance"
    assert result.metric_value > 0.05


def test_performance_no_drift() -> None:
    baseline = np.concatenate([np.ones(60), np.zeros(40)])  # 60%
    recent = np.concatenate([np.ones(58), np.zeros(42)])    # 58%
    result = check_performance_drift(recent, baseline)
    assert not result.is_drifted


def test_performance_insufficient_samples() -> None:
    result = check_performance_drift(np.ones(10), np.ones(10))
    assert not result.is_drifted
    assert result.details["reason"] == "insufficient_samples"


# --- check_distribution_drift tests ---


def test_distribution_drift_detected() -> None:
    rng = np.random.RandomState(42)
    baseline = {"feat_a": rng.normal(0, 1, 500)}
    current = {"feat_a": rng.normal(3, 1, 500)}
    result = check_distribution_drift(baseline, current)
    assert result.is_drifted
    assert "feat_a" in result.details["drifted_features"]


def test_distribution_no_drift() -> None:
    rng = np.random.RandomState(42)
    baseline = {"feat_a": rng.normal(0, 1, 500)}
    current = {"feat_a": rng.normal(0.05, 1, 500)}  # tiny shift
    result = check_distribution_drift(baseline, current)
    assert not result.is_drifted


# --- check_calibration_drift tests ---


def test_calibration_drift_detected() -> None:
    probs = np.full(200, 0.9)
    labels = np.zeros(200)
    labels[:20] = 1.0
    result = check_calibration_drift(probs, labels)
    assert result.is_drifted
    assert result.metric_value > 0.10


def test_calibration_no_drift() -> None:
    rng = np.random.RandomState(42)
    probs = rng.uniform(0, 1, 500)
    labels = (rng.uniform(0, 1, 500) < probs).astype(float)
    result = check_calibration_drift(probs, labels)
    assert not result.is_drifted


def test_calibration_insufficient_samples() -> None:
    result = check_calibration_drift(np.array([0.5] * 10), np.array([1.0] * 10))
    assert not result.is_drifted


# --- run_all_drift_checks tests ---


def test_run_all_checks() -> None:
    rng = np.random.RandomState(42)
    results = run_all_drift_checks(
        recent_probs=rng.uniform(0, 1, 100),
        recent_labels=(rng.uniform(0, 1, 100) > 0.5).astype(float),
        recent_correct=np.ones(100),
        baseline_correct=np.ones(100),
        baseline_features={"f1": rng.normal(0, 1, 200)},
        current_features={"f1": rng.normal(0, 1, 200)},
    )
    assert len(results) == 3
    types = {r.check_type for r in results}
    assert types == {"performance", "distribution", "calibration"}


def test_run_partial_checks() -> None:
    rng = np.random.RandomState(42)
    results = run_all_drift_checks(
        recent_probs=rng.uniform(0, 1, 100),
        recent_labels=(rng.uniform(0, 1, 100) > 0.5).astype(float),
    )
    assert len(results) == 1
    assert results[0].check_type == "calibration"


def test_drift_result_to_dict() -> None:
    result = check_performance_drift(np.ones(100), np.ones(100))
    d = result.to_dict()
    assert "check_type" in d
    assert "is_drifted" in d
    assert "metric_value" in d
    assert "threshold" in d
    assert "details" in d
