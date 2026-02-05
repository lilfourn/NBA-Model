"""Model drift detection: performance, distribution, and calibration drift.

Provides three drift checks:
- **Performance drift**: rolling accuracy drops significantly vs baseline
- **Distribution drift**: Population Stability Index (PSI) on key features
- **Calibration drift**: Expected Calibration Error (ECE) exceeds threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Thresholds
PSI_THRESHOLD = 0.20          # PSI > 0.20 = significant distribution shift
ECE_THRESHOLD = 0.10          # ECE > 0.10 = poor calibration
PERF_DELTA_THRESHOLD = 0.05   # 5% accuracy drop = performance drift
MIN_SAMPLES_PERF = 50         # minimum samples for performance check
MIN_SAMPLES_CAL = 30          # minimum samples for calibration check


@dataclass
class DriftResult:
    """Result of a single drift check."""
    check_type: str          # "performance", "distribution", "calibration"
    is_drifted: bool
    metric_value: float
    threshold: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_type": self.check_type,
            "is_drifted": self.is_drifted,
            "metric_value": round(self.metric_value, 4),
            "threshold": self.threshold,
            "details": self.details,
        }


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.10: no significant shift
    PSI 0.10-0.20: moderate shift
    PSI > 0.20: significant shift
    """
    baseline = np.asarray(baseline, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()

    # Remove non-finite values
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]

    if len(baseline) < 10 or len(current) < 10:
        return 0.0

    # Use baseline quantiles as bin edges for consistency
    edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Remove duplicate edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    base_counts = np.histogram(baseline, bins=edges)[0].astype(np.float64)
    curr_counts = np.histogram(current, bins=edges)[0].astype(np.float64)

    # Normalize to proportions with smoothing
    eps = 1e-4
    base_pct = (base_counts + eps) / (base_counts.sum() + eps * len(base_counts))
    curr_pct = (curr_counts + eps) / (curr_counts.sum() + eps * len(curr_counts))

    psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    return max(0.0, psi)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Measures how well predicted probabilities match actual frequencies.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    mask = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[mask]
    labels = labels[mask]

    if len(probs) < 10:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        count = in_bin.sum()
        if count == 0:
            continue
        avg_prob = probs[in_bin].mean()
        avg_label = labels[in_bin].mean()
        ece += (count / len(probs)) * abs(avg_prob - avg_label)

    return float(ece)


def check_performance_drift(
    recent_correct: np.ndarray,
    baseline_correct: np.ndarray,
    threshold: float = PERF_DELTA_THRESHOLD,
) -> DriftResult:
    """Compare recent accuracy to baseline accuracy.

    Parameters
    ----------
    recent_correct : binary array of recent prediction correctness
    baseline_correct : binary array of baseline prediction correctness
    threshold : accuracy drop threshold to flag drift
    """
    recent = np.asarray(recent_correct, dtype=np.float64).ravel()
    baseline = np.asarray(baseline_correct, dtype=np.float64).ravel()
    recent = recent[np.isfinite(recent)]
    baseline = baseline[np.isfinite(baseline)]

    if len(recent) < MIN_SAMPLES_PERF or len(baseline) < MIN_SAMPLES_PERF:
        return DriftResult(
            check_type="performance",
            is_drifted=False,
            metric_value=0.0,
            threshold=threshold,
            details={"reason": "insufficient_samples", "n_recent": len(recent), "n_baseline": len(baseline)},
        )

    acc_recent = float(recent.mean())
    acc_baseline = float(baseline.mean())
    delta = acc_baseline - acc_recent  # positive = degradation

    return DriftResult(
        check_type="performance",
        is_drifted=delta > threshold,
        metric_value=delta,
        threshold=threshold,
        details={
            "accuracy_recent": round(acc_recent, 4),
            "accuracy_baseline": round(acc_baseline, 4),
            "delta": round(delta, 4),
            "n_recent": len(recent),
            "n_baseline": len(baseline),
        },
    )


def check_distribution_drift(
    baseline_features: dict[str, np.ndarray],
    current_features: dict[str, np.ndarray],
    threshold: float = PSI_THRESHOLD,
) -> DriftResult:
    """Check PSI on each feature, flag if any exceeds threshold."""
    psi_scores: dict[str, float] = {}
    for name in baseline_features:
        if name not in current_features:
            continue
        psi = compute_psi(baseline_features[name], current_features[name])
        psi_scores[name] = psi

    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    drifted_features = {k: v for k, v in psi_scores.items() if v > threshold}

    return DriftResult(
        check_type="distribution",
        is_drifted=len(drifted_features) > 0,
        metric_value=max_psi,
        threshold=threshold,
        details={
            "psi_scores": {k: round(v, 4) for k, v in psi_scores.items()},
            "drifted_features": list(drifted_features.keys()),
            "n_features_checked": len(psi_scores),
        },
    )


def check_calibration_drift(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = ECE_THRESHOLD,
) -> DriftResult:
    """Check if ECE exceeds threshold on recent predictions."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    mask = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[mask]
    labels = labels[mask]

    if len(probs) < MIN_SAMPLES_CAL:
        return DriftResult(
            check_type="calibration",
            is_drifted=False,
            metric_value=0.0,
            threshold=threshold,
            details={"reason": "insufficient_samples", "n_samples": len(probs)},
        )

    ece = compute_ece(probs, labels)

    return DriftResult(
        check_type="calibration",
        is_drifted=ece > threshold,
        metric_value=ece,
        threshold=threshold,
        details={"ece": round(ece, 4), "n_samples": len(probs)},
    )


def run_all_drift_checks(
    *,
    recent_probs: np.ndarray | None = None,
    recent_labels: np.ndarray | None = None,
    recent_correct: np.ndarray | None = None,
    baseline_correct: np.ndarray | None = None,
    baseline_features: dict[str, np.ndarray] | None = None,
    current_features: dict[str, np.ndarray] | None = None,
) -> list[DriftResult]:
    """Run all available drift checks and return results."""
    results: list[DriftResult] = []

    if recent_correct is not None and baseline_correct is not None:
        results.append(check_performance_drift(recent_correct, baseline_correct))

    if baseline_features is not None and current_features is not None:
        results.append(check_distribution_drift(baseline_features, current_features))

    if recent_probs is not None and recent_labels is not None:
        results.append(check_calibration_drift(recent_probs, recent_labels))

    return results
