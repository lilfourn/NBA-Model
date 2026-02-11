"""Per-expert probability calibration.

Supports two calibrators:
- **Isotonic** (CalibratedExpert): non-parametric, powerful on large holdouts
- **Platt** (PlattCalibrator): parametric sigmoid, smoother on small holdouts

``best_calibrator()`` fits both and returns whichever achieves the lower
Brier score on the supplied holdout data.

Serialization uses simple dicts so artifacts are sklearn-version-independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibratedExpert:
    """Isotonic calibration fitted on holdout predictions."""

    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    n_cal: int
    y_min: float = 0.01
    y_max: float = 0.99

    @classmethod
    def fit(cls, probs: np.ndarray, labels: np.ndarray) -> "CalibratedExpert":
        probs = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        mask = np.isfinite(probs) & np.isfinite(labels)
        probs = probs[mask]
        labels = labels[mask]
        if len(probs) < 20:
            raise ValueError(f"Need >= 20 calibration samples, got {len(probs)}")
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(probs, labels)
        return cls(
            x_thresholds=iso.X_thresholds_.copy(),
            y_thresholds=iso.y_thresholds_.copy(),
            n_cal=len(probs),
        )

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64).ravel()
        clipped = np.clip(probs, self.x_thresholds[0], self.x_thresholds[-1])
        out = np.interp(clipped, self.x_thresholds, self.y_thresholds)
        return np.clip(out, self.y_min, self.y_max).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "X_thresholds": self.x_thresholds.tolist(),
            "y_thresholds": self.y_thresholds.tolist(),
            "n_cal": self.n_cal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibratedExpert":
        return cls(
            x_thresholds=np.array(data["X_thresholds"], dtype=np.float64),
            y_thresholds=np.array(data["y_thresholds"], dtype=np.float64),
            n_cal=int(data.get("n_cal", 0)),
        )


class Calibrator(Protocol):
    """Common interface shared by CalibratedExpert and PlattCalibrator."""

    n_cal: int

    def transform(self, probs: np.ndarray) -> np.ndarray: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class PlattCalibrator:
    """Platt (logistic) calibration: sigmoid(a * logit(p) + b).

    Smoother than isotonic — better when calibration set is small (<100).
    """

    a: float
    b: float
    n_cal: int
    y_min: float = 0.01
    y_max: float = 0.99

    @classmethod
    def fit(cls, probs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        probs = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        mask = np.isfinite(probs) & np.isfinite(labels)
        probs = probs[mask]
        labels = labels[mask]
        if len(probs) < 20:
            raise ValueError(f"Need >= 20 calibration samples, got {len(probs)}")
        # Logit-transform probabilities, clipping to avoid inf
        eps = 1e-6
        p_clip = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6)
        lr.fit(logits, labels.astype(int))
        a = float(lr.coef_[0, 0])
        b = float(lr.intercept_[0])
        return cls(a=a, b=b, n_cal=len(probs))

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64).ravel()
        eps = 1e-6
        p_clip = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clip / (1.0 - p_clip))
        scaled = self.a * logits + self.b
        # Numerically stable sigmoid
        out = np.where(
            scaled >= 0,
            1.0 / (1.0 + np.exp(-scaled)),
            np.exp(scaled) / (1.0 + np.exp(scaled)),
        )
        return np.clip(out, self.y_min, self.y_max).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "platt",
            "a": self.a,
            "b": self.b,
            "n_cal": self.n_cal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlattCalibrator":
        return cls(
            a=float(data["a"]),
            b=float(data["b"]),
            n_cal=int(data.get("n_cal", 0)),
        )


def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    return float(np.mean((probs - labels) ** 2))


def best_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
) -> CalibratedExpert | PlattCalibrator:
    """Fit isotonic + Platt, return whichever has lower Brier score.

    Uses leave-one-out-ish approach: fit on full data, evaluate on same data.
    Isotonic will always win on in-sample Brier, so we add a complexity
    penalty: isotonic needs to beat Platt by >0.005 to be chosen (avoids
    overfitting isotonic on tiny holdouts).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()

    iso = CalibratedExpert.fit(probs, labels)
    platt = PlattCalibrator.fit(probs, labels)

    brier_iso = _brier_score(iso.transform(probs), labels)
    brier_platt = _brier_score(platt.transform(probs), labels)

    # Isotonic must beat Platt by a margin to compensate for its flexibility
    if brier_iso < brier_platt - 0.005:
        return iso
    return platt


def load_calibrator(data: dict[str, Any]) -> CalibratedExpert | PlattCalibrator:
    """Load a calibrator from a serialized dict (auto-detects type)."""
    if data.get("type") == "platt":
        return PlattCalibrator.from_dict(data)
    return CalibratedExpert.from_dict(data)
