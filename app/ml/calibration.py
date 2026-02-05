"""Per-expert isotonic calibration.

Wraps sklearn IsotonicRegression to map raw model probabilities to
well-calibrated probabilities. Fitted on holdout data during training,
applied at inference time.

Serialization uses (X_thresholds, y_thresholds) arrays and np.interp
for reconstruction — avoids depending on sklearn internal _build_f.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


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
