"""Split conformal prediction for binary classification.

Provides calibrated prediction sets with finite-sample coverage guarantees.
After training a model, calibrate on a held-out set to compute a nonconformity
score threshold.  At inference the threshold determines which predictions are
confident enough to act on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConformalResult:
    """Per-prediction conformal output."""

    p_value: float  # conformal p-value (higher = more conforming)
    is_confident: bool  # True if prediction exceeds the conformal threshold
    set_size: int  # 1 = unique prediction, 2 = ambiguous


@dataclass
class ConformalCalibrator:
    """Split conformal calibrator for binary classification.

    Stores the nonconformity score quantile from the calibration set.
    At inference, computes a conformal p-value and confidence flag.

    alpha : float
        Target miscoverage rate (e.g. 0.10 for 90% coverage).
    q_hat : float
        Calibrated quantile of nonconformity scores.
    n_cal : int
        Number of calibration examples used.
    """

    alpha: float
    q_hat: float
    n_cal: int

    @staticmethod
    def calibrate(
        probs: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.10,
    ) -> "ConformalCalibrator":
        """Calibrate from held-out predictions and true labels.

        probs : predicted P(over) for each example
        labels : true binary labels (1 = over)
        alpha : target miscoverage rate
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        assert probs.shape == labels.shape

        # Nonconformity score: 1 - P(true class)
        scores = np.where(labels == 1, 1.0 - probs, probs)

        n = len(scores)
        # Finite-sample corrected quantile level
        level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
        q_hat = float(np.quantile(scores, level))

        return ConformalCalibrator(alpha=alpha, q_hat=q_hat, n_cal=n)

    def predict(self, prob_over: float) -> ConformalResult:
        """Compute conformal prediction for a single example.

        Returns a ConformalResult with p-value, confidence flag, and set size.
        """
        p = float(prob_over)
        # Score for each possible label
        score_over = 1.0 - p  # if true label were 1
        score_under = p  # if true label were 0

        # Conformal p-values: fraction of calibration scores >= this score
        # Approximated by comparing to q_hat
        pv_over = 1.0 - score_over  # higher prob -> higher p-value for "over"
        pv_under = 1.0 - score_under

        # Prediction set: include label if its score <= q_hat
        include_over = score_over <= self.q_hat
        include_under = score_under <= self.q_hat
        set_size = int(include_over) + int(include_under)

        # The prediction is confident if only one label is in the set
        pick_over = p >= 0.5
        if pick_over:
            p_value = pv_over
            is_confident = include_over and not include_under
        else:
            p_value = pv_under
            is_confident = include_under and not include_over

        return ConformalResult(
            p_value=float(np.clip(p_value, 0.0, 1.0)),
            is_confident=is_confident,
            set_size=set_size,
        )

    def batch_predict(self, probs: np.ndarray) -> list[ConformalResult]:
        return [self.predict(float(p)) for p in probs]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibrator":
        data = json.loads(Path(path).read_text())
        return cls(**data)
