"""Learned gating model: context features → expert weight simplex.

Uses sklearn LogisticRegression (one-vs-rest) to learn a mapping from
context features to per-expert quality scores, then softmax-normalizes
to produce weights. Robust on small datasets (~2k rows).

Training minimizes per-expert binary cross-entropy: for each expert,
"was this expert correct?" as the target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


@dataclass
class GatingModel:
    """Learned mapping from context features to expert weight simplex."""

    experts: list[str]
    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    models: dict[str, LogisticRegression] = field(default_factory=dict)
    max_weight: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    is_fitted: bool = False

    def fit(
        self,
        context_features: np.ndarray,
        expert_probs: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> "GatingModel":
        """Train one LR per expert: P(expert_correct | context).

        Parameters
        ----------
        context_features : (N, D) array of context features
        expert_probs : dict mapping expert name → (N,) predicted probs
        labels : (N,) binary outcomes (1 = over)
        """
        labels = np.asarray(labels, dtype=np.float64).ravel()
        context_features = np.asarray(context_features, dtype=np.float64)
        n = len(labels)
        if n < 30:
            raise ValueError(f"Need >= 30 samples to train gating model, got {n}")

        for expert in self.experts:
            probs = expert_probs.get(expert)
            if probs is None:
                continue
            probs = np.asarray(probs, dtype=np.float64).ravel()

            # Target: was this expert correct? (predicted direction matches outcome)
            expert_correct = (
                ((probs >= 0.5) & (labels == 1))
                | ((probs < 0.5) & (labels == 0))
            ).astype(int)

            # Need both classes
            if len(np.unique(expert_correct)) < 2:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(context_features)

            lr = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                C=1.0,
                random_state=42,
            )
            lr.fit(X_scaled, expert_correct)

            self.scalers[expert] = scaler
            self.models[expert] = lr

        self.n_train = n
        self.is_fitted = len(self.models) > 0
        return self

    def predict_weights(self, context_features: np.ndarray) -> np.ndarray:
        """Predict expert weights for given context features.

        Parameters
        ----------
        context_features : (D,) or (N, D) array

        Returns
        -------
        weights : (N, n_experts) array, rows sum to 1
        """
        context_features = np.asarray(context_features, dtype=np.float64)
        if context_features.ndim == 1:
            context_features = context_features.reshape(1, -1)

        n = context_features.shape[0]
        n_experts = len(self.experts)
        scores = np.ones((n, n_experts)) * 0.5  # default uniform-ish

        for i, expert in enumerate(self.experts):
            if expert not in self.models:
                continue
            X_scaled = self.scalers[expert].transform(context_features)
            # P(expert correct | context) as the quality score
            scores[:, i] = self.models[expert].predict_proba(X_scaled)[:, 1]

        # Apply max_weight caps before softmax
        for i, expert in enumerate(self.experts):
            cap = self.max_weight.get(expert)
            if cap is not None:
                scores[:, i] = np.minimum(scores[:, i], cap)

        # Softmax normalization to get simplex weights
        # Temperature=1.0: sharper differentiation between experts
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + EPS)

        return weights

    def predict_weights_single(self, context_features: np.ndarray) -> dict[str, float]:
        """Predict weights for a single context, returning a dict."""
        w = self.predict_weights(context_features).ravel()
        return {e: float(w[i]) for i, e in enumerate(self.experts)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "experts": self.experts,
            "scalers": self.scalers,
            "models": self.models,
            "max_weight": self.max_weight,
            "n_train": self.n_train,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, path: str) -> "GatingModel":
        artifact = joblib.load(path)
        return cls(
            experts=artifact["experts"],
            scalers=artifact.get("scalers", {}),
            models=artifact.get("models", {}),
            max_weight=artifact.get("max_weight", {}),
            n_train=artifact.get("n_train", 0),
            is_fitted=artifact.get("is_fitted", False),
        )


def build_context_features(
    expert_probs: dict[str, np.ndarray],
    n_eff: np.ndarray | None = None,
) -> np.ndarray:
    """Build context feature matrix from expert predictions.

    Features:
    - expert_agreement: fraction of experts agreeing on OVER
    - expert_spread: max(p) - min(p)
    - mean_confidence: avg |p - 0.5|
    - log_n_eff: log(1 + effective sample size)
    """
    expert_arrays = []
    for name, probs in expert_probs.items():
        expert_arrays.append(np.asarray(probs, dtype=np.float64).ravel())

    if not expert_arrays:
        raise ValueError("No expert probabilities provided")

    stacked = np.column_stack(expert_arrays)  # (N, n_experts)
    n = stacked.shape[0]

    # Expert agreement: fraction predicting OVER (p >= 0.5)
    agreement = (stacked >= 0.5).mean(axis=1)

    # Expert spread: max - min probability
    spread = stacked.max(axis=1) - stacked.min(axis=1)

    # Mean confidence: average distance from 0.5
    mean_conf = np.abs(stacked - 0.5).mean(axis=1)

    # Log n_eff
    if n_eff is not None:
        log_n_eff = np.log1p(np.asarray(n_eff, dtype=np.float64).ravel())
    else:
        log_n_eff = np.zeros(n)

    return np.column_stack([agreement, spread, mean_conf, log_n_eff])
