"""Hybrid ensemble combiner: Thompson Sampling + Gating Model + Meta-learner.

Three-layer combination with learned mixing coefficients:
1. Thompson Sampling weights (exploration-aware, online-learned)
2. Gating model weights (context-learned, batch-trained)
3. Meta-learner probability (stacked logistic regression)

Final output is a convex combination of the three sub-predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.modeling.gating_model import GatingModel, build_context_features
from app.modeling.thompson_ensemble import ThompsonSamplingEnsembler

EPS = 1e-6


def _logit_average(expert_probs: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted logit-space average of expert probabilities."""
    logit_sum = 0.0
    w_sum = 0.0
    for e, w in weights.items():
        p = expert_probs.get(e)
        if p is None or w < EPS:
            continue
        p = float(np.clip(p, EPS, 1.0 - EPS))
        logit_sum += w * np.log(p / (1.0 - p))
        w_sum += w
    if w_sum < EPS:
        return 0.5
    avg_logit = logit_sum / w_sum
    return float(1.0 / (1.0 + np.exp(-avg_logit)))


@dataclass
class HybridEnsembleCombiner:
    """Combines Thompson Sampling, Gating Model, and Meta-learner.

    Mixing coefficients (alpha, beta, gamma) control how much each
    sub-ensemble contributes. They default to equal weight and can be
    learned from validation data via ``fit_mixing()``.
    """

    thompson: ThompsonSamplingEnsembler | None = None
    gating: GatingModel | None = None
    experts: list[str] = field(default_factory=list)
    # Mixing weights: thompson, gating, meta-learner
    alpha: float = 0.34  # thompson weight
    beta: float = 0.33   # gating weight
    gamma: float = 0.33  # meta-learner weight

    def predict(
        self,
        expert_probs: dict[str, float],
        ctx: tuple[str, ...],
        context_features: np.ndarray | None = None,
        p_meta: float | None = None,
        n_eff: float | None = None,
    ) -> float:
        """Produce final probability from all available sub-ensembles.

        Gracefully degrades: if a sub-ensemble is unavailable, its weight
        is redistributed to the others.
        """
        predictions: list[tuple[float, float]] = []  # (weight, prediction)

        # 1. Thompson Sampling
        if self.thompson is not None:
            p_ts = self.thompson.predict(expert_probs, ctx, deterministic=True)
            predictions.append((self.alpha, p_ts))

        # 2. Gating model
        if self.gating is not None and self.gating.is_fitted and context_features is not None:
            gating_weights = self.gating.predict_weights_single(context_features)
            p_gating = _logit_average(expert_probs, gating_weights)
            predictions.append((self.beta, p_gating))

        # 3. Meta-learner (passed in as pre-computed probability)
        if p_meta is not None:
            predictions.append((self.gamma, p_meta))

        if not predictions:
            # Ultimate fallback: simple average of expert probs
            valid = [p for p in expert_probs.values() if p is not None]
            return float(np.mean(valid)) if valid else 0.5

        # Normalize mixing weights
        total_w = sum(w for w, _ in predictions)
        if total_w < EPS:
            return 0.5

        # Combine in logit space for better calibration
        logit_sum = 0.0
        for w, p in predictions:
            p_clip = float(np.clip(p, EPS, 1.0 - EPS))
            logit_sum += (w / total_w) * np.log(p_clip / (1.0 - p_clip))

        return float(1.0 / (1.0 + np.exp(-logit_sum)))

    def update_thompson(
        self,
        expert_probs: dict[str, float],
        y: int,
        ctx: tuple[str, ...],
    ) -> None:
        """Online update for the Thompson Sampling component."""
        if self.thompson is not None:
            self.thompson.update(expert_probs, y, ctx)

    def fit_mixing(
        self,
        expert_probs_list: list[dict[str, float]],
        ctx_list: list[tuple[str, ...]],
        context_features_list: np.ndarray | None,
        p_meta_list: np.ndarray | None,
        labels: np.ndarray,
    ) -> None:
        """Learn optimal mixing coefficients from validation data.

        Grid search over mixing weights to minimize log-loss.
        """
        labels = np.asarray(labels, dtype=np.float64).ravel()
        n = len(labels)
        if n < 20:
            return  # not enough data

        # Get sub-ensemble predictions for each sample
        p_thompson = np.full(n, np.nan)
        p_gating = np.full(n, np.nan)
        p_meta = np.full(n, np.nan)

        for i in range(n):
            ep = expert_probs_list[i]
            ctx = ctx_list[i]

            if self.thompson is not None:
                p_thompson[i] = self.thompson.predict(ep, ctx, deterministic=True)

            if self.gating is not None and self.gating.is_fitted and context_features_list is not None:
                gw = self.gating.predict_weights_single(context_features_list[i])
                p_gating[i] = _logit_average(ep, gw)

            if p_meta_list is not None:
                p_meta[i] = p_meta_list[i]

        # Grid search over mixing weights
        best_loss = float("inf")
        best_mix = (self.alpha, self.beta, self.gamma)
        grid = np.arange(0.0, 1.05, 0.1)

        for a in grid:
            for b in grid:
                g = 1.0 - a - b
                if g < -0.01:
                    continue
                g = max(g, 0.0)

                combined = np.zeros(n)
                w_total = 0.0

                if not np.all(np.isnan(p_thompson)):
                    valid = ~np.isnan(p_thompson)
                    combined[valid] += a * np.clip(p_thompson[valid], EPS, 1.0 - EPS)
                    w_total += a

                if not np.all(np.isnan(p_gating)):
                    valid = ~np.isnan(p_gating)
                    combined[valid] += b * np.clip(p_gating[valid], EPS, 1.0 - EPS)
                    w_total += b

                if not np.all(np.isnan(p_meta)):
                    valid = ~np.isnan(p_meta)
                    combined[valid] += g * np.clip(p_meta[valid], EPS, 1.0 - EPS)
                    w_total += g

                if w_total < EPS:
                    continue

                combined = combined / w_total
                combined = np.clip(combined, EPS, 1.0 - EPS)

                # Log-loss
                loss = -np.mean(
                    labels * np.log(combined) + (1.0 - labels) * np.log(1.0 - combined)
                )

                if loss < best_loss:
                    best_loss = loss
                    best_mix = (a, b, g)

        self.alpha, self.beta, self.gamma = best_mix

    def get_mixing_weights(self) -> dict[str, float]:
        return {
            "thompson": self.alpha,
            "gating": self.beta,
            "meta_learner": self.gamma,
        }

    def to_state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "type": "hybrid",
            "experts": self.experts,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
        if self.thompson is not None:
            state["thompson"] = self.thompson.to_state_dict()
        return state

    @classmethod
    def from_components(
        cls,
        thompson: ThompsonSamplingEnsembler | None = None,
        gating: GatingModel | None = None,
        experts: list[str] | None = None,
        alpha: float = 0.34,
        beta: float = 0.33,
        gamma: float = 0.33,
    ) -> "HybridEnsembleCombiner":
        return cls(
            thompson=thompson,
            gating=gating,
            experts=experts or [],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
