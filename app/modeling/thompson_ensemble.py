"""Thompson Sampling ensemble for expert weight selection.

Beta-Bernoulli Thompson Sampling per expert per context bucket.
Natural exploration via sampling variance — no learning rate to tune.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-6
STATE_VERSION = 1


def _ctx_key(ctx: tuple[str, ...]) -> str:
    return json.dumps(list(ctx))


@dataclass
class ThompsonSamplingEnsembler:
    """Beta-Bernoulli Thompson Sampling per expert per context bucket.

    Each expert in each context has a Beta(alpha, beta) posterior.
    At prediction time, we sample from each posterior and normalize
    the samples to produce a weight simplex.

    Updates use a continuous reward: reward = 1 - |y - p| so that
    experts who predict closer to the true outcome get more credit.
    """

    experts: list[str]
    alpha: dict[str, dict[str, float]] = field(default_factory=dict)
    beta: dict[str, dict[str, float]] = field(default_factory=dict)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    max_weight: dict[str, float] = field(default_factory=dict)
    n_updates: int = 0
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42))

    def _ensure_ctx(self, ctx_key: str) -> None:
        if ctx_key not in self.alpha:
            self.alpha[ctx_key] = {e: self.prior_alpha for e in self.experts}
            self.beta[ctx_key] = {e: self.prior_beta for e in self.experts}

    def _sample_weights(self, ctx_key: str) -> dict[str, float]:
        """Sample from each expert's Beta posterior, normalize to simplex."""
        self._ensure_ctx(ctx_key)
        samples = {}
        for e in self.experts:
            a = max(self.alpha[ctx_key][e], EPS)
            b = max(self.beta[ctx_key][e], EPS)
            samples[e] = float(self.rng.beta(a, b))

        # Apply max_weight caps
        for e, cap in self.max_weight.items():
            if e in samples:
                samples[e] = min(samples[e], cap)

        total = sum(samples.values())
        if total < EPS:
            uniform = 1.0 / len(self.experts)
            return {e: uniform for e in self.experts}
        return {e: v / total for e, v in samples.items()}

    def _mean_weights(self, ctx_key: str) -> dict[str, float]:
        """Deterministic weights from Beta means (for stable inference)."""
        self._ensure_ctx(ctx_key)
        means = {}
        for e in self.experts:
            a = self.alpha[ctx_key][e]
            b = self.beta[ctx_key][e]
            means[e] = a / (a + b) if (a + b) > EPS else 0.5

        for e, cap in self.max_weight.items():
            if e in means:
                means[e] = min(means[e], cap)

        total = sum(means.values())
        if total < EPS:
            uniform = 1.0 / len(self.experts)
            return {e: uniform for e in self.experts}
        return {e: v / total for e, v in means.items()}

    def predict(
        self,
        expert_probs: dict[str, float],
        ctx: tuple[str, ...],
        *,
        deterministic: bool = False,
    ) -> float:
        """Weighted logit-average of expert probabilities.

        Parameters
        ----------
        expert_probs : dict mapping expert name → P(over)
        ctx : context tuple (stat_type, regime, n_eff_bucket)
        deterministic : if True use Beta means; if False sample weights
        """
        ctx_key = _ctx_key(ctx)
        weights = self._mean_weights(ctx_key) if deterministic else self._sample_weights(ctx_key)

        logit_sum = 0.0
        w_sum = 0.0
        for e in self.experts:
            p = expert_probs.get(e)
            if p is None:
                continue
            p = float(np.clip(p, EPS, 1.0 - EPS))
            w = weights.get(e, 0.0)
            logit_sum += w * np.log(p / (1.0 - p))
            w_sum += w

        if w_sum < EPS:
            return 0.5
        avg_logit = logit_sum / w_sum
        return float(1.0 / (1.0 + np.exp(-avg_logit)))

    def update(
        self,
        expert_probs: dict[str, float],
        y: int,
        ctx: tuple[str, ...],
    ) -> None:
        """Update Beta posteriors with continuous reward.

        reward = 1 - |y - p|  (closer prediction = higher reward)
        alpha += reward, beta += (1 - reward)
        """
        ctx_key = _ctx_key(ctx)
        self._ensure_ctx(ctx_key)
        y_f = float(y)

        for e in self.experts:
            p = expert_probs.get(e)
            if p is None:
                continue
            p = float(np.clip(p, EPS, 1.0 - EPS))
            reward = 1.0 - abs(y_f - p)
            self.alpha[ctx_key][e] += reward
            self.beta[ctx_key][e] += (1.0 - reward)

        self.n_updates += 1

    def get_weights(self, ctx: tuple[str, ...], *, deterministic: bool = True) -> dict[str, float]:
        """Return current weights for a context (for logging/inspection)."""
        ctx_key = _ctx_key(ctx)
        return self._mean_weights(ctx_key) if deterministic else self._sample_weights(ctx_key)

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "type": "thompson",
            "experts": self.experts,
            "alpha": self.alpha,
            "beta": self.beta,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "max_weight": self.max_weight,
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ThompsonSamplingEnsembler":
        return cls(
            experts=state["experts"],
            alpha=state.get("alpha", {}),
            beta=state.get("beta", {}),
            prior_alpha=state.get("prior_alpha", 1.0),
            prior_beta=state.get("prior_beta", 1.0),
            max_weight=state.get("max_weight", {}),
            n_updates=state.get("n_updates", 0),
        )

    def save(self, path: str) -> None:
        import json as _json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(_json.dumps(self.to_state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ThompsonSamplingEnsembler":
        import json as _json
        from pathlib import Path
        state = _json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(state)
