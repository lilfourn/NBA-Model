from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EPS = 1e-6
STATE_VERSION = 1


def clip01(p: float, *, eps: float = EPS) -> float:
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


def logit(p: float) -> float:
    p = clip01(float(p))
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logloss(y: int, p: float) -> float:
    p = clip01(float(p))
    y = int(y)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def neff_bucket(n_eff: float | None) -> str:
    if n_eff is None:
        return "neff_unknown"
    n_eff = float(n_eff)
    if n_eff < 5:
        return "neff_lt5"
    if n_eff < 15:
        return "neff_5_15"
    return "neff_ge15"


@dataclass(frozen=True)
class Context:
    stat_type: str
    is_live: bool
    n_eff: float | None

    def key(self) -> tuple[str, str, str]:
        return (
            str(self.stat_type or ""),
            "live" if self.is_live else "pregame",
            neff_bucket(self.n_eff),
        )


def _ctx_key_to_str(ctx_key: tuple[str, str, str]) -> str:
    return json.dumps(list(ctx_key), ensure_ascii=False, separators=(",", ":"))


def _ctx_key_from_str(value: str) -> tuple[str, str, str]:
    try:
        parts = json.loads(value)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise ValueError(f"Invalid context key: {value}") from exc
    if not isinstance(parts, list) or len(parts) != 3:
        raise ValueError(f"Invalid context key: {value}")
    return str(parts[0]), str(parts[1]), str(parts[2])


class ContextualHedgeEnsembler:
    """
    Online expert-advice learner (full-feedback).

    - Maintains multiplicative-weights per context bucket
    - Predicts by averaging expert logits (more stable than averaging probabilities)
    - Updates weights from realized outcomes using log loss
    """

    def __init__(
        self,
        experts: list[str],
        *,
        eta: float = 0.2,
        shrink_to_uniform: float = 0.01,
    ) -> None:
        if not experts:
            raise ValueError("experts must be non-empty")
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if shrink_to_uniform < 0 or shrink_to_uniform >= 1:
            raise ValueError("shrink_to_uniform must be in [0, 1)")

        self.experts = list(experts)
        self.eta = float(eta)
        self.shrink = float(shrink_to_uniform)
        self.weights: dict[tuple[str, str, str], dict[str, float]] = {}

    def _init_weights(self) -> dict[str, float]:
        w0 = 1.0 / float(len(self.experts))
        return {expert: w0 for expert in self.experts}

    def _get_weights(self, ctx_key: tuple[str, str, str]) -> dict[str, float]:
        if ctx_key not in self.weights:
            self.weights[ctx_key] = self._init_weights()
        return self.weights[ctx_key]

    def predict(self, expert_probs: dict[str, float | None], ctx: Context) -> float:
        ctx_key = ctx.key()
        w = self._get_weights(ctx_key)

        avail = [
            expert
            for expert in self.experts
            if expert in expert_probs and expert_probs[expert] is not None
        ]
        if not avail:
            return 0.5

        wsum = sum(float(w.get(expert, 0.0)) for expert in avail)
        if wsum <= 0:
            # If weights got corrupted, fall back to uniform over available experts.
            z = sum(logit(float(expert_probs[expert])) for expert in avail) / float(len(avail))
            return sigmoid(z)

        z = 0.0
        for expert in avail:
            z += (float(w.get(expert, 0.0)) / (wsum + 1e-12)) * logit(float(expert_probs[expert]))
        return sigmoid(z)

    def update(self, expert_probs: dict[str, float | None], y: int, ctx: Context) -> None:
        ctx_key = ctx.key()
        w = self._get_weights(ctx_key)

        if self.shrink > 0:
            u = 1.0 / float(len(self.experts))
            for expert in self.experts:
                w[expert] = (1.0 - self.shrink) * float(w.get(expert, 0.0)) + self.shrink * u

        for expert in self.experts:
            p = expert_probs.get(expert)
            if p is None:
                continue
            loss = logloss(int(y), float(p))
            w[expert] = float(w.get(expert, 0.0)) * math.exp(-self.eta * loss)

        total = sum(float(w.get(expert, 0.0)) for expert in self.experts)
        if total <= 0:
            self.weights[ctx_key] = self._init_weights()
            return
        for expert in self.experts:
            w[expert] = float(w.get(expert, 0.0)) / total

    def weights_for_context(self, ctx: Context) -> dict[str, float]:
        w = self._get_weights(ctx.key())
        return {expert: float(w.get(expert, 0.0)) for expert in self.experts}

    def iter_contexts(self) -> Iterable[tuple[tuple[str, str, str], dict[str, float]]]:
        for ctx_key, weights in self.weights.items():
            yield ctx_key, {expert: float(weights.get(expert, 0.0)) for expert in self.experts}

    def to_state_dict(self) -> dict[str, Any]:
        weights_out: dict[str, dict[str, float]] = {}
        for ctx_key, weights in self.weights.items():
            weights_out[_ctx_key_to_str(ctx_key)] = {
                expert: float(weights.get(expert, 0.0)) for expert in self.experts
            }

        return {
            "version": STATE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experts": list(self.experts),
            "eta": float(self.eta),
            "shrink_to_uniform": float(self.shrink),
            "weights": weights_out,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ContextualHedgeEnsembler:
        version = int(state.get("version") or 0)
        if version != STATE_VERSION:
            raise ValueError(f"Unsupported ensemble state version: {version}")

        experts_raw = state.get("experts")
        if not isinstance(experts_raw, list) or not experts_raw:
            raise ValueError("Invalid ensemble state: experts")
        experts = [str(e) for e in experts_raw]
        eta = float(state.get("eta") or 0.2)
        shrink = float(state.get("shrink_to_uniform") or 0.0)
        inst = cls(experts, eta=eta, shrink_to_uniform=shrink)

        weights_in = state.get("weights") or {}
        if not isinstance(weights_in, dict):
            return inst

        for ctx_key_str, weights_obj in weights_in.items():
            if not isinstance(ctx_key_str, str):
                continue
            ctx_key = _ctx_key_from_str(ctx_key_str)
            if not isinstance(weights_obj, dict):
                continue
            w = inst._init_weights()
            for expert in inst.experts:
                if expert in weights_obj:
                    try:
                        w[expert] = float(weights_obj[expert])
                    except (TypeError, ValueError):
                        continue
            total = sum(w.values())
            if total > 0:
                for expert in w:
                    w[expert] /= total
            inst.weights[ctx_key] = w

        return inst

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_state_dict()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> ContextualHedgeEnsembler:
        path = Path(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Invalid ensemble state file")
        return cls.from_state_dict(state)

