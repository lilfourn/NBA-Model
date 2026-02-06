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
    p = float(p)
    if not math.isfinite(p):
        return 0.5
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
    - Weight floor/cap prevents collapse into a single expert
    - Adaptive eta decay prevents late-stage exponential divergence
    """

    # Anti-collapse constants
    MIN_WEIGHT_FLOOR = 0.02  # No expert below 2%
    MAX_WEIGHT_CAP = 0.45  # No single expert above 45%
    ETA_DECAY = 0.0005  # eta_t = eta_0 / (1 + decay * t)

    def __init__(
        self,
        experts: list[str],
        *,
        eta: float = 0.2,
        shrink_to_uniform: float = 0.05,
        max_weight: dict[str, float] | None = None,
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
        self.max_weight: dict[str, float] = dict(max_weight) if max_weight else {}
        self.weights: dict[tuple[str, str, str], dict[str, float]] = {}
        self._update_count: int = 0  # Tracks total updates for eta decay

    # OOF AUC-based quality priors — prevents weak experts from dragging
    # down ensemble during the slow online learning phase.
    # Updated from OOF analysis on 2145 rows (2026-02-05).
    QUALITY_PRIOR: dict[str, float] = {
        "p_lgbm": 0.753,
        "p_xgb": 0.747,
        "p_lr": 0.577,
        "p_nn": 0.554,
        "p_forecast_cal": 0.511,
    }

    def _init_weights(self) -> dict[str, float]:
        raw = {}
        for expert in self.experts:
            # Use AUC-0.5 as quality signal (0.5 = random)
            auc = self.QUALITY_PRIOR.get(expert, 0.55)
            raw[expert] = max(auc - 0.5, 0.01)
        total = sum(raw.values())
        return {e: v / total for e, v in raw.items()}

    def _get_weights(self, ctx_key: tuple[str, str, str]) -> dict[str, float]:
        if ctx_key not in self.weights:
            self.weights[ctx_key] = self._init_weights()
        return self.weights[ctx_key]

    def _clamp_weights(self, w: dict[str, float]) -> None:
        """Apply floor/cap constraints to prevent weight collapse."""
        n = len(self.experts)
        if n == 0:
            return
        floor = self.MIN_WEIGHT_FLOOR
        cap = self.MAX_WEIGHT_CAP
        # Apply floor
        for expert in self.experts:
            val = float(w.get(expert, 0.0))
            if val < floor:
                w[expert] = floor
        # Apply per-expert caps (both global cap and user-specified max_weight)
        for expert in self.experts:
            val = float(w.get(expert, 0.0))
            effective_cap = cap
            user_cap = self.max_weight.get(expert)
            if user_cap is not None:
                effective_cap = min(cap, float(user_cap))
            if val > effective_cap:
                w[expert] = effective_cap
        # Renormalize
        total = sum(float(w.get(e, 0.0)) for e in self.experts)
        if total > 0 and math.isfinite(total):
            for expert in self.experts:
                w[expert] = float(w.get(expert, 0.0)) / total

    def predict(self, expert_probs: dict[str, float | None], ctx: Context) -> float:
        ctx_key = ctx.key()
        w = self._get_weights(ctx_key)
        # Apply floor/cap at inference to guard against stale collapsed weights
        self._clamp_weights(w)

        avail = []
        for expert in self.experts:
            if expert not in expert_probs:
                continue
            p = expert_probs[expert]
            if p is None:
                continue
            try:
                p_val = float(p)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p_val):
                continue
            avail.append(expert)
        if not avail:
            return 0.5

        weights = []
        for expert in avail:
            w_val = float(w.get(expert, 0.0))
            if not math.isfinite(w_val) or w_val < 0:
                w_val = 0.0
            weights.append(w_val)

        wsum = sum(weights)
        if (not math.isfinite(wsum)) or wsum <= 0:
            # If weights got corrupted, fall back to uniform over available experts.
            z = sum(logit(float(expert_probs[expert])) for expert in avail) / float(
                len(avail)
            )
            return sigmoid(z)

        z = 0.0
        for expert, w_val in zip(avail, weights):
            if w_val <= 0:
                continue
            z += (w_val / (wsum + 1e-12)) * logit(float(expert_probs[expert]))
        return sigmoid(z)

    def update(
        self, expert_probs: dict[str, float | None], y: int, ctx: Context
    ) -> None:
        ctx_key = ctx.key()
        w = self._get_weights(ctx_key)
        self._update_count += 1

        if self.shrink > 0:
            u = 1.0 / float(len(self.experts))
            for expert in self.experts:
                w[expert] = (1.0 - self.shrink) * float(
                    w.get(expert, 0.0)
                ) + self.shrink * u

        # Adaptive eta: decay over time to prevent late-stage collapse
        eta_t = self.eta / (1.0 + self.ETA_DECAY * self._update_count)

        for expert in self.experts:
            p = expert_probs.get(expert)
            if p is None:
                continue
            try:
                p_val = float(p)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p_val):
                continue
            loss = logloss(int(y), p_val)
            w[expert] = float(w.get(expert, 0.0)) * math.exp(-eta_t * loss)

        total = sum(float(w.get(expert, 0.0)) for expert in self.experts)
        if (not math.isfinite(total)) or total <= 0:
            self.weights[ctx_key] = self._init_weights()
            return
        for expert in self.experts:
            w[expert] = float(w.get(expert, 0.0)) / total

        # Apply floor/cap to prevent collapse
        self._clamp_weights(w)

    def weights_for_context(self, ctx: Context) -> dict[str, float]:
        w = self._get_weights(ctx.key())
        return {expert: float(w.get(expert, 0.0)) for expert in self.experts}

    def iter_contexts(self) -> Iterable[tuple[tuple[str, str, str], dict[str, float]]]:
        for ctx_key, weights in self.weights.items():
            yield ctx_key, {
                expert: float(weights.get(expert, 0.0)) for expert in self.experts
            }

    def to_state_dict(self) -> dict[str, Any]:
        weights_out: dict[str, dict[str, float]] = {}
        for ctx_key, weights in self.weights.items():
            weights_out[_ctx_key_to_str(ctx_key)] = {
                expert: float(weights.get(expert, 0.0)) for expert in self.experts
            }

        state: dict[str, Any] = {
            "version": STATE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experts": list(self.experts),
            "eta": float(self.eta),
            "shrink_to_uniform": float(self.shrink),
            "update_count": self._update_count,
            "weights": weights_out,
        }
        if self.max_weight:
            state["max_weight"] = {str(k): float(v) for k, v in self.max_weight.items()}
        return state

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
        max_weight_raw = state.get("max_weight")
        max_weight: dict[str, float] | None = None
        if isinstance(max_weight_raw, dict):
            max_weight = {str(k): float(v) for k, v in max_weight_raw.items()}
        inst = cls(experts, eta=eta, shrink_to_uniform=shrink, max_weight=max_weight)
        inst._update_count = int(state.get("update_count", 0))

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
                        value = float(weights_obj[expert])
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value) or value < 0:
                        w[expert] = 0.0
                    else:
                        w[expert] = value
            total = sum(w.values())
            if (not math.isfinite(total)) or total <= 0:
                w = inst._init_weights()
            else:
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
