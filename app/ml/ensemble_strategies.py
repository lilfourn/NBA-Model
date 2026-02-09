"""Ensemble strategies for backtesting different combination methods.

Pure-math module — no DB/IO. All strategies share the same interface:
- combine(expert_probs) -> float
- update(expert_probs, outcome) -> None  (for online strategies)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

EXPERT_PROB_FLOOR = 0.25
EXPERT_PROB_CEIL = 0.75
ROLLING_WINDOWS = [50, 100, 200]


def _logit(p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _clip(p: float) -> float:
    return max(EXPERT_PROB_FLOOR, min(EXPERT_PROB_CEIL, p))


def _available_probs(expert_probs: dict[str, float | None]) -> list[float]:
    return [_clip(v) for v in expert_probs.values() if v is not None]


def _logit_avg(probs: list[float]) -> float:
    if not probs:
        return 0.5
    return _sigmoid(sum(_logit(p) for p in probs) / len(probs))


# --- Metrics ---


def _logloss(y: int, p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def rolling_metrics(
    probs: np.ndarray, labels: np.ndarray, window: int = 50
) -> dict[str, Any]:
    if len(probs) < window:
        return {}
    recent_p = probs[-window:]
    recent_y = labels[-window:]
    picks = (recent_p >= 0.5).astype(int)
    accuracy = float((picks == recent_y).mean())
    avg_ll = float(
        np.mean([_logloss(int(y), float(p)) for y, p in zip(recent_y, recent_p)])
    )
    brier = float(np.mean((recent_p - recent_y) ** 2))
    return {
        "rolling_accuracy": round(accuracy, 4),
        "rolling_logloss": round(avg_ll, 4),
        "rolling_brier": round(brier, 4),
        "n": int(len(recent_p)),
    }


def multi_window_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    windows: list[int] | None = None,
) -> dict[str, Any]:
    windows = windows or ROLLING_WINDOWS
    result: dict[str, Any] = {}
    for w in windows:
        m = rolling_metrics(probs, labels, window=w)
        if m:
            result[f"last_{w}"] = m
    if len(probs) > 0:
        picks = (probs >= 0.5).astype(int)
        result["all_time"] = {
            "rolling_accuracy": round(float((picks == labels).mean()), 4),
            "rolling_logloss": round(
                float(
                    np.mean([_logloss(int(y), float(p)) for y, p in zip(labels, probs)])
                ),
                4,
            ),
            "rolling_brier": round(float(np.mean((probs - labels) ** 2)), 4),
            "n": int(len(probs)),
        }
    return result


# --- Strategy interface ---


class EnsembleStrategy(ABC):
    name: str

    @abstractmethod
    def combine(self, expert_probs: dict[str, float | None]) -> float: ...

    def update(
        self,
        expert_probs: dict[str, float | None],
        outcome: int,
        stat_type: str = "",
    ) -> None:
        pass


# --- Strategies ---


class LogitAvgStrategy(EnsembleStrategy):
    name = "logit_avg"

    def __init__(self, experts: list[str] | None = None):
        self.experts = experts

    def combine(self, expert_probs: dict[str, float | None]) -> float:
        if self.experts:
            probs = [
                _clip(v) for e in self.experts if (v := expert_probs.get(e)) is not None
            ]
        else:
            probs = _available_probs(expert_probs)
        return _logit_avg(probs)


class StackingStrategy(EnsembleStrategy):
    name = "stacking"

    def __init__(self, stacking_model: Any):
        self._model = stacking_model

    def combine(self, expert_probs: dict[str, float | None]) -> float:
        from app.ml.stacking import EXPERT_COLS, predict_stacking

        clipped = {
            k: _clip(v) if v is not None else None for k, v in expert_probs.items()
        }
        return predict_stacking(self._model, {e: clipped.get(e) for e in EXPERT_COLS})


class RecencyWeightedStrategy(EnsembleStrategy):
    name = "recency_weighted"

    def __init__(self, window: int = 100, temperature: float = 5.0):
        self.window = window
        self.temperature = temperature
        self._history: list[tuple[dict[str, float | None], int]] = []

    def _expert_accuracies(self) -> dict[str, float]:
        recent = self._history[-self.window :]
        counts: dict[str, int] = defaultdict(int)
        correct: dict[str, int] = defaultdict(int)
        for probs, outcome in recent:
            for e, p in probs.items():
                if p is None:
                    continue
                counts[e] += 1
                pred = 1 if _clip(p) >= 0.5 else 0
                if pred == outcome:
                    correct[e] += 1
        return {e: correct[e] / counts[e] for e in counts if counts[e] > 0}

    def combine(self, expert_probs: dict[str, float | None]) -> float:
        if len(self._history) < 10:
            return _logit_avg(_available_probs(expert_probs))

        accs = self._expert_accuracies()
        available = {
            e: _clip(v) for e, v in expert_probs.items() if v is not None and e in accs
        }
        if not available:
            return _logit_avg(_available_probs(expert_probs))

        scores = np.array([accs[e] for e in available])
        exp_scores = np.exp(self.temperature * (scores - scores.max()))
        weights = exp_scores / exp_scores.sum()

        logits = np.array([_logit(p) for p in available.values()])
        return _sigmoid(float(np.dot(weights, logits)))

    def update(
        self,
        expert_probs: dict[str, float | None],
        outcome: int,
        stat_type: str = "",
    ) -> None:
        self._history.append((dict(expert_probs), outcome))


class PerStatStrategy(EnsembleStrategy):
    name = "per_stat"

    def __init__(self, window: int = 100, min_history: int = 20):
        self.window = window
        self.min_history = min_history
        self._history: dict[str, list[tuple[dict[str, float | None], int]]] = (
            defaultdict(list)
        )
        self._current_stat_type = ""

    def _best_expert(self, stat_type: str) -> str | None:
        hist = self._history.get(stat_type, [])
        recent = hist[-self.window :]
        if len(recent) < self.min_history:
            return None

        counts: dict[str, int] = defaultdict(int)
        correct: dict[str, int] = defaultdict(int)
        for probs, outcome in recent:
            for e, p in probs.items():
                if p is None:
                    continue
                counts[e] += 1
                if (1 if _clip(p) >= 0.5 else 0) == outcome:
                    correct[e] += 1

        best_e, best_acc = None, -1.0
        for e in counts:
            if counts[e] < self.min_history // 2:
                continue
            acc = correct[e] / counts[e]
            if acc > best_acc:
                best_acc = acc
                best_e = e
        return best_e

    def combine(self, expert_probs: dict[str, float | None]) -> float:
        best = self._best_expert(self._current_stat_type)
        if best and expert_probs.get(best) is not None:
            return _clip(expert_probs[best])  # type: ignore[arg-type]
        return _logit_avg(_available_probs(expert_probs))

    def update(
        self,
        expert_probs: dict[str, float | None],
        outcome: int,
        stat_type: str = "",
    ) -> None:
        self._current_stat_type = stat_type
        self._history[stat_type].append((dict(expert_probs), outcome))


class TopKStrategy(EnsembleStrategy):
    name = "top_k"

    def __init__(self, k: int = 3, window: int = 100, min_history: int = 20):
        self.k = k
        self.window = window
        self.min_history = min_history
        self._history: list[tuple[dict[str, float | None], int]] = []

    def _top_k_experts(self) -> list[str] | None:
        recent = self._history[-self.window :]
        if len(recent) < self.min_history:
            return None

        counts: dict[str, int] = defaultdict(int)
        correct: dict[str, int] = defaultdict(int)
        for probs, outcome in recent:
            for e, p in probs.items():
                if p is None:
                    continue
                counts[e] += 1
                if (1 if _clip(p) >= 0.5 else 0) == outcome:
                    correct[e] += 1

        ranked = sorted(
            (e for e in counts if counts[e] >= self.min_history // 2),
            key=lambda e: correct[e] / counts[e],
            reverse=True,
        )
        return ranked[: self.k] if ranked else None

    def combine(self, expert_probs: dict[str, float | None]) -> float:
        top = self._top_k_experts()
        if top:
            probs = [_clip(v) for e in top if (v := expert_probs.get(e)) is not None]
            if probs:
                return _logit_avg(probs)
        return _logit_avg(_available_probs(expert_probs))

    def update(
        self,
        expert_probs: dict[str, float | None],
        outcome: int,
        stat_type: str = "",
    ) -> None:
        self._history.append((dict(expert_probs), outcome))


# --- Factory ---


def build_strategies(
    stacking_model: Any = None,
    experts: list[str] | None = None,
) -> dict[str, EnsembleStrategy]:
    strategies: dict[str, EnsembleStrategy] = {
        "logit_avg": LogitAvgStrategy(experts=experts),
        "recency_weighted": RecencyWeightedStrategy(),
        "per_stat": PerStatStrategy(),
        "top_k": TopKStrategy(),
    }
    if stacking_model is not None:
        strategies["stacking"] = StackingStrategy(stacking_model)
    return strategies
