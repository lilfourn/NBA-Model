"""Ensemble weight history logging.

Appends a timestamped snapshot of ensemble weights to a JSONL file
after each training run. Used for visualizing weight evolution over time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HISTORY_PATH = "data/reports/weight_history.jsonl"


def log_weights(
    *,
    hedge_weights: dict[str, dict[str, float]] | None = None,
    thompson_weights: dict[str, dict[str, float]] | None = None,
    gating_weights: dict[str, float] | None = None,
    mixing_weights: dict[str, float] | None = None,
    n_updates: int = 0,
    path: str = DEFAULT_HISTORY_PATH,
) -> None:
    """Append a weight snapshot to the JSONL history file."""
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_updates": n_updates,
    }

    # Store a summary of hedge weights (global context only to keep compact)
    if hedge_weights is not None:
        # Average across contexts for a summary view
        all_experts: set[str] = set()
        for ctx_weights in hedge_weights.values():
            all_experts.update(ctx_weights.keys())
        if all_experts:
            avg_weights: dict[str, float] = {}
            for expert in sorted(all_experts):
                vals = [cw.get(expert, 0.0) for cw in hedge_weights.values()]
                avg_weights[expert] = round(sum(vals) / max(len(vals), 1), 4)
            entry["hedge_avg"] = avg_weights

    if thompson_weights is not None:
        entry["thompson_avg"] = {
            k: round(v, 4) for k, v in thompson_weights.items()
        }

    if gating_weights is not None:
        entry["gating"] = {
            k: round(v, 4) for k, v in gating_weights.items()
        }

    if mixing_weights is not None:
        entry["mixing"] = {
            k: round(v, 4) for k, v in mixing_weights.items()
        }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_weight_history(path: str = DEFAULT_HISTORY_PATH) -> list[dict[str, Any]]:
    """Load all weight history entries."""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
