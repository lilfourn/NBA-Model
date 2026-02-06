"""Load and apply expert inversion corrections from model health report.

When an expert is consistently inverted (1-p would be more accurate than p),
the scoring pipeline flips its probability before passing to the ensemble.

Usage:
    flags = load_inversion_flags()          # from local model_health.json
    flags = load_inversion_flags(engine)    # from DB artifact

    for expert, should_flip in flags.items():
        if should_flip and expert_probs.get(expert) is not None:
            expert_probs[expert] = 1.0 - expert_probs[expert]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Default local path for model_health.json
_DEFAULT_HEALTH_PATH = Path(
    os.environ.get("HEALTH_REPORT_PATH", "data/reports/model_health.json")
)

# Experts that should never be flipped (p_final is derived, not an input)
_SKIP_EXPERTS = {"p_final"}


def load_inversion_flags(
    engine_or_path: Any = None,
) -> dict[str, bool]:
    """Load inversion flags from model health report.

    Returns dict mapping expert name -> True if the expert should be inverted.
    Only returns True when BOTH inversion_improves_accuracy AND
    inversion_improves_logloss are True (strict condition).

    Args:
        engine_or_path: One of:
            - None: load from default local path
            - str/Path: load from explicit file path
            - SQLAlchemy Engine: load from DB artifact store
    """
    report = _load_report(engine_or_path)
    if not report:
        return {}

    expert_metrics = report.get("expert_metrics", {})
    flags: dict[str, bool] = {}

    for expert, data in expert_metrics.items():
        if expert in _SKIP_EXPERTS:
            continue
        if not isinstance(data, dict):
            continue
        inv = data.get("inversion_test", {})
        if not isinstance(inv, dict):
            continue
        acc_inv = bool(inv.get("inversion_improves_accuracy", False))
        ll_inv = bool(inv.get("inversion_improves_logloss", False))
        if acc_inv and ll_inv:
            flags[expert] = True

    return flags


def _load_report(engine_or_path: Any) -> dict | None:
    """Load model_health.json from local file, explicit path, or DB."""
    # Explicit path
    if isinstance(engine_or_path, (str, Path)):
        return _load_from_file(Path(engine_or_path))

    # SQLAlchemy engine — try DB artifact
    if engine_or_path is not None and hasattr(engine_or_path, "connect"):
        try:
            from app.ml.artifact_store import load_latest_artifact

            data = load_latest_artifact(engine_or_path, "model_health")
            if data:
                return json.loads(data.decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
        # Fall through to local file

    # Default local file
    return _load_from_file(_DEFAULT_HEALTH_PATH)


def _load_from_file(path: Path) -> dict | None:
    """Load JSON report from a local file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
