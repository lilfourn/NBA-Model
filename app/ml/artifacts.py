from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = None  # type: ignore[assignment]


def load_joblib_artifact(path: str | Path, *, strict_sklearn_version: bool = True) -> Any:
    """
    Load a joblib artifact while trapping sklearn version-mismatch warnings.

    strict_sklearn_version=True converts mismatch warnings into RuntimeError
    so callers can skip stale artifacts deterministically.
    """
    path_str = str(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = joblib.load(path_str)

    if not strict_sklearn_version:
        return payload

    for entry in caught:
        message = str(entry.message)
        is_version_warning = bool(
            InconsistentVersionWarning is not None and isinstance(entry.message, InconsistentVersionWarning)
        )
        is_text_match = "Trying to unpickle estimator" in message and "from version" in message
        if is_version_warning or is_text_match:
            raise RuntimeError(f"Incompatible sklearn artifact '{path_str}': {message}")

    return payload


def latest_compatible_joblib_path(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob(pattern), reverse=True)
    for candidate in candidates:
        try:
            load_joblib_artifact(candidate)
        except Exception:  # noqa: BLE001
            continue
        return candidate
    return None
