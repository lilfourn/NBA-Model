"""Team abbreviation normalization using data/team_abbrev_overrides.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_OVERRIDES: dict[str, str] | None = None
_OVERRIDES_PATH: str | None = None


def _load_overrides(path: str | None = None) -> dict[str, str]:
    global _OVERRIDES, _OVERRIDES_PATH  # noqa: PLW0603
    if _OVERRIDES is not None and _OVERRIDES_PATH == path:
        return _OVERRIDES
    _OVERRIDES_PATH = path
    if not path:
        from app.core.config import settings
        path = settings.team_abbrev_overrides_path
    p = Path(path)
    if not p.exists():
        _OVERRIDES = {}
        return _OVERRIDES
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _OVERRIDES = {}
        return _OVERRIDES
    overrides: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str):
                overrides[key.strip().upper()] = value.strip().upper()
    _OVERRIDES = overrides
    return _OVERRIDES


def normalize_team_abbrev(abbrev: Any) -> str | None:
    """Normalize a team abbreviation using the overrides file.

    Returns the canonical abbreviation (e.g. "BKN" not "BRK") or None if input is falsy.
    """
    if abbrev is None:
        return None
    text = str(abbrev).strip().upper()
    if not text:
        return None
    overrides = _load_overrides()
    return overrides.get(text, text)
