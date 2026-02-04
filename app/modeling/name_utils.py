from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from app.core.config import settings

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_OVERRIDES: dict[str, str] | None = None


def _load_overrides() -> dict[str, str]:
    global _OVERRIDES  # noqa: PLW0603
    if _OVERRIDES is not None:
        return _OVERRIDES
    path = Path(settings.player_name_overrides_path)
    if not path.exists():
        _OVERRIDES = {}
        return _OVERRIDES
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _OVERRIDES = {}
        return _OVERRIDES
    overrides: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            overrides[key.strip().lower()] = value.strip().lower()
    _OVERRIDES = overrides
    return _OVERRIDES


def normalize_player_name(name: str | None) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    parts = [part.lower() for part in text.split() if part.strip()]
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    normalized = " ".join(parts)
    overrides = _load_overrides()
    return overrides.get(normalized, normalized)
