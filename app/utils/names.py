from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

NON_ALNUM = re.compile(r"[^a-z0-9]")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_OVERRIDES: dict[str, str] | None = None
_OVERRIDES_PATH: str | None = None


def _normalize_raw(name: str) -> str:
    """Normalize without override application (used to build override map)."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    parts = [part.lower() for part in text.split() if part.strip()]
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _load_overrides(path: str | None = None) -> dict[str, str]:
    global _OVERRIDES, _OVERRIDES_PATH  # noqa: PLW0603
    if _OVERRIDES is not None and _OVERRIDES_PATH == path:
        return _OVERRIDES
    _OVERRIDES_PATH = path
    if not path:
        from app.core.config import settings
        path = settings.player_name_overrides_path
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
                # Normalize both sides through the same pipeline (without overrides).
                norm_key = _normalize_raw(key)
                norm_value = _normalize_raw(value)
                if norm_key and norm_value and norm_key != norm_value:
                    overrides[norm_key] = norm_value
    _OVERRIDES = overrides
    return _OVERRIDES


def normalize_player_name(name: str | None) -> str:
    """Canonical player name: NFKD → strip diacritics → lowercase → strip suffixes → apply overrides.

    Returns a human-readable normalized name (e.g. "lebron james").
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    parts = [part.lower() for part in text.split() if part.strip()]
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    normalized = " ".join(parts)
    overrides = _load_overrides()
    return overrides.get(normalized, normalized)


def normalize_name(value: Any) -> str | None:
    """Name key for DB indexing: strips all non-alnum after canonical normalization.

    Returns a compact key like "lebronjames" or None.
    """
    if value is None:
        return None
    canonical = normalize_player_name(str(value))
    if not canonical:
        return None
    key = NON_ALNUM.sub("", canonical)
    return key or None
