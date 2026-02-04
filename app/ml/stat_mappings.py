from __future__ import annotations

import math
import re
from typing import Any

NON_ALNUM = re.compile(r"[^a-z0-9]")


# Normalized stat_type -> DB column components to sum.
STAT_COMPONENTS: dict[str, list[str]] = {
    # Core single stats.
    "points": ["points"],
    "rebounds": ["rebounds"],
    "offensiverebounds": ["oreb"],
    "defensiverebounds": ["dreb"],
    "assists": ["assists"],
    "steals": ["steals"],
    "blocks": ["blocks"],
    "blockedshots": ["blocks"],
    "turnovers": ["turnovers"],
    "personalfouls": ["pf"],
    # Shooting stats.
    "3ptmade": ["fg3m"],
    "3pt": ["fg3m"],
    "fg3m": ["fg3m"],
    "3ptattempted": ["fg3a"],
    "fg3a": ["fg3a"],
    "fgmade": ["fgm"],
    "fgm": ["fgm"],
    "fgattempted": ["fga"],
    "fga": ["fga"],
    "freethrowsmade": ["ftm"],
    "ftm": ["ftm"],
    "freethrowsattempted": ["fta"],
    "fta": ["fta"],
    # Common combos.
    "ptsrebs": ["points", "rebounds"],
    "ptsasts": ["points", "assists"],
    "rebsasts": ["rebounds", "assists"],
    "ptsrebsasts": ["points", "rebounds", "assists"],
    "pra": ["points", "rebounds", "assists"],
    "blkstl": ["blocks", "steals"],
    "stlblk": ["blocks", "steals"],
    "blksstls": ["blocks", "steals"],
    # PrizePicks "(Combo)" stat types (map to the underlying stat).
    "pointscombo": ["points"],
    "reboundscombo": ["rebounds"],
    "assistscombo": ["assists"],
    "3ptmadecombo": ["fg3m"],
    "offensivereboundscombo": ["oreb"],
    "defensivereboundscombo": ["dreb"],
    "personalfoulscombo": ["pf"],
}


# Normalized stat_type -> (base_col, sub_col) meaning value = base_col - sub_col.
SPECIAL_DIFFS: dict[str, tuple[str, str]] = {
    "twopointersmade": ("fgm", "fg3m"),
    "twopointersattempted": ("fga", "fg3a"),
}


def normalize_stat_type(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip().lower()
    if not text_value:
        return None
    normalized = NON_ALNUM.sub("", text_value)
    return normalized or None


def stat_components(stat_type: Any) -> list[str] | None:
    normalized = normalize_stat_type(stat_type)
    if not normalized:
        return None
    return STAT_COMPONENTS.get(normalized)


def stat_diff_components(stat_type: Any) -> tuple[str, str] | None:
    normalized = normalize_stat_type(stat_type)
    if not normalized:
        return None
    return SPECIAL_DIFFS.get(normalized)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def stat_value_from_row(stat_type: Any, row: Any) -> float | None:
    """
    Compute the stat total from a row that has DB columns (points, rebounds, fg3m, ...).
    Handles special derived stats like Two Pointers Made = FGM - FG3M.
    """
    diff = stat_diff_components(stat_type)
    if diff is not None:
        base_col, sub_col = diff
        base = _to_float(row.get(base_col))
        sub = _to_float(row.get(sub_col))
        if base is None or sub is None:
            return None
        return base - sub

    components = stat_components(stat_type)
    if not components:
        return None
    total = 0.0
    for component in components:
        value = _to_float(row.get(component))
        if value is None:
            return None
        total += value
    return total


STAT_COLUMNS: list[str] = sorted(
    {
        *{col for cols in STAT_COMPONENTS.values() for col in cols},
        *{col for diff in SPECIAL_DIFFS.values() for col in diff},
    }
)
