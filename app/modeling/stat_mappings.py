from __future__ import annotations

from typing import Any

STAT_TYPE_MAP: dict[str, list[str]] = {
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
    "Steals": ["STL"],
    "Blocked Shots": ["BLK"],
    "Turnovers": ["TOV"],
    "Personal Fouls": ["PF"],
    "Offensive Rebounds": ["OREB"],
    "Defensive Rebounds": ["DREB"],
    "3-PT Made": ["FG3M"],
    "3-PT Attempted": ["FG3A"],
    "FG Made": ["FGM"],
    "FG Attempted": ["FGA"],
    "Free Throws Made": ["FTM"],
    "Free Throws Attempted": ["FTA"],
    "Pts+Rebs": ["PTS", "REB"],
    "Pts+Asts": ["PTS", "AST"],
    "Rebs+Asts": ["REB", "AST"],
    "Pts+Rebs+Asts": ["PTS", "REB", "AST"],
    "Blks+Stls": ["BLK", "STL"],
}

SPECIAL_STATS = {
    "Two Pointers Made": ("FGM", "FG3M"),
    "Two Pointers Attempted": ("FGA", "FG3A"),
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def stat_value(stat_type: str, stats: dict[str, Any]) -> float | None:
    if stat_type in SPECIAL_STATS:
        base_key, sub_key = SPECIAL_STATS[stat_type]
        base = _to_float(stats.get(base_key))
        sub = _to_float(stats.get(sub_key))
        if base is None or sub is None:
            return None
        return base - sub

    keys = STAT_TYPE_MAP.get(stat_type)
    if not keys:
        return None
    total = 0.0
    for key in keys:
        value = _to_float(stats.get(key))
        if value is None:
            return None
        total += value
    return total
