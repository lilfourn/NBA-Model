from __future__ import annotations

import math
from typing import Any

STAT_TYPE_MAP: dict[str, list[str]] = {
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
    "Steals": ["STL"],
    "Blocked Shots": ["BLK"],
    "Dunks": ["DUNKS"],
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

FANTASY_SCORE_WEIGHTS: dict[str, float] = {
    "PTS": 1.0,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 3.0,
    "BLK": 3.0,
    "TOV": -1.0,
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    return None


def stat_value(stat_type: str, stats: dict[str, Any]) -> float | None:
    if stat_type == "Fantasy Score":
        total = 0.0
        for key, weight in FANTASY_SCORE_WEIGHTS.items():
            value = _to_float(stats.get(key))
            if value is None:
                return None
            total += float(weight) * float(value)
        return total

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
