from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class Projection:
    projection_id: str
    player_id: str
    player_name: str
    stat_type: str
    line_score: float
    start_time: datetime | None
    game_id: str | None
    event_type: str | None
    projection_type: str | None
    trending_count: int | None
    is_today: bool | None
    is_combo: bool


@dataclass(frozen=True)
class PlayerGameLog:
    player_id: str | None
    player_name: str | None
    game_date: date | None
    stats: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    projection: Projection
    pick: str
    prob_over: float
    confidence: float
    mean: float | None
    std: float | None
    model_version: str
    details: dict[str, Any] | None = None
