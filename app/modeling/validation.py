from __future__ import annotations

from typing import Iterable

from app.modeling.name_utils import normalize_player_name
from app.modeling.stat_mappings import stat_value
from app.modeling.types import PlayerGameLog, Projection


def find_game_log_for_projection(
    projection: Projection,
    game_logs: Iterable[PlayerGameLog],
) -> PlayerGameLog | None:
    if not projection.start_time:
        return None
    target_date = projection.start_time.date()
    player_key = normalize_player_name(projection.player_name)
    for log in game_logs:
        if normalize_player_name(log.player_name) != player_key:
            continue
        if log.game_date == target_date:
            return log
    return None


def outcome_for_projection(
    projection: Projection,
    game_log: PlayerGameLog,
) -> str | None:
    value = stat_value(projection.stat_type, game_log.stats)
    if value is None:
        return None
    if value > projection.line_score:
        return "OVER"
    if value < projection.line_score:
        return "UNDER"
    return "PUSH"
