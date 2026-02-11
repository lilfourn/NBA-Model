from __future__ import annotations


import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.modeling.types import PlayerGameLog


def load_db_game_logs(engine: Engine) -> list[PlayerGameLog]:
    query = text(
        """
        select
            np.full_name as player_name,
            ng.game_date as game_date,
            s.stats_json as stats_json
        from nba_player_game_stats s
        join nba_games ng on ng.id = s.game_id
        join nba_players np on np.id = s.player_id
        """
    )
    df = pd.read_sql(query, engine)
    logs: list[PlayerGameLog] = []
    for row in df.itertuples(index=False):
        stats = row.stats_json or {}
        logs.append(
            PlayerGameLog(
                player_id=None,
                player_name=row.player_name,
                game_date=row.game_date,
                stats=stats,
            )
        )
    return logs
