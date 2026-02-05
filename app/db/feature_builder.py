from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Engine


def build_projection_features(engine: Engine, snapshot_id: UUID) -> dict[str, Any]:
    sql = text(
        """
        insert into projection_features (
            snapshot_id,
            projection_id,
            league_id,
            player_id,
            team_id,
            stat_type_id,
            projection_type_id,
            game_id,
            duration_id,
            line_score,
            line_score_prev,
            line_score_delta,
            line_movement,
            stat_type,
            projection_type,
            odds_type,
            trending_count,
            is_promo,
            is_live,
            in_game,
            today,
            start_time,
            board_time,
            end_time,
            fetched_at,
            minutes_to_start
        )
        select
            p.snapshot_id,
            p.projection_id,
            p.league_id,
            p.player_id,
            pl.team_id,
            p.stat_type_id,
            p.projection_type_id,
            p.game_id,
            p.duration_id,
            p.line_score,
            p.line_score_prev,
            p.line_score_delta,
            p.line_movement,
            p.stat_type,
            p.projection_type,
            coalesce(p.odds_type, 0) as odds_type,
            p.trending_count,
            p.is_promo,
            p.is_live,
            p.in_game,
            p.today,
            p.start_time,
            p.board_time,
            p.end_time,
            s.fetched_at,
            case
                when p.start_time is not null and s.fetched_at is not null then
                    floor(extract(epoch from (p.start_time - s.fetched_at)) / 60)
                else null
            end as minutes_to_start
        from projections p
        join snapshots s on s.id = p.snapshot_id
        left join players pl on pl.id = p.player_id
        where p.snapshot_id = :snapshot_id
          and coalesce(p.odds_type, 0) = 0
        on conflict (snapshot_id, projection_id) do update
        set
            league_id = excluded.league_id,
            player_id = excluded.player_id,
            team_id = excluded.team_id,
            stat_type_id = excluded.stat_type_id,
            projection_type_id = excluded.projection_type_id,
            game_id = excluded.game_id,
            duration_id = excluded.duration_id,
            line_score = excluded.line_score,
            line_score_prev = excluded.line_score_prev,
            line_score_delta = excluded.line_score_delta,
            line_movement = excluded.line_movement,
            stat_type = excluded.stat_type,
            projection_type = excluded.projection_type,
            odds_type = excluded.odds_type,
            trending_count = excluded.trending_count,
            is_promo = excluded.is_promo,
            is_live = excluded.is_live,
            in_game = excluded.in_game,
            today = excluded.today,
            start_time = excluded.start_time,
            board_time = excluded.board_time,
            end_time = excluded.end_time,
            fetched_at = excluded.fetched_at,
            minutes_to_start = excluded.minutes_to_start
        """
    )

    with engine.begin() as conn:
        result = conn.execute(sql, {"snapshot_id": snapshot_id})

    rowcount = result.rowcount
    if rowcount is None or rowcount < 0:
        rowcount = 0

    return {"snapshot_id": str(snapshot_id), "rows": rowcount}
