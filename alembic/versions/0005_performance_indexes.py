from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_performance_indexes"
down_revision = "0004_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_nba_player_stats_player_id",
        "nba_player_game_stats",
        ["player_id"],
    )
    op.create_index(
        "idx_nba_player_stats_player_game",
        "nba_player_game_stats",
        ["player_id", "game_id"],
    )
    op.create_index(
        "idx_nba_games_date_home_away",
        "nba_games",
        ["game_date", "home_team_abbreviation", "away_team_abbreviation"],
    )
    op.create_index(
        "idx_projections_snapshot_odds_type",
        "projections",
        ["snapshot_id", "odds_type"],
    )
    op.execute(
        sa.text(
            """
            create index if not exists idx_projection_features_start_date_snapshot
            on projection_features (((start_time at time zone 'America/New_York')::date), snapshot_id)
            """
        )
    )


def downgrade() -> None:
    op.execute("drop index if exists idx_projection_features_start_date_snapshot")
    op.drop_index("idx_projections_snapshot_odds_type", table_name="projections")
    op.drop_index("idx_nba_games_date_home_away", table_name="nba_games")
    op.drop_index("idx_nba_player_stats_player_game", table_name="nba_player_game_stats")
    op.drop_index("idx_nba_player_stats_player_id", table_name="nba_player_game_stats")
