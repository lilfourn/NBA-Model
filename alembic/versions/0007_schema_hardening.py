"""CHECK constraints, updated_at columns for data integrity."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_schema_hardening"
down_revision = "0006_prediction_outcomes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- updated_at columns on NBA tables --
    op.add_column("nba_players", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("nba_games", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("nba_player_game_stats", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))

    # -- CHECK constraints on nba_player_game_stats --
    op.create_check_constraint("ck_stats_points_gte_0", "nba_player_game_stats", "points IS NULL OR points >= 0")
    op.create_check_constraint("ck_stats_rebounds_gte_0", "nba_player_game_stats", "rebounds IS NULL OR rebounds >= 0")
    op.create_check_constraint("ck_stats_assists_gte_0", "nba_player_game_stats", "assists IS NULL OR assists >= 0")
    op.create_check_constraint("ck_stats_steals_gte_0", "nba_player_game_stats", "steals IS NULL OR steals >= 0")
    op.create_check_constraint("ck_stats_blocks_gte_0", "nba_player_game_stats", "blocks IS NULL OR blocks >= 0")
    op.create_check_constraint("ck_stats_turnovers_gte_0", "nba_player_game_stats", "turnovers IS NULL OR turnovers >= 0")
    op.create_check_constraint("ck_stats_fg3m_gte_0", "nba_player_game_stats", "fg3m IS NULL OR fg3m >= 0")
    op.create_check_constraint("ck_stats_fg3a_gte_0", "nba_player_game_stats", "fg3a IS NULL OR fg3a >= 0")
    op.create_check_constraint("ck_stats_fgm_gte_0", "nba_player_game_stats", "fgm IS NULL OR fgm >= 0")
    op.create_check_constraint("ck_stats_fga_gte_0", "nba_player_game_stats", "fga IS NULL OR fga >= 0")
    op.create_check_constraint("ck_stats_ftm_gte_0", "nba_player_game_stats", "ftm IS NULL OR ftm >= 0")
    op.create_check_constraint("ck_stats_fta_gte_0", "nba_player_game_stats", "fta IS NULL OR fta >= 0")
    op.create_check_constraint(
        "ck_stats_fg_pct_range", "nba_player_game_stats", "fg_pct IS NULL OR (fg_pct >= 0 AND fg_pct <= 1)"
    )
    op.create_check_constraint(
        "ck_stats_fg3_pct_range", "nba_player_game_stats", "fg3_pct IS NULL OR (fg3_pct >= 0 AND fg3_pct <= 1)"
    )
    op.create_check_constraint(
        "ck_stats_ft_pct_range", "nba_player_game_stats", "ft_pct IS NULL OR (ft_pct >= 0 AND ft_pct <= 1)"
    )
    op.create_check_constraint(
        "ck_stats_minutes_gte_0", "nba_player_game_stats", "minutes IS NULL OR minutes >= 0"
    )

    # -- CHECK constraints on projections --
    op.create_check_constraint(
        "ck_projections_line_score_gt_0", "projections", "line_score IS NULL OR line_score > 0"
    )

    # -- CHECK constraints on projection_predictions --
    op.create_check_constraint(
        "ck_predictions_prob_over_range",
        "projection_predictions",
        "prob_over IS NULL OR (prob_over >= 0 AND prob_over <= 1)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_predictions_prob_over_range", "projection_predictions", type_="check")
    op.drop_constraint("ck_projections_line_score_gt_0", "projections", type_="check")

    op.drop_constraint("ck_stats_minutes_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_ft_pct_range", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fg3_pct_range", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fg_pct_range", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fta_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_ftm_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fga_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fgm_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fg3a_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_fg3m_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_turnovers_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_blocks_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_steals_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_assists_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_rebounds_gte_0", "nba_player_game_stats", type_="check")
    op.drop_constraint("ck_stats_points_gte_0", "nba_player_game_stats", type_="check")

    op.drop_column("nba_player_game_stats", "updated_at")
    op.drop_column("nba_games", "updated_at")
    op.drop_column("nba_players", "updated_at")
