from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_features_nba"
down_revision = "0002_line_movement"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("players", sa.Column("name_key", sa.Text(), nullable=True))
    op.create_index("idx_players_name_key", "players", ["name_key"])

    op.create_table(
        "snapshot_audits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    )
    op.create_index(
        "idx_snapshot_audits_snapshot_id",
        "snapshot_audits",
        ["snapshot_id"],
        unique=True,
    )

    op.create_table(
        "projection_features",
        sa.Column(
            "snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("snapshots.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("projection_id", sa.Text(), primary_key=True),
        sa.Column("league_id", sa.Text(), nullable=True),
        sa.Column("player_id", sa.Text(), nullable=True),
        sa.Column("team_id", sa.Text(), nullable=True),
        sa.Column("stat_type_id", sa.Text(), nullable=True),
        sa.Column("projection_type_id", sa.Text(), nullable=True),
        sa.Column("game_id", sa.Text(), nullable=True),
        sa.Column("duration_id", sa.Text(), nullable=True),
        sa.Column("line_score", sa.Numeric(), nullable=True),
        sa.Column("line_score_prev", sa.Numeric(), nullable=True),
        sa.Column("line_score_delta", sa.Numeric(), nullable=True),
        sa.Column("line_movement", sa.Text(), nullable=True),
        sa.Column("stat_type", sa.Text(), nullable=True),
        sa.Column("projection_type", sa.Text(), nullable=True),
        sa.Column("odds_type", sa.Integer(), nullable=True),
        sa.Column("trending_count", sa.Integer(), nullable=True),
        sa.Column("is_promo", sa.Boolean(), nullable=True),
        sa.Column("is_live", sa.Boolean(), nullable=True),
        sa.Column("in_game", sa.Boolean(), nullable=True),
        sa.Column("today", sa.Boolean(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("board_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("minutes_to_start", sa.Integer(), nullable=True),
    )
    op.create_index("idx_projection_features_snapshot_id", "projection_features", ["snapshot_id"])
    op.create_index("idx_projection_features_player_id", "projection_features", ["player_id"])
    op.create_index("idx_projection_features_game_id", "projection_features", ["game_id"])

    op.create_table(
        "nba_players",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("full_name", sa.Text(), nullable=True),
        sa.Column("name_key", sa.Text(), nullable=True),
        sa.Column("team_id", sa.Text(), nullable=True),
        sa.Column("team_abbreviation", sa.Text(), nullable=True),
    )
    op.create_index("idx_nba_players_name_key", "nba_players", ["name_key"])

    op.create_table(
        "nba_games",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("game_date", sa.Date(), nullable=True),
        sa.Column("status_text", sa.Text(), nullable=True),
        sa.Column("home_team_id", sa.Text(), nullable=True),
        sa.Column("away_team_id", sa.Text(), nullable=True),
        sa.Column("home_team_abbreviation", sa.Text(), nullable=True),
        sa.Column("away_team_abbreviation", sa.Text(), nullable=True),
    )

    op.create_table(
        "nba_player_game_stats",
        sa.Column("game_id", sa.Text(), primary_key=True),
        sa.Column("player_id", sa.Text(), primary_key=True),
        sa.Column("team_id", sa.Text(), nullable=True),
        sa.Column("team_abbreviation", sa.Text(), nullable=True),
        sa.Column("minutes", sa.Numeric(), nullable=True),
        sa.Column("points", sa.Integer(), nullable=True),
        sa.Column("rebounds", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("steals", sa.Integer(), nullable=True),
        sa.Column("blocks", sa.Integer(), nullable=True),
        sa.Column("turnovers", sa.Integer(), nullable=True),
        sa.Column("fg3m", sa.Integer(), nullable=True),
        sa.Column("fg3a", sa.Integer(), nullable=True),
        sa.Column("fg3_pct", sa.Numeric(), nullable=True),
        sa.Column("fgm", sa.Integer(), nullable=True),
        sa.Column("fga", sa.Integer(), nullable=True),
        sa.Column("fg_pct", sa.Numeric(), nullable=True),
        sa.Column("ftm", sa.Integer(), nullable=True),
        sa.Column("fta", sa.Integer(), nullable=True),
        sa.Column("ft_pct", sa.Numeric(), nullable=True),
        sa.Column("plus_minus", sa.Numeric(), nullable=True),
        sa.Column("stats_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index("idx_nba_player_stats_game_id", "nba_player_game_stats", ["game_id"])

    op.create_table(
        "model_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("train_rows", sa.Integer(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("model_runs")
    op.drop_index("idx_nba_player_stats_game_id", table_name="nba_player_game_stats")
    op.drop_table("nba_player_game_stats")
    op.drop_table("nba_games")
    op.drop_index("idx_nba_players_name_key", table_name="nba_players")
    op.drop_table("nba_players")
    op.drop_index("idx_projection_features_game_id", table_name="projection_features")
    op.drop_index("idx_projection_features_player_id", table_name="projection_features")
    op.drop_index("idx_projection_features_snapshot_id", table_name="projection_features")
    op.drop_table("projection_features")
    op.drop_index("idx_snapshot_audits_snapshot_id", table_name="snapshot_audits")
    op.drop_table("snapshot_audits")
    op.drop_index("idx_players_name_key", table_name="players")
    op.drop_column("players", "name_key")
