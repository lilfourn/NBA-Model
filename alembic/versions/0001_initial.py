from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("league_id", sa.Text(), nullable=False),
        sa.Column("per_page", sa.Integer(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("snapshot_path", sa.Text(), nullable=True),
        sa.Column("data_count", sa.Integer(), nullable=True),
        sa.Column("included_count", sa.Integer(), nullable=True),
        sa.Column("links", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index("idx_snapshots_fetched_at", "snapshots", ["fetched_at"])
    op.create_index("idx_snapshots_league_id", "snapshots", ["league_id"])

    op.create_table(
        "players",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("team", sa.Text(), nullable=True),
        sa.Column("team_name", sa.Text(), nullable=True),
        sa.Column("position", sa.Text(), nullable=True),
        sa.Column("market", sa.Text(), nullable=True),
        sa.Column("jersey_number", sa.Text(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("league_id", sa.Text(), nullable=True),
        sa.Column("league", sa.Text(), nullable=True),
        sa.Column("combo", sa.Boolean(), nullable=True),
        sa.Column("team_id", sa.Text(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("relationships", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("abbreviation", sa.Text(), nullable=True),
        sa.Column("market", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("primary_color", sa.Text(), nullable=True),
        sa.Column("secondary_color", sa.Text(), nullable=True),
        sa.Column("tertiary_color", sa.Text(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "stat_types",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("lfg_ignored_leagues", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "games",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("is_live", sa.Boolean(), nullable=True),
        sa.Column("external_game_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("home_team_id", sa.Text(), nullable=True),
        sa.Column("away_team_id", sa.Text(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("relationships", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "projection_types",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "leagues",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=True),
        sa.Column("projections_count", sa.Integer(), nullable=True),
        sa.Column("icon", sa.Text(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("parent_id", sa.Text(), nullable=True),
        sa.Column("parent_name", sa.Text(), nullable=True),
        sa.Column("f2p_enabled", sa.Boolean(), nullable=True),
        sa.Column("has_live_projections", sa.Boolean(), nullable=True),
        sa.Column("last_five_games_enabled", sa.Boolean(), nullable=True),
        sa.Column("league_icon_id", sa.Text(), nullable=True),
        sa.Column("show_trending", sa.Boolean(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("relationships", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "durations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_table(
        "projections",
        sa.Column("snapshot_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("projection_id", sa.Text(), nullable=False),
        sa.Column("league_id", sa.Text(), nullable=True),
        sa.Column("player_id", sa.Text(), nullable=True),
        sa.Column("stat_type_id", sa.Text(), nullable=True),
        sa.Column("projection_type_id", sa.Text(), nullable=True),
        sa.Column("game_id", sa.Text(), nullable=True),
        sa.Column("duration_id", sa.Text(), nullable=True),
        sa.Column("line_score", sa.Numeric(), nullable=True),
        sa.Column("adjusted_odds", sa.Numeric(), nullable=True),
        sa.Column("discount_percentage", sa.Numeric(), nullable=True),
        sa.Column("flash_sale_line_score", sa.Numeric(), nullable=True),
        sa.Column("odds_type", sa.Integer(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("trending_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("stat_type", sa.Text(), nullable=True),
        sa.Column("stat_display_name", sa.Text(), nullable=True),
        sa.Column("projection_type", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=True),
        sa.Column("group_key", sa.Text(), nullable=True),
        sa.Column("tv_channel", sa.Text(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("board_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_promo", sa.Boolean(), nullable=True),
        sa.Column("is_live", sa.Boolean(), nullable=True),
        sa.Column("is_live_scored", sa.Boolean(), nullable=True),
        sa.Column("in_game", sa.Boolean(), nullable=True),
        sa.Column("today", sa.Boolean(), nullable=True),
        sa.Column("refundable", sa.Boolean(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("relationships", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["snapshot_id"], ["snapshots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("snapshot_id", "projection_id"),
    )
    op.create_index("idx_projections_projection_id", "projections", ["projection_id"])
    op.create_index("idx_projections_player_id", "projections", ["player_id"])
    op.create_index("idx_projections_stat_type_id", "projections", ["stat_type_id"])
    op.create_index("idx_projections_game_id", "projections", ["game_id"])
    op.create_index("idx_projections_league_id", "projections", ["league_id"])
    op.create_index("idx_projections_start_time", "projections", ["start_time"])


def downgrade() -> None:
    op.drop_index("idx_projections_start_time", table_name="projections")
    op.drop_index("idx_projections_league_id", table_name="projections")
    op.drop_index("idx_projections_game_id", table_name="projections")
    op.drop_index("idx_projections_stat_type_id", table_name="projections")
    op.drop_index("idx_projections_player_id", table_name="projections")
    op.drop_index("idx_projections_projection_id", table_name="projections")
    op.drop_table("projections")
    op.drop_table("durations")
    op.drop_table("leagues")
    op.drop_table("projection_types")
    op.drop_table("games")
    op.drop_table("stat_types")
    op.drop_table("teams")
    op.drop_table("players")
    op.drop_index("idx_snapshots_league_id", table_name="snapshots")
    op.drop_index("idx_snapshots_fetched_at", table_name="snapshots")
    op.drop_table("snapshots")
