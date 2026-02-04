from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_player_game_logs"
down_revision = "0002_line_movement"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "player_game_logs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("player_id", sa.Text(), nullable=True),
        sa.Column("player_name", sa.Text(), nullable=True),
        sa.Column("game_id", sa.Text(), nullable=True),
        sa.Column("game_date", sa.Date(), nullable=True),
        sa.Column("team_id", sa.Text(), nullable=True),
        sa.Column("team_abbreviation", sa.Text(), nullable=True),
        sa.Column("season", sa.Text(), nullable=True),
        sa.Column("season_type", sa.Text(), nullable=True),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index("idx_player_game_logs_player_id", "player_game_logs", ["player_id"])
    op.create_index("idx_player_game_logs_game_date", "player_game_logs", ["game_date"])

    op.create_table(
        "projection_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("snapshots.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("projection_id", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("pick", sa.Text(), nullable=False),
        sa.Column("prob_over", sa.Numeric(), nullable=True),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column("mean", sa.Numeric(), nullable=True),
        sa.Column("std", sa.Numeric(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "idx_prediction_snapshot_id",
        "projection_predictions",
        ["snapshot_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_prediction_snapshot_id", table_name="projection_predictions")
    op.drop_table("projection_predictions")

    op.drop_index("idx_player_game_logs_game_date", table_name="player_game_logs")
    op.drop_index("idx_player_game_logs_player_id", table_name="player_game_logs")
    op.drop_table("player_game_logs")
