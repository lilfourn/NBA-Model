"""Add market_game_lines table for external sportsbook context features."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0012_market_game_lines"
down_revision = "0011_canonical_view"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_game_lines",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("book", sa.Text(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("home_team_abbreviation", sa.Text(), nullable=False),
        sa.Column("away_team_abbreviation", sa.Text(), nullable=False),
        sa.Column("home_spread", sa.Numeric(), nullable=True),
        sa.Column("away_spread", sa.Numeric(), nullable=True),
        sa.Column("total_points", sa.Numeric(), nullable=True),
        sa.Column("home_moneyline", sa.Integer(), nullable=True),
        sa.Column("away_moneyline", sa.Integer(), nullable=True),
        sa.Column("source_payload", JSONB(), nullable=True),
    )

    op.create_index(
        "idx_market_lines_game_date_teams",
        "market_game_lines",
        ["game_date", "home_team_abbreviation", "away_team_abbreviation"],
    )
    op.create_index(
        "idx_market_lines_captured_at",
        "market_game_lines",
        ["captured_at"],
    )
    op.create_index(
        "idx_market_lines_provider_book",
        "market_game_lines",
        ["provider", "book"],
    )


def downgrade() -> None:
    op.drop_index("idx_market_lines_provider_book", table_name="market_game_lines")
    op.drop_index("idx_market_lines_captured_at", table_name="market_game_lines")
    op.drop_index("idx_market_lines_game_date_teams", table_name="market_game_lines")
    op.drop_table("market_game_lines")
