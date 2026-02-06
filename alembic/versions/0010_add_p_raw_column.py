"""Add p_raw (pre-shrinkage probability) column to projection_predictions."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0010_add_p_raw"
down_revision = "0009_p_tabdl_col"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projection_predictions", sa.Column("p_raw", sa.Numeric(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("projection_predictions", "p_raw")
