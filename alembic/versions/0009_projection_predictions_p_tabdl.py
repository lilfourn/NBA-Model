"""Add typed p_tabdl column to projection_predictions."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0009_p_tabdl_col"
down_revision = "0008_model_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projection_predictions", sa.Column("p_tabdl", sa.Numeric(), nullable=True))
    op.create_index(
        "idx_prediction_p_tabdl_not_null",
        "projection_predictions",
        ["p_tabdl"],
        postgresql_where=sa.text("p_tabdl is not null"),
    )


def downgrade() -> None:
    op.drop_index("idx_prediction_p_tabdl_not_null", table_name="projection_predictions")
    op.drop_column("projection_predictions", "p_tabdl")
