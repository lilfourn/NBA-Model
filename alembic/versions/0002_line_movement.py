from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_line_movement"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projections", sa.Column("line_score_prev", sa.Numeric(), nullable=True))
    op.add_column("projections", sa.Column("line_score_delta", sa.Numeric(), nullable=True))
    op.add_column("projections", sa.Column("line_movement", sa.Text(), nullable=True))

    op.create_index(
        "uq_snapshots_snapshot_path",
        "snapshots",
        ["snapshot_path"],
        unique=True,
        postgresql_where=sa.text("snapshot_path IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_snapshots_snapshot_path", table_name="snapshots")
    op.drop_column("projections", "line_movement")
    op.drop_column("projections", "line_score_delta")
    op.drop_column("projections", "line_score_prev")
