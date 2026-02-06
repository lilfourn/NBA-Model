"""Add model_artifacts table for DB-based model storage."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0008_model_artifacts"
down_revision = "0007_schema_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact_data", sa.LargeBinary(), nullable=False),
        sa.Column("artifact_format", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
    )
    op.create_index(
        "idx_model_artifacts_name_created",
        "model_artifacts",
        ["model_name", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_model_artifacts_name_created", table_name="model_artifacts")
    op.drop_table("model_artifacts")
