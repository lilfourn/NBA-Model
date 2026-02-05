from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0006_prediction_outcomes"
down_revision = "0005_performance_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projection_predictions", sa.Column("decision_time", sa.DateTime(timezone=True), nullable=True))
    op.add_column("projection_predictions", sa.Column("player_id", sa.Text(), nullable=True))
    op.add_column("projection_predictions", sa.Column("game_id", sa.Text(), nullable=True))
    op.add_column("projection_predictions", sa.Column("stat_type", sa.Text(), nullable=True))
    op.add_column("projection_predictions", sa.Column("line_score", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("p_forecast_cal", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("p_nn", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("p_lr", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("p_xgb", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("p_lgbm", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("rank_score", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("n_eff", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("actual_value", sa.Numeric(), nullable=True))
    op.add_column("projection_predictions", sa.Column("over_label", sa.Integer(), nullable=True))
    op.add_column("projection_predictions", sa.Column("outcome", sa.Text(), nullable=True))
    op.add_column("projection_predictions", sa.Column("is_correct", sa.Boolean(), nullable=True))
    op.add_column("projection_predictions", sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index("idx_prediction_decision_time", "projection_predictions", ["decision_time"])
    op.create_index("idx_prediction_resolved_at", "projection_predictions", ["resolved_at"])
    op.create_index("idx_prediction_player_game", "projection_predictions", ["player_id", "game_id"])
    op.execute(
        """
        create index if not exists idx_prediction_unresolved
        on projection_predictions (decision_time)
        where actual_value is null
        """
    )


def downgrade() -> None:
    op.execute("drop index if exists idx_prediction_unresolved")
    op.drop_index("idx_prediction_player_game", table_name="projection_predictions")
    op.drop_index("idx_prediction_resolved_at", table_name="projection_predictions")
    op.drop_index("idx_prediction_decision_time", table_name="projection_predictions")

    op.drop_column("projection_predictions", "resolved_at")
    op.drop_column("projection_predictions", "is_correct")
    op.drop_column("projection_predictions", "outcome")
    op.drop_column("projection_predictions", "over_label")
    op.drop_column("projection_predictions", "actual_value")
    op.drop_column("projection_predictions", "n_eff")
    op.drop_column("projection_predictions", "rank_score")
    op.drop_column("projection_predictions", "p_lgbm")
    op.drop_column("projection_predictions", "p_xgb")
    op.drop_column("projection_predictions", "p_lr")
    op.drop_column("projection_predictions", "p_nn")
    op.drop_column("projection_predictions", "p_forecast_cal")
    op.drop_column("projection_predictions", "line_score")
    op.drop_column("projection_predictions", "stat_type")
    op.drop_column("projection_predictions", "game_id")
    op.drop_column("projection_predictions", "player_id")
    op.drop_column("projection_predictions", "decision_time")
