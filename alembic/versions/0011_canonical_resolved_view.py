"""Create vw_resolved_picks_canonical view for unified metric evaluation."""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0011_canonical_view"
down_revision = "0010_add_p_raw"
branch_labels = None
depends_on = None

VIEW_SQL = """
CREATE OR REPLACE VIEW vw_resolved_picks_canonical AS
SELECT
    pp.id,
    pp.projection_id,
    pp.snapshot_id,
    pp.player_id,
    pp.game_id,
    pp.stat_type,
    pp.line_score AS line_at_decision,
    pp.decision_time,
    pp.prob_over AS p_final,
    pp.p_raw,
    pp.p_forecast_cal,
    pp.p_nn,
    coalesce(pp.p_tabdl::text, pp.details->>'p_tabdl') AS p_tabdl,
    pp.p_lr,
    pp.p_xgb,
    pp.p_lgbm,
    pp.over_label,
    pp.outcome,
    pp.is_correct,
    pp.actual_value,
    pp.resolved_at,
    pp.created_at,
    pp.n_eff,
    pp.rank_score,
    pp.details
FROM projection_predictions pp
WHERE pp.outcome IN ('over', 'under')
  AND pp.over_label IS NOT NULL
  AND pp.actual_value IS NOT NULL;
"""


def upgrade() -> None:
    op.execute(text(VIEW_SQL))


def downgrade() -> None:
    op.execute(text("DROP VIEW IF EXISTS vw_resolved_picks_canonical;"))
