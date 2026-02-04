from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.db.engine import get_engine

router = APIRouter()


@router.get("/metrics/line-movement", tags=["metrics"])
def line_movement_metrics() -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        latest = conn.execute(
            text("select id, fetched_at from snapshots order by fetched_at desc limit 1")
        ).first()
        if not latest:
            raise HTTPException(status_code=404, detail="No snapshots loaded")

        snapshot_id = latest.id
        rows = conn.execute(
            text(
                """
                select line_movement, count(*)
                from projections
                where snapshot_id = :sid
                  and lower(coalesce(attributes->>'odds_type', 'standard')) = 'standard'
                group by line_movement
                """
            ),
            {"sid": snapshot_id},
        ).all()

        counts = {str(row[0] or "null"): row[1] for row in rows}
        total = sum(counts.values())

        return {
            "snapshot_id": str(snapshot_id),
            "fetched_at": latest.fetched_at.isoformat() if latest.fetched_at else None,
            "total": total,
            "counts": counts,
        }
