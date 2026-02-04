from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from app.db.engine import get_engine

router = APIRouter()


@router.get("/metrics/snapshots", tags=["metrics"])
def snapshots_metrics(limit: int = Query(10, ge=1, le=100)) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select id, fetched_at, data_count, included_count, snapshot_path
                from snapshots
                order by fetched_at desc
                limit :limit
                """
            ),
            {"limit": limit},
        ).all()

        if not rows:
            raise HTTPException(status_code=404, detail="No snapshots loaded")

        snapshots = [
            {
                "id": str(row.id),
                "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None,
                "data_count": row.data_count,
                "included_count": row.included_count,
                "snapshot_path": row.snapshot_path,
            }
            for row in rows
        ]

        return {"count": len(snapshots), "snapshots": snapshots}
