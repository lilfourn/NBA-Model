from __future__ import annotations

from fastapi import APIRouter, Query

from app.db.engine import get_engine
from app.services.scoring import list_snapshots, score_ensemble

router = APIRouter()


@router.get("/picks", tags=["picks"])
def get_picks(
    snapshot_id: str | None = Query(None, description="Specific snapshot UUID"),
    game_date: str | None = Query(None, description="Slate date YYYY-MM-DD (America/New_York)"),
    top: int = Query(50, ge=1, le=200),
    rank: str = Query("risk_adj", pattern="^(risk_adj|confidence)$"),
    include_non_today: bool = Query(False),
) -> dict:
    engine = get_engine()
    result = score_ensemble(
        engine,
        snapshot_id=snapshot_id,
        game_date=game_date,
        top=top,
        rank=rank,
        include_non_today=include_non_today,
    )
    return result.to_dict()


@router.get("/snapshots-list", tags=["picks"])
def get_snapshots(limit: int = Query(20, ge=1, le=100)) -> dict:
    engine = get_engine()
    snapshots = list_snapshots(engine, limit=limit)
    return {"count": len(snapshots), "snapshots": snapshots}
