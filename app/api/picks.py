from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from app.core.config import settings
from app.db.engine import get_engine
from app.services.scoring import list_snapshots, score_ensemble

router = APIRouter()


@router.get("/picks", tags=["picks"])
async def get_picks(
    snapshot_id: str | None = Query(None, description="Specific snapshot UUID"),
    game_date: str | None = Query(None, description="Slate date YYYY-MM-DD (America/New_York)"),
    top: int = Query(50, ge=1, le=200),
    rank: str = Query("risk_adj", pattern="^(risk_adj|confidence)$"),
    include_non_today: bool = Query(False),
    force: bool = Query(False, description="Bypass scoring cache and re-run ensemble"),
) -> dict:
    engine = get_engine()
    result = await asyncio.to_thread(
        score_ensemble,
        engine,
        snapshot_id=snapshot_id,
        game_date=game_date,
        models_dir=settings.models_dir,
        ensemble_weights_path=settings.ensemble_weights_path,
        top=top,
        rank=rank,
        include_non_today=include_non_today,
        force=force,
    )
    return result.to_dict()


@router.get("/snapshots-list", tags=["picks"])
async def get_snapshots(limit: int = Query(20, ge=1, le=100)) -> dict:
    engine = get_engine()
    snapshots = await asyncio.to_thread(list_snapshots, engine, limit=limit)
    return {"count": len(snapshots), "snapshots": snapshots}
