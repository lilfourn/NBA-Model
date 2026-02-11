from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query

from app.core.config import settings
from app.db.engine import get_engine
from app.services.jobs import JobType, job_manager
from app.services.scoring import list_snapshots, score_ensemble, score_logged_predictions

router = APIRouter()
logger = logging.getLogger(__name__)


def _use_logged_picks_source() -> bool:
    source = (settings.picks_source or "inline").strip().lower()
    return source in {"modal_db", "logged_db", "logged", "db"}


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
    if _use_logged_picks_source():
        if force:
            try:
                if not job_manager.is_running(JobType.COLLECT):
                    job_manager.start_job(JobType.COLLECT)
            except ValueError:
                # Collect already running; fetch current logged picks.
                pass
            except Exception:
                logger.warning(
                    "Failed to trigger Modal collect job from /api/picks force=true",
                    exc_info=True,
                )
        result = await asyncio.to_thread(
            score_logged_predictions,
            engine,
            snapshot_id=snapshot_id,
            game_date=game_date,
            top=top,
            rank=rank,
            include_non_today=include_non_today,
        )
    else:
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
