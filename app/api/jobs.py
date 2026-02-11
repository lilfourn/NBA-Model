from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.jobs import JobType, job_manager

router = APIRouter()

LEGACY_JOB_TYPE_ALIASES: dict[str, str] = {
    "train_baseline": JobType.TRAIN.value,
    "train_nn": JobType.TRAIN.value,
    "train_ensemble": JobType.TRAIN.value,
    "build_backtest": JobType.TRAIN.value,
    "calibrate": JobType.TRAIN.value,
    "collect_now": JobType.COLLECT.value,
    "train_now": JobType.TRAIN.value,
}


class StartJobRequest(BaseModel):
    job_type: str


def _parse_job_type(raw_job_type: str) -> JobType:
    normalized = (raw_job_type or "").strip().lower()
    mapped = LEGACY_JOB_TYPE_ALIASES.get(normalized, normalized)
    return JobType(mapped)


@router.post("/jobs", tags=["jobs"])
def start_job(req: StartJobRequest) -> dict:
    try:
        jt = _parse_job_type(req.job_type)
    except ValueError:
        valid = sorted({*LEGACY_JOB_TYPE_ALIASES.keys(), *(t.value for t in JobType)})
        raise HTTPException(status_code=400, detail=f"Invalid job_type. Valid: {valid}")

    try:
        return job_manager.start_job(jt)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/jobs", tags=["jobs"])
def list_jobs(limit: int = Query(20, ge=1, le=100)) -> dict:
    jobs = job_manager.list_jobs(limit=limit)
    return {"count": len(jobs), "jobs": jobs}


@router.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/job-types", tags=["jobs"])
def list_job_types() -> dict:
    from app.services.jobs import JOB_LABELS
    return {
        "job_types": [
            {"value": jt.value, "label": JOB_LABELS[jt]}
            for jt in JobType
        ]
    }
