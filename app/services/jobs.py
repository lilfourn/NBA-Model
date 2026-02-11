from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


from app.services.scoring import invalidate_scoring_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODAL_BIN = os.getenv("MODAL_BIN", "").strip()
MODAL_APP_REF = os.getenv("MODAL_APP_REF", str(PROJECT_ROOT / "modal_app.py"))
JOB_OUTPUT_MAX_CHARS = 5000


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    COLLECT = "collect"
    TRAIN = "train"


_MODAL_ENTRYPOINTS: dict[JobType, str] = {
    JobType.COLLECT: "collect_now",
    JobType.TRAIN: "train_now",
}

JOB_TIMEOUT_SECONDS: dict[JobType, int] = {
    JobType.COLLECT: int(os.getenv("JOB_TIMEOUT_COLLECT_SECONDS", "3600")),
    JobType.TRAIN: int(os.getenv("JOB_TIMEOUT_TRAIN_SECONDS", "14400")),
}

JOB_LABELS: dict[JobType, str] = {
    JobType.COLLECT: "Collect Pipeline (Modal)",
    JobType.TRAIN: "Train Pipeline (Modal)",
}


def _modal_run_command(job_type: JobType) -> list[str]:
    entrypoint = _MODAL_ENTRYPOINTS[job_type]
    target = f"{MODAL_APP_REF}::{entrypoint}"
    if MODAL_BIN:
        return [MODAL_BIN, "run", target]
    return [sys.executable, "-m", "modal", "run", target]


def _ensure_modal_cli_available() -> None:
    if MODAL_BIN:
        if shutil.which(MODAL_BIN):
            return
        raise RuntimeError(
            f"Modal CLI '{MODAL_BIN}' not found in PATH. "
            "Install modal and provide MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in Railway."
        )
    try:
        import modal  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Python package 'modal' is not installed in this runtime. "
            "Install it in the API image and provide MODAL_TOKEN_ID/MODAL_TOKEN_SECRET in Railway."
        ) from exc


@dataclass
class Job:
    id: str
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    output: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "label": JOB_LABELS.get(self.job_type, self.job_type.value),
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "output": self.output[-JOB_OUTPUT_MAX_CHARS:] if self.output else "",
            "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.started_at or "",
                reverse=True,
            )
            return [j.to_dict() for j in jobs[:limit]]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def is_running(self, job_type: JobType) -> bool:
        with self._lock:
            return any(
                j.job_type == job_type and j.status == JobStatus.RUNNING
                for j in self._jobs.values()
            )

    def start_job(self, job_type: JobType) -> dict[str, Any]:
        _ensure_modal_cli_available()
        if self.is_running(job_type):
            raise ValueError(f"{JOB_LABELS[job_type]} is already running")

        job_id = str(uuid4())
        job = Job(id=job_id, job_type=job_type)

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True,
        )
        thread.start()
        return job.to_dict()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()

        cmd = _modal_run_command(job.job_type)
        timeout_seconds = JOB_TIMEOUT_SECONDS[job.job_type]
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(PROJECT_ROOT),
            )
            elapsed = time.monotonic() - start

            with self._lock:
                job.output = (
                    f"$ {' '.join(cmd)}\n"
                    f"{result.stdout or ''}{result.stderr or ''}"
                )
                job.duration_seconds = round(elapsed, 2)
                job.finished_at = datetime.now(timezone.utc).isoformat()

                if result.returncode == 0:
                    job.status = JobStatus.COMPLETED
                    invalidate_scoring_cache()
                else:
                    job.status = JobStatus.FAILED
                    job.error = f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired as exc:
            with self._lock:
                job.status = JobStatus.FAILED
                out = (exc.stdout or "") + (exc.stderr or "")
                if out:
                    job.output = f"$ {' '.join(cmd)}\n{out}"
                job.error = f"Timed out after {timeout_seconds} seconds"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.duration_seconds = round(time.monotonic() - start, 2)

        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = f"{exc.__class__.__name__}: {exc}"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.duration_seconds = round(time.monotonic() - start, 2)


# Singleton
job_manager = JobManager()
