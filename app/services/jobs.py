from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    TRAIN_BASELINE = "train_baseline"
    TRAIN_NN = "train_nn"
    TRAIN_ENSEMBLE = "train_ensemble"
    BUILD_BACKTEST = "build_backtest"
    CALIBRATE = "calibrate"


JOB_COMMANDS: dict[JobType, list[str]] = {
    JobType.TRAIN_BASELINE: [
        sys.executable, "-m", "scripts.ml.train_baseline_model",
    ],
    JobType.TRAIN_NN: [
        sys.executable, "-m", "scripts.ml.train_nn_model",
    ],
    JobType.TRAIN_ENSEMBLE: [
        sys.executable, "-m", "scripts.ml.train_online_ensemble",
        "--source", "db",
        "--days-back", "90",
        "--log-path", "data/monitoring/prediction_log.csv",
        "--out", "models/ensemble_weights.json",
    ],
    JobType.BUILD_BACKTEST: [
        sys.executable, "-m", "scripts.calibration.build_forecast_backtest_dataset",
        "--output", "data/calibration/forecast_backtest.csv",
    ],
    JobType.CALIBRATE: [
        sys.executable, "-m", "scripts.calibration.calibrate_forecast_distribution",
        "--input", "data/calibration/forecast_backtest.csv",
        "--output", "data/calibration/forecast_calibration.json",
    ],
}

JOB_LABELS: dict[JobType, str] = {
    JobType.TRAIN_BASELINE: "Train Baseline (LR)",
    JobType.TRAIN_NN: "Train Neural Network",
    JobType.TRAIN_ENSEMBLE: "Train Ensemble Weights",
    JobType.BUILD_BACKTEST: "Build Backtest Dataset",
    JobType.CALIBRATE: "Calibrate Forecast",
}


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
            "output": self.output[-5000:] if self.output else "",
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

        cmd = JOB_COMMANDS[job.job_type]
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max
            )
            elapsed = time.monotonic() - start

            with self._lock:
                job.output = (result.stdout or "") + (result.stderr or "")
                job.duration_seconds = round(elapsed, 2)
                job.finished_at = datetime.now(timezone.utc).isoformat()

                if result.returncode == 0:
                    job.status = JobStatus.COMPLETED
                else:
                    job.status = JobStatus.FAILED
                    job.error = f"Exit code {result.returncode}"

        except subprocess.TimeoutExpired:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = "Timed out after 30 minutes"
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
