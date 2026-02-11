from __future__ import annotations

import sys

import pytest

from app.services import jobs


def test_modal_run_command_collect_defaults_to_python_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "MODAL_BIN", "")
    monkeypatch.setattr(jobs, "MODAL_APP_REF", "/app/modal_app.py")
    assert jobs._modal_run_command(jobs.JobType.COLLECT) == [
        sys.executable,
        "-m",
        "modal",
        "run",
        "/app/modal_app.py::collect_now",
    ]


def test_modal_run_command_train_with_custom_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "MODAL_BIN", "modal")
    monkeypatch.setattr(jobs, "MODAL_APP_REF", "/app/modal_app.py")
    assert jobs._modal_run_command(jobs.JobType.TRAIN) == [
        "modal",
        "run",
        "/app/modal_app.py::train_now",
    ]


def test_missing_modal_cli_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "MODAL_BIN", "modal-does-not-exist")
    monkeypatch.setattr(jobs.shutil, "which", lambda _binary: None)
    with pytest.raises(RuntimeError, match="Modal CLI"):
        jobs._ensure_modal_cli_available()


def test_start_job_does_not_enqueue_when_modal_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "MODAL_BIN", "modal-does-not-exist")
    monkeypatch.setattr(jobs.shutil, "which", lambda _binary: None)
    manager = jobs.JobManager()
    with pytest.raises(RuntimeError, match="Modal CLI"):
        manager.start_job(jobs.JobType.COLLECT)
    assert manager.list_jobs(limit=10) == []
