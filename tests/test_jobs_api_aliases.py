from __future__ import annotations

import pytest

from app.api.jobs import _parse_job_type
from app.services.jobs import JobType


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("collect", JobType.COLLECT),
        ("train", JobType.TRAIN),
        ("train_baseline", JobType.TRAIN),
        ("train_nn", JobType.TRAIN),
        ("train_ensemble", JobType.TRAIN),
        ("build_backtest", JobType.TRAIN),
        ("calibrate", JobType.TRAIN),
        ("collect_now", JobType.COLLECT),
        ("train_now", JobType.TRAIN),
    ],
)
def test_parse_job_type_aliases(raw: str, expected: JobType) -> None:
    assert _parse_job_type(raw) == expected


def test_parse_job_type_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _parse_job_type("not-a-real-job")
