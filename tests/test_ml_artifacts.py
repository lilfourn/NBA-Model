from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from app.ml import artifacts


def test_load_joblib_artifact_rejects_sklearn_version_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"model": object()}

    def _fake_load(_path: str):
        warnings.warn(
            "Trying to unpickle estimator OneHotEncoder from version 1.8.0 when using version 1.7.2.",
            UserWarning,
            stacklevel=1,
        )
        return payload

    monkeypatch.setattr(artifacts.joblib, "load", _fake_load)

    with pytest.raises(RuntimeError, match="Incompatible sklearn artifact"):
        artifacts.load_joblib_artifact("models/fake.joblib")


def test_load_joblib_artifact_allows_warning_when_not_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"model": object()}

    def _fake_load(_path: str):
        warnings.warn(
            "Trying to unpickle estimator OneHotEncoder from version 1.8.0 when using version 1.7.2.",
            UserWarning,
            stacklevel=1,
        )
        return payload

    monkeypatch.setattr(artifacts.joblib, "load", _fake_load)

    loaded = artifacts.load_joblib_artifact(
        "models/fake.joblib",
        strict_sklearn_version=False,
    )
    assert loaded is payload


def test_latest_compatible_joblib_path_skips_incompatible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    older = tmp_path / "baseline_logreg_20240101_000000Z.joblib"
    newer = tmp_path / "baseline_logreg_20240102_000000Z.joblib"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")

    def _fake_load(path: str | Path, *, strict_sklearn_version: bool = True):
        if str(path).endswith(newer.name):
            raise RuntimeError("incompatible")
        return {"model": object()}

    monkeypatch.setattr(artifacts, "load_joblib_artifact", _fake_load)

    selected = artifacts.latest_compatible_joblib_path(tmp_path, "baseline_logreg_*.joblib")
    assert selected == older
