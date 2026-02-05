from __future__ import annotations

from pathlib import Path

from app.ml.nn import infer as nn_infer


def test_latest_compatible_checkpoint_skips_incompatible(monkeypatch, tmp_path: Path) -> None:
    older = tmp_path / "nn_gru_attention_20240101_000000Z.pt"
    newer = tmp_path / "nn_gru_attention_20240102_000000Z.pt"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")

    def _fake_torch_load(path: str, map_location: str = "cpu"):
        return {"path": path}

    def _fake_is_compatible(payload):
        return not str(payload.get("path", "")).endswith(newer.name)

    monkeypatch.setattr(nn_infer.torch, "load", _fake_torch_load)
    monkeypatch.setattr(nn_infer, "_is_compatible_payload", _fake_is_compatible)

    selected = nn_infer.latest_compatible_checkpoint(tmp_path)
    assert selected == older


def test_latest_compatible_checkpoint_returns_none_when_no_valid(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "nn_gru_attention_20240103_000000Z.pt"
    candidate.write_text("candidate", encoding="utf-8")

    monkeypatch.setattr(nn_infer.torch, "load", lambda path, map_location="cpu": {"path": path})
    monkeypatch.setattr(nn_infer, "_is_compatible_payload", lambda payload: False)

    assert nn_infer.latest_compatible_checkpoint(tmp_path) is None
