from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.nn import infer as nn_infer


class _FakeModel:
    def eval(self):
        return self

    def to(self, _device: str):
        return self

    def __call__(self, _cat_ids, x_num, _x_seq):
        assert int(x_num.shape[1]) == 2
        return torch.zeros((x_num.shape[0], 1), dtype=torch.float32)


def test_infer_aligns_numeric_columns_to_checkpoint(monkeypatch) -> None:
    payload = {
        "cat_maps": {"stat_type": {"Points": 1}},
        "numeric_cols": ["line_score", "minutes_to_start"],
        "history_len": 10,
    }

    frame = pd.DataFrame(
        [
            {
                "projection_id": "p1",
                "player_id": "pl1",
                "player_name": "Player 1",
                "stat_type": "Points",
                "line_score": 20.5,
            }
        ]
    )
    numeric = pd.DataFrame(
        [{"line_score": 0.4, "minutes_to_start": 0.2, "new_feature": 9.9}]
    )
    sequences = np.zeros((1, 10, 2), dtype=np.float32)

    monkeypatch.setattr(nn_infer.torch, "load", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        nn_infer,
        "build_inference_data",
        lambda **_kwargs: (frame.copy(), numeric.copy(), sequences.copy()),
    )
    monkeypatch.setattr(
        nn_infer,
        "load_model",
        lambda *_args, **_kwargs: (
            _FakeModel(),
            {
                "cat_maps": payload["cat_maps"],
                "history_len": 10,
                "temperature": 1.0,
                "numeric_cols": payload["numeric_cols"],
            },
        ),
    )
    monkeypatch.setattr(nn_infer.torch.cuda, "is_available", lambda: False)

    result = nn_infer.infer_over_probs(
        engine=object(),
        model_path="dummy.pt",
        snapshot_id="snapshot-1",
    )

    assert list(result.numeric.columns) == ["line_score", "minutes_to_start"]
    assert len(result.probs) == 1


def test_latest_checkpoint_skips_payload_without_numeric_schema(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "nn_gru_attention_20240201_000000Z.pt"
    candidate.write_text("candidate", encoding="utf-8")

    monkeypatch.setattr(nn_infer.torch, "load", lambda *_args, **_kwargs: {"cat_maps": {}, "state_dict": {}})

    assert nn_infer.latest_compatible_checkpoint(tmp_path) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"cat_maps": {"a": {"x": 1}}, "state_dict": {}, "numeric_cols": []},
        {"cat_maps": {"a": {"x": 1}}, "state_dict": {}, "numeric_stats": {}},
    ],
)
def test_is_compatible_payload_requires_numeric_schema(payload: dict) -> None:
    assert nn_infer._is_compatible_payload(payload) is False
