from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.modeling.probability import confidence_from_probability
from app.ml.nn.dataset import NNDataset
from app.ml.nn.features import build_inference_data
from app.ml.nn.model import GRUAttentionTabularClassifier


@dataclass
class InferenceResult:
    frame: Any
    numeric: Any
    probs: np.ndarray


def load_model(
    path: str,
    num_numeric: int,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[GRUAttentionTabularClassifier, dict[str, Any]]:
    if payload is None:
        payload = torch.load(path, map_location="cpu")
    cat_maps = payload["cat_maps"]
    cat_emb_dims = payload.get("cat_emb_dims", [8, 8, 4])
    cat_cardinalities = [len(mapping) + 1 for mapping in cat_maps.values()]
    history_len = int(payload.get("history_len", 10))
    temperature = float(payload.get("temperature") or 1.0)
    if temperature <= 0:
        temperature = 1.0

    model = GRUAttentionTabularClassifier(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=cat_emb_dims,
        seq_d_in=2,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {"cat_maps": cat_maps, "history_len": history_len, "temperature": temperature}


@torch.no_grad()
def infer_over_probs(
    *,
    engine,
    model_path: str,
    snapshot_id: str,
    device: str | None = None,
) -> InferenceResult:
    payload = torch.load(model_path, map_location="cpu")
    cat_maps = payload["cat_maps"]
    history_len = int(payload.get("history_len", 10))
    numeric_stats = payload.get("numeric_stats")
    temperature = float(payload.get("temperature") or 1.0)
    if temperature <= 0:
        temperature = 1.0

    frame, numeric, sequences = build_inference_data(
        engine=engine,
        snapshot_id=snapshot_id,
        history_len=history_len,
        cat_maps=cat_maps,
        numeric_stats=numeric_stats,
    )
    if frame.empty:
        return InferenceResult(
            frame=frame,
            numeric=numeric,
            probs=np.zeros((0,), dtype=np.float32),
        )

    num_numeric = numeric.shape[1]
    model, info = load_model(model_path, num_numeric=num_numeric, payload=payload)
    cat_maps = info["cat_maps"]
    temperature = float(info.get("temperature") or temperature)
    if temperature <= 0:
        temperature = 1.0

    dataset = NNDataset(
        frame=frame,
        numeric=numeric,
        sequences=sequences,
        cat_maps=cat_maps,
        cat_keys=list(cat_maps.keys()),
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    probs = []
    for cat_ids, x_num, x_seq, _ in loader:
        cat_ids = cat_ids.to(device)
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)
        logits = model(cat_ids, x_num, x_seq)
        probs.append(torch.sigmoid(logits / temperature).cpu().numpy().reshape(-1))
    return InferenceResult(frame=frame, numeric=numeric, probs=np.concatenate(probs))


def format_predictions(frame, probs: np.ndarray) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, prob in enumerate(probs):
        row = frame.iloc[idx]
        pick = "OVER" if prob >= 0.5 else "UNDER"
        results.append(
            {
                "projection_id": row.get("projection_id"),
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name") or row.get("prizepicks_name_key"),
                "stat_type": row.get("stat_type"),
                "line_score": float(row.get("line_score") or 0.0),
                "pick": pick,
                "prob_over": float(prob),
                "confidence": float(confidence_from_probability(float(prob))),
            }
        )
    return results
