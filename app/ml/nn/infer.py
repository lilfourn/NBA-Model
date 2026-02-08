from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.ml.calibration import CalibratedExpert, load_calibrator
from app.ml.nn.dataset import NNDataset
from app.ml.nn.features import build_inference_data
from app.ml.nn.model import GRUAttentionTabularClassifier
from app.modeling.probability import confidence_from_probability


@dataclass
class InferenceResult:
    frame: Any
    numeric: Any
    probs: np.ndarray


def _is_compatible_payload(payload: dict[str, Any]) -> bool:
    try:
        cat_maps = payload["cat_maps"]
        state_dict = payload["state_dict"]
    except KeyError:
        return False
    if not isinstance(cat_maps, dict):
        return False
    numeric_cols = payload.get("numeric_cols")
    if isinstance(numeric_cols, list):
        num_numeric = len(numeric_cols)
    else:
        numeric_stats = payload.get("numeric_stats")
        if isinstance(numeric_stats, dict):
            num_numeric = len(numeric_stats)
        else:
            return False
    if num_numeric <= 0:
        return False

    cat_emb_dims = payload.get("cat_emb_dims", [8, 8, 4])
    cat_cardinalities = [len(mapping) + 1 for mapping in cat_maps.values()]
    # Recover architecture hyper-parameters to match checkpoint exactly
    arch_kwargs: dict[str, Any] = {}
    for key in ("seq_hidden", "attn_dim", "mlp_hidden", "dropout", "seq_d_in"):
        if key in payload:
            arch_kwargs[key] = payload[key]
    if "seq_d_in" not in arch_kwargs:
        arch_kwargs["seq_d_in"] = 2
    try:
        model = GRUAttentionTabularClassifier(
            num_numeric=num_numeric,
            cat_cardinalities=cat_cardinalities,
            cat_emb_dims=cat_emb_dims,
            **arch_kwargs,
        )
        model.load_state_dict(state_dict)
    except Exception:  # noqa: BLE001
        return False
    return True


def latest_compatible_checkpoint(
    models_dir: Path, pattern: str = "nn_gru_attention_*.pt"
) -> Path | None:
    if not models_dir.exists():
        return None
    for candidate in sorted(models_dir.glob(pattern), reverse=True):
        try:
            payload = torch.load(str(candidate), map_location="cpu")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict) and _is_compatible_payload(payload):
            return candidate
    return None


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

    # Recover architecture hyper-parameters saved during training so the
    # model skeleton matches the checkpoint weights exactly.
    arch_kwargs: dict[str, Any] = {}
    for key in ("seq_hidden", "attn_dim", "mlp_hidden", "dropout", "seq_d_in"):
        if key in payload:
            arch_kwargs[key] = payload[key]
    if "seq_d_in" not in arch_kwargs:
        arch_kwargs["seq_d_in"] = 2

    model = GRUAttentionTabularClassifier(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=cat_emb_dims,
        **arch_kwargs,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {
        "cat_maps": cat_maps,
        "history_len": history_len,
        "temperature": temperature,
    }


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

    probs_list = []
    for cat_ids, x_num, x_seq, _ in loader:
        cat_ids = cat_ids.to(device)
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)
        logits = model(cat_ids, x_num, x_seq)
        probs_list.append(torch.sigmoid(logits / temperature).cpu().numpy().reshape(-1))
    probs = np.concatenate(probs_list)

    # Apply probability calibration if available (isotonic or Platt)
    cal_data = payload.get("isotonic")
    if cal_data is not None:
        try:
            calibrator = load_calibrator(cal_data)
            probs = calibrator.transform(probs)
        except Exception:  # noqa: BLE001
            pass

    return InferenceResult(frame=frame, numeric=numeric, probs=probs)


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
