from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.calibration import CalibratedExpert, PlattCalibrator, load_calibrator
from app.ml.dataset import _add_history_features
from app.ml.tabdl.model import TabularMLPClassifier, default_embedding_dims
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS


@dataclass
class InferenceResult:
    frame: pd.DataFrame
    probs: np.ndarray
    model_path: str


def _load_inference_frame(engine: Engine, snapshot_id: str) -> pd.DataFrame:
    query = text(
        """
        select
            pf.snapshot_id,
            pf.projection_id,
            pf.player_id,
            pf.game_id,
            pf.line_score,
            pf.line_score_prev,
            pf.line_score_delta,
            pf.line_movement,
            pf.stat_type,
            pf.projection_type,
            pf.odds_type,
            pf.trending_count,
            pf.is_promo,
            pf.is_live,
            pf.in_game,
            pf.today,
            pf.minutes_to_start,
            pf.fetched_at,
            pf.start_time,
            pl.name_key as prizepicks_name_key,
            pl.display_name as player_name,
            pl.combo as combo
        from projection_features pf
        join projections p
            on p.snapshot_id = pf.snapshot_id
            and p.projection_id = pf.projection_id
        join players pl on pl.id = pf.player_id
        where pf.snapshot_id = :snapshot_id
          and coalesce(p.odds_type, 0) = 0
          and lower(coalesce(p.event_type, p.attributes->>'event_type', '')) <> 'combo'
          and (pl.combo is null or pl.combo = false)
          and (pf.is_live is null or pf.is_live = false)
          and (pf.in_game is null or pf.in_game = false)
          and (pf.minutes_to_start is null or pf.minutes_to_start >= 0)
        """
    )
    return pd.read_sql(query, engine, params={"snapshot_id": snapshot_id})


def _is_compatible_payload(payload: dict[str, Any]) -> bool:
    try:
        state_dict = payload["state_dict"]
        cat_maps = payload["cat_maps"]
    except KeyError:
        return False
    if not isinstance(state_dict, dict) or not isinstance(cat_maps, dict):
        return False
    cat_cols = list(payload.get("categorical_cols") or CATEGORICAL_COLS)
    num_cols = list(payload.get("numeric_cols") or NUMERIC_COLS)
    cat_cardinalities = [len(cat_maps.get(col, {})) + 1 for col in cat_cols]
    model_cfg = payload.get("model_config") or {}
    emb_dims = list(model_cfg.get("emb_dims") or default_embedding_dims(cat_cardinalities))
    hidden_dims = tuple(int(v) for v in model_cfg.get("hidden_dims") or [256, 128])
    dropout = float(model_cfg.get("dropout") or 0.2)

    try:
        model = TabularMLPClassifier(
            num_numeric=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            cat_emb_dims=emb_dims,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        model.load_state_dict(state_dict, strict=False)
    except Exception:  # noqa: BLE001
        return False
    return True


def latest_compatible_checkpoint(models_dir: Path, pattern: str = "tabdl_*.pt") -> Path | None:
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


def _load_model(path: str) -> tuple[TabularMLPClassifier, dict[str, Any], CalibratedExpert | PlattCalibrator | None]:
    payload = torch.load(path, map_location="cpu")
    cat_maps = payload["cat_maps"]
    cat_cols = list(payload.get("categorical_cols") or CATEGORICAL_COLS)
    num_cols = list(payload.get("numeric_cols") or NUMERIC_COLS)
    cat_cardinalities = [len(cat_maps.get(col, {})) + 1 for col in cat_cols]
    model_cfg = payload.get("model_config") or {}
    emb_dims = list(model_cfg.get("emb_dims") or default_embedding_dims(cat_cardinalities))
    hidden_dims = tuple(int(v) for v in model_cfg.get("hidden_dims") or [256, 128])
    dropout = float(model_cfg.get("dropout") or 0.2)

    model = TabularMLPClassifier(
        num_numeric=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=emb_dims,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    calibrator = None
    if "isotonic" in payload:
        try:
            calibrator = load_calibrator(payload["isotonic"])
        except Exception:  # noqa: BLE001
            pass
    return model, payload, calibrator


def _encode_inference_features(
    frame: pd.DataFrame,
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    cat_cols = list(payload.get("categorical_cols") or CATEGORICAL_COLS)
    num_cols = list(payload.get("numeric_cols") or NUMERIC_COLS)
    cat_maps: dict[str, dict[str, int]] = payload.get("cat_maps") or {}

    work = frame.copy()
    for col in cat_cols:
        if col not in work.columns:
            work[col] = "unknown"
        work[col] = work[col].fillna("unknown").astype(str)

    for col in num_cols:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[num_cols] = work[num_cols].fillna(0.0).astype(float)
    if "trending_count" in work.columns:
        work["trending_count"] = np.log1p(work["trending_count"].clip(lower=0.0))

    cat_ids = np.zeros((len(work), len(cat_cols)), dtype=np.int64)
    for idx, col in enumerate(cat_cols):
        mapping = cat_maps.get(col, {})
        values = work[col].tolist()
        cat_ids[:, idx] = np.array([mapping.get(str(v), 0) for v in values], dtype=np.int64)

    x_num = work[num_cols].to_numpy(dtype=np.float32, copy=True)
    numeric_stats = payload.get("numeric_stats") or {}
    mean = np.array(numeric_stats.get("mean") or [], dtype=np.float32)
    std = np.array(numeric_stats.get("std") or [], dtype=np.float32)
    if mean.shape[0] == x_num.shape[1] and std.shape[0] == x_num.shape[1]:
        safe_std = np.where(std < 1e-6, 1.0, std)
        x_num = (x_num - mean) / safe_std
    return cat_ids, x_num


def infer_over_probs(*, engine: Engine, model_path: str, snapshot_id: str, device: str | None = None) -> InferenceResult:
    frame = _load_inference_frame(engine, snapshot_id)
    if frame.empty:
        return InferenceResult(frame=frame, probs=np.zeros((0,), dtype=np.float32), model_path=model_path)

    frame = _add_history_features(frame, engine)
    model, payload, calibrator = _load_model(model_path)
    cat_ids, x_num = _encode_inference_features(frame, payload)

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(run_device)

    cat_tensor = torch.from_numpy(cat_ids).long()
    num_tensor = torch.from_numpy(x_num).float()

    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        batch_size = 512
        for start in range(0, len(frame), batch_size):
            end = start + batch_size
            batch_cat = cat_tensor[start:end].to(run_device)
            batch_num = num_tensor[start:end].to(run_device)
            logits = model(batch_cat, batch_num)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(np.float32)
            probs_list.append(probs)

    probs_out = np.concatenate(probs_list) if probs_list else np.zeros((0,), dtype=np.float32)
    if calibrator is not None and len(probs_out):
        probs_out = calibrator.transform(probs_out)
    return InferenceResult(frame=frame, probs=probs_out, model_path=model_path)

