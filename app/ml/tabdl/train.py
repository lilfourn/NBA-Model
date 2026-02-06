from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.db import schema
from app.ml.calibration import best_calibrator
from app.ml.dataset import load_training_data
from app.ml.stat_mappings import stat_value_from_row
from app.ml.tabdl.model import TabularMLPClassifier, default_embedding_dims
from app.ml.train import CATEGORICAL_COLS, MIN_TRAIN_ROWS, NUMERIC_COLS, _time_split
from app.modeling.conformal import ConformalCalibrator


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int


def _load_tuned_params() -> dict[str, Any]:
    candidates: list[Path] = []
    tuning_dir = os.getenv("TUNING_DIR", "").strip()
    if tuning_dir:
        candidates.append(Path(tuning_dir) / "best_params_tabdl.json")
    candidates.append(Path("data/tuning/best_params_tabdl.json"))
    candidates.append(Path("/state/data/tuning/best_params_tabdl.json"))
    import json
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            return data
    return {}


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    frame = df.copy()
    if "is_combo" in frame.columns:
        frame = frame[frame["is_combo"].fillna(False) == False]  # noqa: E712
    if "player_name" in frame.columns:
        frame = frame[~frame["player_name"].fillna("").astype(str).str.contains("+", regex=False)]

    frame["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in frame.itertuples(index=False)
    ]
    frame = frame.dropna(subset=["line_score", "actual_value", "fetched_at"])

    if "minutes_to_start" in frame.columns:
        frame = frame[frame["minutes_to_start"].fillna(0) >= 0]
    if "is_live" in frame.columns:
        frame = frame[frame["is_live"].fillna(False) == False]  # noqa: E712
    if "in_game" in frame.columns:
        frame = frame[frame["in_game"].fillna(False) == False]  # noqa: E712

    frame["over"] = (frame["actual_value"] > frame["line_score"]).astype(int)

    dedup_cols = ["player_id", "nba_game_id", "stat_type"]
    if all(col in frame.columns for col in dedup_cols):
        frame = frame.sort_values("fetched_at").drop_duplicates(subset=dedup_cols, keep="first")

    frame[CATEGORICAL_COLS] = frame[CATEGORICAL_COLS].fillna("unknown").astype(str)
    for col in NUMERIC_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[NUMERIC_COLS] = frame[NUMERIC_COLS].fillna(0.0).astype(float)
    if "trending_count" in frame.columns:
        frame["trending_count"] = np.log1p(frame["trending_count"].clip(lower=0.0))

    X = frame[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = frame["over"]
    return X, y, frame


def _build_cat_maps(train_frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    cat_maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        values = sorted({str(v) for v in train_frame[col].fillna("unknown").astype(str).tolist()})
        cat_maps[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return cat_maps


def _encode_cats(frame: pd.DataFrame, cat_maps: dict[str, dict[str, int]]) -> np.ndarray:
    encoded = np.zeros((len(frame), len(CATEGORICAL_COLS)), dtype=np.int64)
    for idx, col in enumerate(CATEGORICAL_COLS):
        mapping = cat_maps.get(col, {})
        values = frame[col].fillna("unknown").astype(str).tolist()
        encoded[:, idx] = np.array([mapping.get(value, 0) for value in values], dtype=np.int64)
    return encoded


def _build_loader(
    cat_ids: np.ndarray,
    numeric: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(cat_ids).long(),
        torch.from_numpy(numeric).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)


def _predict_probs(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for cat_ids, x_num, y in loader:
            cat_ids = cat_ids.to(device)
            x_num = x_num.to(device)
            logits = model(cat_ids, x_num)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probs.append(batch_probs)
            labels.append(y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs), np.concatenate(labels)


def train_tabdl(
    engine,
    model_dir: Path,
    *,
    batch_size: int = 512,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.2,
    device: str | None = None,
    persist: bool = True,
    use_tuned_params: bool = True,
    use_pos_weight: bool = True,
    seed: int = 42,
) -> TrainResult:
    tuned = _load_tuned_params() if use_tuned_params else {}
    if tuned:
        batch_size = int(tuned.get("batch_size", batch_size))
        epochs = int(tuned.get("epochs", epochs))
        learning_rate = float(tuned.get("learning_rate", learning_rate))
        weight_decay = float(tuned.get("weight_decay", weight_decay))
        patience = int(tuned.get("patience", patience))
        tuned_hidden = tuned.get("hidden_dims")
        if isinstance(tuned_hidden, list) and tuned_hidden:
            hidden_dims = tuple(int(v) for v in tuned_hidden)
        dropout = float(tuned.get("dropout", dropout))
        use_pos_weight = bool(tuned.get("use_pos_weight", use_pos_weight))
        seed = int(tuned.get("seed", seed))

    # Stabilize TabDL runs across retrains.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    raw = load_training_data(engine)
    if raw.empty:
        raise RuntimeError("No training data available. Did you load NBA stats and build features?")

    X, y, used = _prepare_features(raw)
    if used.empty:
        raise RuntimeError("Not enough training data after cleaning.")
    if y.nunique() < 2:
        raise RuntimeError("Not enough class variety to train yet.")
    if len(used) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Not enough training data available yet (rows={len(used)}).")

    X_train, X_test, y_train, y_test = _time_split(used, X, y)
    if len(X_train) < 30 or len(X_test) < 10:
        raise RuntimeError("Not enough rows after holdout split for tabular deep model.")

    train_frame = X_train.copy()
    test_frame = X_test.copy()
    cat_maps = _build_cat_maps(train_frame)
    train_cat = _encode_cats(train_frame, cat_maps)
    test_cat = _encode_cats(test_frame, cat_maps)

    train_num = X_train[NUMERIC_COLS].to_numpy(dtype=np.float32, copy=True)
    test_num = X_test[NUMERIC_COLS].to_numpy(dtype=np.float32, copy=True)
    num_mean = train_num.mean(axis=0, dtype=np.float64).astype(np.float32)
    num_std = train_num.std(axis=0, dtype=np.float64).astype(np.float32)
    num_std = np.where(num_std < 1e-6, 1.0, num_std).astype(np.float32)
    train_num = (train_num - num_mean) / num_std
    test_num = (test_num - num_mean) / num_std

    train_y = y_train.to_numpy(dtype=np.float32)
    test_y = y_test.to_numpy(dtype=np.float32)

    train_loader = _build_loader(train_cat, train_num, train_y, batch_size=batch_size, shuffle=True)
    test_loader = _build_loader(test_cat, test_num, test_y, batch_size=batch_size, shuffle=False)

    cat_cardinalities = [len(cat_maps[col]) + 1 for col in CATEGORICAL_COLS]
    emb_dims = default_embedding_dims(cat_cardinalities)
    model = TabularMLPClassifier(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=emb_dims,
        hidden_dims=hidden_dims,
        dropout=float(dropout),
    )

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    pos_weight = max(1.0, neg / max(pos, 1.0))
    if use_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=run_device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for _ in range(int(epochs)):
        model.train()
        for cat_ids, x_num, labels in train_loader:
            cat_ids = cat_ids.to(run_device)
            x_num = x_num.to(run_device)
            labels = labels.to(run_device)
            optimizer.zero_grad()
            logits = model(cat_ids, x_num)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        probs, labels_np = _predict_probs(model, test_loader, run_device)
        val_loss = float(log_loss(labels_np.astype(int), np.clip(probs, 1e-6, 1.0 - 1e-6)))
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("Tabular deep model failed to converge.")
    model.load_state_dict(best_state)

    probs, labels_np = _predict_probs(model, test_loader, run_device)
    labels_int = labels_np.astype(int)
    preds = (probs >= 0.5).astype(int)

    conformal = ConformalCalibrator.calibrate(probs, labels_int, alpha=0.10)
    calibrator_data = None
    try:
        calibrator = best_calibrator(probs, labels_int)
        calibrator_data = calibrator.to_dict()
    except ValueError:
        pass

    metrics = {
        "accuracy": float(accuracy_score(labels_int, preds)),
        "roc_auc": float(roc_auc_score(labels_int, probs)) if len(np.unique(labels_int)) > 1 else None,
        "logloss": float(log_loss(labels_int, np.clip(probs, 1e-6, 1.0 - 1e-6))),
        "seed": int(seed),
        "conformal_q_hat": conformal.q_hat,
        "conformal_n_cal": conformal.n_cal,
    }

    model_path = model_dir / "tabdl_tmp.pt"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if persist:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"tabdl_{timestamp}.pt"
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "categorical_cols": list(CATEGORICAL_COLS),
        "numeric_cols": list(NUMERIC_COLS),
        "cat_maps": cat_maps,
        "numeric_stats": {
            "mean": num_mean.tolist(),
            "std": num_std.tolist(),
        },
        "model_config": {
            "hidden_dims": [int(v) for v in hidden_dims],
            "dropout": float(dropout),
            "emb_dims": emb_dims,
        },
        "conformal": {
            "alpha": conformal.alpha,
            "q_hat": conformal.q_hat,
            "n_cal": conformal.n_cal,
        },
    }
    if calibrator_data:
        payload["isotonic"] = calibrator_data
    if persist:
        torch.save(payload, model_path)

        try:
            from app.ml.artifact_store import upload_file
            upload_file(engine, model_name="tabdl_mlp", file_path=model_path)
        except Exception:  # noqa: BLE001
            pass

        with engine.begin() as conn:
            conn.execute(
                schema.model_runs.insert().values(
                    {
                        "id": uuid4(),
                        "created_at": datetime.now(timezone.utc),
                        "model_name": "tabdl_mlp",
                        "train_rows": int(len(used)),
                        "metrics": metrics,
                        "params": {
                            "batch_size": int(batch_size),
                            "epochs": int(epochs),
                            "learning_rate": float(learning_rate),
                            "weight_decay": float(weight_decay),
                            "patience": int(patience),
                            "hidden_dims": [int(v) for v in hidden_dims],
                            "dropout": float(dropout),
                            "use_pos_weight": bool(use_pos_weight),
                            "seed": int(seed),
                        },
                        "artifact_path": str(model_path),
                    }
                )
            )

    return TrainResult(model_path=str(model_path) if persist else "", metrics=metrics, rows=int(len(used)))
