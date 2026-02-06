from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from app.db import schema
from app.ml.calibration import best_calibrator
from app.ml.nn.dataset import NNDataset
from app.ml.nn.features import TrainingData, build_training_data
from app.ml.nn.model import GRUAttentionTabularClassifier
from app.modeling.conformal import ConformalCalibrator


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int
    cat_maps: dict[str, dict[str, int]]
    numeric_cols: list[str]


def _time_split(frame, holdout_ratio: float = 0.2):
    df_sorted = frame.sort_values("fetched_at")
    cutoff = int(len(df_sorted) * (1 - holdout_ratio))
    train_idx = df_sorted.index[:cutoff]
    test_idx = df_sorted.index[cutoff:]
    return train_idx, test_idx


def _train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    count = 0
    for cat_ids, x_num, x_seq, y in loader:
        cat_ids = cat_ids.to(device)
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(cat_ids, x_num, x_seq)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch = y.size(0)
        total += float(loss.item()) * batch
        count += batch
    return total / max(count, 1)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probs = []
    labels = []
    for cat_ids, x_num, x_seq, y in loader:
        cat_ids = cat_ids.to(device)
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)

        logits = model(cat_ids, x_num, x_seq)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs.append(p)
        labels.append(y.numpy().reshape(-1))
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def _predict_logits(model, loader, device):
    model.eval()
    logits_out = []
    labels_out = []
    for cat_ids, x_num, x_seq, y in loader:
        cat_ids = cat_ids.to(device)
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)

        logits = model(cat_ids, x_num, x_seq).cpu().numpy().reshape(-1)
        logits_out.append(logits)
        labels_out.append(y.numpy().reshape(-1))
    return np.concatenate(logits_out), np.concatenate(labels_out)


def _temperature_scale(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Fit temperature T>0 by grid-search to minimize NLL on a holdout set.
    Returns (best_T, best_logloss).
    """
    eps = 1e-6
    # Search a wide but reasonable range; >1 typically reduces overconfidence.
    candidates = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.0, 10.0, 15)[1:]])
    best_T = 1.0
    best_loss = float("inf")
    for T in candidates:
        p = 1.0 / (1.0 + np.exp(-(logits / float(T))))
        loss = -float(np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T, best_loss


def _load_tuned_params() -> dict[str, Any]:
    """Load Optuna-tuned NN params if available."""
    candidates: list[Path] = []
    tuning_dir = os.getenv("TUNING_DIR", "").strip()
    if tuning_dir:
        candidates.append(Path(tuning_dir) / "best_params_nn.json")
    candidates.append(Path("data/tuning/best_params_nn.json"))
    candidates.append(Path("/state/data/tuning/best_params_nn.json"))

    import json
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return {}


def train_nn(
    *,
    engine,
    model_dir: Path,
    history_len: int = 10,
    batch_size: int = 512,
    epochs: int = 25,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: str | None = None,
) -> TrainResult:
    # Override defaults with tuned params if available
    tuned = _load_tuned_params()
    if tuned:
        learning_rate = tuned.get("learning_rate", learning_rate)
        weight_decay = tuned.get("weight_decay", weight_decay)
        batch_size = tuned.get("batch_size", batch_size)

    dropout = tuned.get("dropout", 0.3)
    mlp_hidden = tuned.get("mlp_hidden", 128)
    seq_hidden = tuned.get("seq_hidden", 64)

    data: TrainingData = build_training_data(engine=engine, history_len=history_len)
    if data.frame.empty:
        raise RuntimeError("No training data available. Did you load NBA stats and build features?")

    train_idx, test_idx = _time_split(data.frame)
    train_ds = NNDataset(
        frame=data.frame.loc[train_idx],
        numeric=data.numeric.loc[train_idx],
        sequences=data.sequences[train_idx],
        cat_maps=data.cat_maps,
        cat_keys=list(data.cat_maps.keys()),
    )
    test_ds = NNDataset(
        frame=data.frame.loc[test_idx],
        numeric=data.numeric.loc[test_idx],
        sequences=data.sequences[test_idx],
        cat_maps=data.cat_maps,
        cat_keys=list(data.cat_maps.keys()),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    cat_cardinalities = [len(mapping) + 1 for mapping in data.cat_maps.values()]
    cat_emb_dims = [8, 8, 4]
    model = GRUAttentionTabularClassifier(
        num_numeric=data.numeric.shape[1],
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=cat_emb_dims,
        seq_d_in=2,
        seq_hidden=seq_hidden,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Warm-start: try to load weights from previous best checkpoint
    from app.ml.nn.infer import latest_compatible_checkpoint as _find_prev
    prev_ckpt = _find_prev(model_dir)
    if prev_ckpt is not None:
        try:
            prev_payload = torch.load(str(prev_ckpt), map_location="cpu")
            model.load_state_dict(prev_payload["state_dict"], strict=False)
            print(f"Warm-start: loaded weights from {prev_ckpt.name}")
        except Exception:  # noqa: BLE001
            print("Warm-start: previous checkpoint incompatible, training from scratch.")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(epochs):
        _train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        probs, labels = _predict(model, test_loader, device)
        eps = 1e-6
        logloss = -np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps))

        if logloss < best_loss - 1e-4:
            best_loss = logloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs, labels = _predict(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)
    logits, labels_logits = _predict_logits(model, test_loader, device)
    # labels should match
    if labels_logits.shape == labels.shape:
        labels = labels_logits
    temperature, logloss_cal = _temperature_scale(logits, labels)
    probs_cal = 1.0 / (1.0 + np.exp(-(logits / temperature)))
    preds_cal = (probs_cal >= 0.5).astype(int)

    conformal = ConformalCalibrator.calibrate(probs_cal, labels.astype(int), alpha=0.10)

    calibrator_data = None
    try:
        calibrator = best_calibrator(probs_cal, labels.astype(int))
        calibrator_data = calibrator.to_dict()
    except ValueError:
        pass

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else None,
        "logloss": float(best_loss),
        "temperature": float(temperature),
        "logloss_cal": float(logloss_cal),
        "accuracy_cal": float(accuracy_score(labels, preds_cal)),
        "conformal_q_hat": conformal.q_hat,
        "conformal_n_cal": conformal.n_cal,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"nn_gru_attention_{timestamp}.pt"
    payload = {
        "state_dict": model.state_dict(),
        "cat_maps": data.cat_maps,
        "numeric_cols": data.numeric_cols,
        "numeric_stats": data.numeric_stats,
        "history_len": history_len,
        "cat_emb_dims": cat_emb_dims,
        "temperature": float(temperature),
        "conformal": {"alpha": conformal.alpha, "q_hat": conformal.q_hat, "n_cal": conformal.n_cal},
    }
    if calibrator_data:
        payload["isotonic"] = calibrator_data
    torch.save(payload, model_path)

    try:
        from app.ml.artifact_store import upload_file
        upload_file(engine, model_name="nn_gru_attention", file_path=model_path)
        print(f"Uploaded nn_gru_attention artifact to DB ({model_path})")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: DB upload failed for nn_gru_attention: {exc}")

    run_id = uuid4()
    with engine.begin() as conn:
        conn.execute(
            schema.model_runs.insert().values(
                {
                    "id": run_id,
                    "created_at": datetime.now(timezone.utc),
                    "model_name": "nn_gru_attention",
                    "train_rows": int(len(data.frame)),
                    "metrics": metrics,
                    "params": {
                        "history_len": history_len,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "patience": patience,
                    },
                    "artifact_path": str(model_path),
                }
            )
        )

    return TrainResult(
        model_path=str(model_path),
        metrics=metrics,
        rows=int(len(data.frame)),
        cat_maps=data.cat_maps,
        numeric_cols=data.numeric_cols,
    )
