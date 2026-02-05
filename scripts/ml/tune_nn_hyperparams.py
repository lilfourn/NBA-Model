"""Optuna hyperparameter tuning for the NN (GRU-Attention) model.

Tunes: learning_rate, weight_decay, dropout, mlp_hidden, seq_hidden,
batch_size, history_len. Uses walk-forward CV with early stopping.
Saves best params to data/tuning/best_params_nn.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.nn.dataset import NNDataset  # noqa: E402
from app.ml.nn.features import build_training_data  # noqa: E402
from app.ml.nn.model import GRUAttentionTabularClassifier  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _walk_forward_splits(frame, n_folds: int = 3):
    """Time-ordered walk-forward CV splits returning index arrays."""
    df = frame.copy()
    df["_sort_key"] = df.index  # already time-sorted from build_training_data
    sorted_idx = df.sort_values("fetched_at").index
    n = len(sorted_idx)
    init_train_end = int(n * 0.4)
    remaining = n - init_train_end
    fold_size = remaining // n_folds

    splits = []
    for fold in range(n_folds):
        test_start = init_train_end + fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        train_idx = sorted_idx[:test_start]
        test_idx = sorted_idx[test_start:test_end]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _train_and_evaluate(
    data,
    train_idx,
    test_idx,
    *,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    mlp_hidden: int,
    seq_hidden: int,
    batch_size: int,
    epochs: int = 15,
    patience: int = 3,
    device: str = "cpu",
) -> float:
    """Train NN on one fold and return validation log-loss."""
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

    cat_cardinalities = [len(m) + 1 for m in data.cat_maps.values()]
    model = GRUAttentionTabularClassifier(
        num_numeric=data.numeric.shape[1],
        cat_cardinalities=cat_cardinalities,
        cat_emb_dims=[8, 8, 4],
        seq_d_in=2,
        seq_hidden=seq_hidden,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    bad_epochs = 0

    for _ in range(epochs):
        # Train
        model.train()
        for cat_ids, x_num, x_seq, y in train_loader:
            cat_ids, x_num, x_seq, y = (t.to(device) for t in (cat_ids, x_num, x_seq, y))
            optimizer.zero_grad()
            logits = model(cat_ids, x_num, x_seq)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Eval
        model.eval()
        eps = 1e-6
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for cat_ids, x_num, x_seq, y in test_loader:
                cat_ids, x_num, x_seq, y = (t.to(device) for t in (cat_ids, x_num, x_seq, y))
                logits = model(cat_ids, x_num, x_seq)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                labels = y.cpu().numpy().ravel()
                fold_loss = -np.mean(
                    labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)
                )
                total_loss += fold_loss * len(labels)
                count += len(labels)

        val_loss = total_loss / max(count, 1)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return best_loss


def _objective(trial: optuna.Trial, data, splits, device: str) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    mlp_hidden = trial.suggest_categorical("mlp_hidden", [64, 128, 256])
    seq_hidden = trial.suggest_categorical("seq_hidden", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    losses = []
    for train_idx, test_idx in splits:
        loss = _train_and_evaluate(
            data, train_idx, test_idx,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            mlp_hidden=mlp_hidden,
            seq_hidden=seq_hidden,
            batch_size=batch_size,
            device=device,
        )
        losses.append(loss)

    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna NN hyperparameter tuning.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--history-len", type=int, default=10)
    ap.add_argument("--output-dir", default="data/tuning")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building training data (history_len={args.history_len})...")
    data = build_training_data(engine=engine, history_len=args.history_len)
    if data.frame.empty or len(data.frame) < 200:
        raise SystemExit(f"Not enough data for tuning ({len(data.frame)} rows).")

    splits = _walk_forward_splits(data.frame, n_folds=args.n_folds)
    if not splits:
        raise SystemExit("Could not create walk-forward CV splits.")
    print(f"Walk-forward CV: {len(splits)} folds, sizes: {[len(s[1]) for s in splits]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTuning NN ({args.n_trials} trials, device={device})...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: _objective(t, data, splits, device), n_trials=args.n_trials)

    best = study.best_params
    print(f"NN best logloss: {study.best_value:.4f}")
    print(f"NN best params: {best}")

    out_path = output_dir / "best_params_nn.json"
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
