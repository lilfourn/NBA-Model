from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.tabdl.train import train_tabdl  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _sample_hidden_dims() -> tuple[int, ...]:
    choices = [
        (64, 32),
        (128, 64),
        (192, 96),
        (256, 128),
        (256, 128, 64),
    ]
    return random.choice(choices)


def _sample_params() -> dict[str, object]:
    return {
        "batch_size": random.choice([128, 256, 512]),
        "epochs": random.choice([16, 20, 24, 28, 32]),
        "learning_rate": 10 ** random.uniform(-4.2, -1.9),
        "weight_decay": 10 ** random.uniform(-6.0, -3.0),
        "patience": random.choice([5, 7, 9]),
        "hidden_dims": _sample_hidden_dims(),
        "dropout": random.uniform(0.0, 0.25),
        "use_pos_weight": random.choice([True, False]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Random-search tuning for TabDL.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/tuning/best_params_tabdl.json")
    ap.add_argument("--model-dir", "--models-dir", default="models")
    ap.add_argument("--no-train-best", action="store_true")
    args = ap.parse_args()

    load_env()
    random.seed(int(args.seed))
    engine = get_engine(args.database_url)

    best_params: dict[str, object] | None = None
    best_auc = float("-inf")
    best_logloss = float("inf")

    for trial in range(1, max(1, int(args.trials)) + 1):
        params = _sample_params()
        try:
            result = train_tabdl(
                engine=engine,
                model_dir=Path(args.model_dir),
                batch_size=int(params["batch_size"]),
                epochs=int(params["epochs"]),
                learning_rate=float(params["learning_rate"]),
                weight_decay=float(params["weight_decay"]),
                patience=int(params["patience"]),
                hidden_dims=tuple(int(v) for v in params["hidden_dims"]),  # type: ignore[arg-type]
                dropout=float(params["dropout"]),
                use_pos_weight=bool(params["use_pos_weight"]),
                persist=False,
                use_tuned_params=False,
                seed=int(args.seed),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[tabdl tune] trial {trial:02d}: failed ({exc.__class__.__name__}: {exc})")
            continue

        metrics = result.metrics or {}
        auc_raw = metrics.get("roc_auc")
        auc = float(auc_raw) if auc_raw is not None else float("-inf")
        logloss = float(metrics.get("logloss", 1e9))
        print(
            f"[tabdl tune] trial {trial:02d}: auc={auc:.4f} logloss={logloss:.4f} "
            f"params={params}"
        )

        better = False
        if auc > best_auc + 1e-6:
            better = True
        elif abs(auc - best_auc) <= 1e-6 and logloss < best_logloss - 1e-6:
            better = True
        if better:
            best_auc = auc
            best_logloss = logloss
            best_params = {
                "batch_size": int(params["batch_size"]),
                "epochs": int(params["epochs"]),
                "learning_rate": float(params["learning_rate"]),
                "weight_decay": float(params["weight_decay"]),
                "patience": int(params["patience"]),
                "hidden_dims": [int(v) for v in params["hidden_dims"]],  # type: ignore[index]
                "dropout": float(params["dropout"]),
                "use_pos_weight": bool(params["use_pos_weight"]),
                "seed": int(args.seed),
            }

    if best_params is None:
        raise SystemExit("No successful TabDL tuning trials.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"[tabdl tune] best auc={best_auc:.4f} logloss={best_logloss:.4f}")
    print(f"[tabdl tune] wrote best params -> {out_path}")

    if args.no_train_best:
        return

    best = train_tabdl(
        engine=engine,
        model_dir=Path(args.model_dir),
        batch_size=int(best_params["batch_size"]),
        epochs=int(best_params["epochs"]),
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(best_params["weight_decay"]),
        patience=int(best_params["patience"]),
        hidden_dims=tuple(int(v) for v in best_params["hidden_dims"]),  # type: ignore[arg-type]
        dropout=float(best_params["dropout"]),
        use_pos_weight=bool(best_params["use_pos_weight"]),
        persist=True,
        use_tuned_params=False,
        seed=int(best_params.get("seed", args.seed)),
    )
    print(
        "[tabdl tune] retrained best model:",
        {"rows": best.rows, "metrics": best.metrics, "model_path": best.model_path},
    )


if __name__ == "__main__":
    main()
