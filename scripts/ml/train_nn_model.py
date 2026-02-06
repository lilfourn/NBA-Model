from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.nn.train import train_nn  # noqa: E402
from scripts.ml.train_baseline_model import load_env, report_training_data_state  # noqa: E402


def report_nn_data_state(engine) -> None:
    print("NN data preflight (non-combo)")
    print("=" * 80)
    queries = {
        "non_combo_projections": """
            select count(*)
            from projection_features pf
            join players pl on pl.id = pf.player_id
            where pl.combo is null or pl.combo = false
        """,
        "non_combo_with_game_match": """
            select count(*)
            from projection_features pf
            join players pl on pl.id = pf.player_id
            join games g on g.id = pf.game_id
            left join teams ht on ht.id = g.home_team_id
            left join teams at on at.id = g.away_team_id
            join nba_games ng
              on ng.game_date = (g.start_time at time zone 'America/New_York')::date
             and ng.home_team_abbreviation = ht.abbreviation
             and ng.away_team_abbreviation = at.abbreviation
            where pl.combo is null or pl.combo = false
        """,
        "non_combo_with_stats": """
            select count(*)
            from projection_features pf
            join players pl on pl.id = pf.player_id
            join games g on g.id = pf.game_id
            left join teams ht on ht.id = g.home_team_id
            left join teams at on at.id = g.away_team_id
            join nba_games ng
              on ng.game_date = (g.start_time at time zone 'America/New_York')::date
             and ng.home_team_abbreviation = ht.abbreviation
             and ng.away_team_abbreviation = at.abbreviation
            join nba_player_game_stats s on s.game_id = ng.id
            where pl.combo is null or pl.combo = false
        """,
    }
    with engine.connect() as conn:
        for label, sql in queries.items():
            try:
                count = conn.execute(text(sql)).scalar()
            except Exception as exc:  # noqa: BLE001
                count = f"ERR: {exc.__class__.__name__}"
            print(f"{label:28s} {count}")


def _safe_auc(metrics: dict[str, Any] | None) -> float:
    if not isinstance(metrics, dict):
        return float("-inf")
    value = metrics.get("roc_auc")
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _persist_recursive_summary(model_dir: Path, summary: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "nn_recursive_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _promote_best_checkpoint(model_dir: Path, best_model_path: str | None) -> str | None:
    if not best_model_path:
        return None
    src = Path(best_model_path)
    if not src.exists():
        return best_model_path
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    dst = model_dir / f"nn_gru_attention_{ts}.pt"
    if src.resolve() == dst.resolve():
        return str(src)
    shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU+attention NN model.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model-dir", "--models-dir", default="models")
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--round-patience", type=int, default=2)
    parser.add_argument("--min-roc-auc-improvement", type=float, default=0.001)
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    model_dir = Path(args.model_dir)

    rounds_requested = max(1, int(args.rounds))
    round_patience = max(1, int(args.round_patience))
    min_improvement = max(0.0, float(args.min_roc_auc_improvement))
    lr_decay = max(0.0, float(args.lr_decay))

    current_lr = float(args.learning_rate)
    no_improve_streak = 0
    best_auc = float("-inf")
    best_round: int | None = None
    best_model_path: str | None = None
    best_metrics: dict[str, Any] | None = None
    rounds: list[dict[str, Any]] = []

    try:
        for round_idx in range(1, rounds_requested + 1):
            result = train_nn(
                engine=engine,
                model_dir=model_dir,
                history_len=args.history_len,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=current_lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
            metrics = result.metrics or {}
            auc = _safe_auc(metrics)
            improved = auc > (best_auc + min_improvement)
            if auc > best_auc:
                best_auc = auc
                best_round = round_idx
                best_model_path = result.model_path
                best_metrics = metrics

            rounds.append(
                {
                    "round": round_idx,
                    "learning_rate": current_lr,
                    "rows": result.rows,
                    "roc_auc": None if auc == float("-inf") else auc,
                    "metrics": metrics,
                    "model_path": result.model_path,
                }
            )

            if round_idx > 1:
                if improved:
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
                    if no_improve_streak >= round_patience:
                        break

            current_lr *= lr_decay
    except RuntimeError as exc:
        if "No training data available" in str(exc):
            print("No training data available yet. Skipping training.")
            report_training_data_state(engine)
            report_nn_data_state(engine)
            return
        raise

    selected_model_path = _promote_best_checkpoint(model_dir, best_model_path)
    summary = {
        "rounds_requested": rounds_requested,
        "rounds_completed": len(rounds),
        "round_patience": round_patience,
        "min_roc_auc_improvement": min_improvement,
        "lr_decay": lr_decay,
        "best_round": best_round,
        "best_roc_auc": None if best_auc == float("-inf") else best_auc,
        "best_metrics": best_metrics,
        "selected_model_path": selected_model_path,
        "rounds": rounds,
    }
    summary_path = _persist_recursive_summary(model_dir, summary)

    print(
        {
            "rounds_completed": len(rounds),
            "best_round": best_round,
            "best_roc_auc": None if best_auc == float("-inf") else best_auc,
            "selected_model_path": selected_model_path,
            "summary_path": str(summary_path),
        }
    )


if __name__ == "__main__":
    main()
