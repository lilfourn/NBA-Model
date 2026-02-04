import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.ml.nn.infer import format_predictions, infer_over_probs  # noqa: E402
from app.modeling.baseline import BaselinePredictor  # noqa: E402
from app.modeling.game_logs import discover_game_log_files, load_game_logs, merge_game_logs  # noqa: E402
from app.modeling.types import Projection  # noqa: E402
from scripts.train_baseline_model import load_env  # noqa: E402


def _latest_model_path(models_dir: Path) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob("nn_gru_attention_*.pt"))
    return candidates[-1] if candidates else None


def _latest_snapshot_id(engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("select id from snapshots order by fetched_at desc limit 1")
        ).scalar()


def _load_baseline_predictor(official_dir: str, fallback_dir: str, min_games: int):
    game_log_files = discover_game_log_files(official_dir)
    game_logs = load_game_logs(game_log_files) if game_log_files else []
    fallback_files = discover_game_log_files(fallback_dir)
    if fallback_files:
        fallback_logs = load_game_logs(fallback_files)
        game_logs = merge_game_logs(game_logs, fallback_logs)
    return BaselinePredictor(game_logs, min_games=min_games)


def _to_projection(row) -> Projection:
    return Projection(
        projection_id=str(row.get("projection_id")),
        player_id=str(row.get("player_id")),
        player_name=str(row.get("player_name") or row.get("prizepicks_name_key") or ""),
        stat_type=str(row.get("stat_type") or ""),
        line_score=float(row.get("line_score") or 0.0),
        start_time=row.get("start_time"),
        game_id=row.get("game_id"),
        event_type=None,
        projection_type=row.get("projection_type"),
        trending_count=row.get("trending_count"),
        is_today=row.get("today"),
        is_combo=bool(row.get("combo") or False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print top picks from NN model.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--official-dir", default="data/official")
    parser.add_argument("--fallback-dir", default="data/fallback")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    snapshot_id = args.snapshot_id or _latest_snapshot_id(engine)
    if not snapshot_id:
        print("No snapshots found.")
        return

    models_dir = Path(args.models_dir)
    model_path = _latest_model_path(models_dir)
    if not model_path:
        print("No NN model found. Train first with scripts/train_nn_model.py")
        return

    inference = infer_over_probs(
        engine=engine,
        model_path=str(model_path),
        snapshot_id=str(snapshot_id),
    )
    if inference.frame.empty:
        print("No projections available for this snapshot.")
        return

    baseline = _load_baseline_predictor(args.official_dir, args.fallback_dir, 1)

    predictions = []
    for idx, pred in enumerate(format_predictions(inference.frame, inference.probs)):
        hist_n = float(inference.numeric.iloc[idx].get("hist_n", 0.0))
        if hist_n < args.min_games:
            proj = _to_projection(inference.frame.iloc[idx])
            baseline_pred = baseline.predict(proj)
            if baseline_pred is None:
                continue
            predictions.append(
                {
                    "player_name": proj.player_name,
                    "stat_type": proj.stat_type,
                    "line_score": proj.line_score,
                    "pick": baseline_pred.pick,
                    "prob_over": baseline_pred.prob_over,
                    "confidence": baseline_pred.confidence,
                    "source": "baseline",
                }
            )
        else:
            predictions.append({**pred, "source": "nn"})

    if not predictions:
        print("No predictions generated. Check data coverage and filters.")
        return

    predictions.sort(key=lambda item: item["confidence"], reverse=True)
    top_predictions = predictions[: args.top]

    print(f"Top {len(top_predictions)} NN Picks")
    print("=" * 80)
    for rank, pred in enumerate(top_predictions, start=1):
        line = (
            f"{pred['player_name']} | {pred['stat_type']} | line {pred['line_score']:.2f}"
        )
        pick = (
            f"{pred['pick']} ({pred['confidence']:.2%} conf, "
            f"P_over={pred['prob_over']:.2%}, {pred['source']})"
        )
        print(f"{rank:>2}. {line} -> {pick}")


if __name__ == "__main__":
    main()
