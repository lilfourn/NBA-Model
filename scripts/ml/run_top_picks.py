import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modeling.baseline import BaselinePredictor
from app.modeling.game_logs import discover_game_log_files, load_game_logs, merge_game_logs
from app.modeling.prizepicks_data import load_projections


def main() -> None:
    parser = argparse.ArgumentParser(description="Print top PrizePicks over/under picks.")
    parser.add_argument(
        "--normalized-dir",
        default="data/normalized",
        help="Directory containing normalized PrizePicks tables.",
    )
    parser.add_argument(
        "--official-dir",
        default="data/official",
        help="Directory containing official NBA game logs.",
    )
    parser.add_argument(
        "--fallback-dir",
        default="data/fallback",
        help="Directory containing fallback NBA game logs.",
    )
    parser.add_argument(
        "--include-fallback",
        action="store_true",
        help="Merge fallback logs when official logs are missing.",
    )
    parser.add_argument("--min-games", type=int, default=5, help="Minimum games for a player.")
    parser.add_argument("--top", type=int, default=25, help="Number of picks to show.")
    parser.add_argument(
        "--include-non-today",
        action="store_true",
        help="Include projections not marked as today.",
    )
    args = parser.parse_args()

    projections = load_projections(args.normalized_dir)

    if not args.include_non_today:
        projections = [proj for proj in projections if proj.is_today]

    projections = [
        proj
        for proj in projections
        if not proj.is_combo and str(proj.event_type or "").lower() != "combo"
    ]
    if not projections:
        if args.include_non_today:
            print("No projections available after filters.")
        else:
            print(
                "No projections marked as today. "
                "Try --include-non-today if upcoming games are acceptable."
            )
        return

    game_log_files = discover_game_log_files(args.official_dir)
    if not game_log_files:
        print("No official game logs found. Run scripts/nba/fetch_nba_player_gamelogs.py first.")
        return

    game_logs = load_game_logs(game_log_files)
    if args.include_fallback:
        fallback_files = discover_game_log_files(args.fallback_dir)
        if fallback_files:
            fallback_logs = load_game_logs(fallback_files)
            game_logs = merge_game_logs(game_logs, fallback_logs)
    predictor = BaselinePredictor(game_logs, min_games=args.min_games)

    predictions = []
    for projection in projections:
        prediction = predictor.predict(projection)
        if prediction is None:
            continue
        predictions.append(prediction)

    if not predictions:
        print("No predictions generated. Check data coverage and filters.")
        return

    predictions.sort(key=lambda item: item.confidence, reverse=True)
    top_predictions = predictions[: args.top]

    print(f"Top {len(top_predictions)} Picks")
    print("=" * 80)
    for rank, pred in enumerate(top_predictions, start=1):
        proj = pred.projection
        line = f"{proj.player_name} | {proj.stat_type} | line {proj.line_score:.2f}"
        pick = f"{pred.pick} ({pred.confidence:.2%} conf, P_over={pred.prob_over:.2%})"
        print(f"{rank:>2}. {line} -> {pick}")


if __name__ == "__main__":
    main()
