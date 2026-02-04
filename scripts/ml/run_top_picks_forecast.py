import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.modeling.db_logs import load_db_game_logs  # noqa: E402
from app.modeling.game_logs import discover_game_log_files, load_game_logs, merge_game_logs  # noqa: E402
from app.modeling.prizepicks_data import load_projections  # noqa: E402
from app.modeling.forecast_calibration import ForecastDistributionCalibrator  # noqa: E402
from app.modeling.stat_forecast import ForecastParams, LeaguePriors, StatForecastPredictor  # noqa: E402
from scripts.log_decisions import PRED_LOG_DEFAULT, append_prediction_log  # noqa: E402
from scripts.train_baseline_model import load_env  # noqa: E402


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_over_raw(*, mu: float, sigma: float, line: float, continuity: float = 0.5) -> float:
    if sigma <= 0:
        return 0.5
    z = ((line + continuity) - mu) / sigma
    return 1.0 - _normal_cdf(z)

def _risk_adjusted_confidence(*, p_over: float, n_eff: float | None, status: str) -> float:
    p_pick = max(p_over, 1.0 - p_over)
    rho = 0.85
    if n_eff is None or n_eff < 5:
        rho = 0.65
    if status == "raw":
        rho = min(rho, 0.60)
    elif status == "fallback_global":
        rho = min(rho, 0.75)
    return 0.5 + rho * (p_pick - 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print top picks from stat-forecast model.")
    parser.add_argument("--normalized-dir", default="data/normalized")
    parser.add_argument("--official-dir", default="data/official")
    parser.add_argument("--fallback-dir", default="data/fallback")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--include-non-today", action="store_true")
    parser.add_argument("--tau-short", type=float, default=7.0)
    parser.add_argument("--tau-long", type=float, default=21.0)
    parser.add_argument("--use-db", action="store_true", help="Load game logs from DB")
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional forecast calibration JSON. If set, uses calibrated P(over).",
    )
    parser.add_argument(
        "--rank",
        default="risk_adj",
        choices=["risk_adj", "confidence", "edge", "ev"],
        help="Ranking strategy for Top Picks.",
    )
    parser.add_argument(
        "--decimal-odds",
        type=float,
        default=None,
        help="Optional decimal odds used for --rank edge/ev (slip-level if applicable).",
    )
    parser.add_argument(
        "--break-even-prob",
        type=float,
        default=None,
        help="Optional break-even probability used for --rank edge (overrides --decimal-odds).",
    )
    parser.add_argument(
        "--exclude-raw",
        action="store_true",
        help="Exclude rows that could not be calibrated (neither per-stat nor global).",
    )
    parser.add_argument("--log-decisions", action="store_true", help="Append predictions to CSV log.")
    parser.add_argument("--log-path", default=PRED_LOG_DEFAULT, help="Path to prediction log CSV.")
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Log all scored projections (default: log only displayed top picks).",
    )
    args = parser.parse_args()
    if args.rank in {"edge", "ev"} and args.decimal_odds is None and args.break_even_prob is None:
        raise SystemExit("--rank edge/ev requires --decimal-odds or --break-even-prob.")

    load_env()
    projections = load_projections(args.normalized_dir)
    if not args.include_non_today:
        projections = [proj for proj in projections if proj.is_today]

    projections = [proj for proj in projections if not proj.is_combo]
    if not projections:
        print("No projections available after filters.")
        return

    if args.use_db:
        engine = get_engine(args.database_url)
        game_logs = load_db_game_logs(engine)
    else:
        game_log_files = discover_game_log_files(args.official_dir)
        if not game_log_files:
            print("No official game logs found. Run scripts/fetch_nba_player_gamelogs.py first.")
            return
        game_logs = load_game_logs(game_log_files)
        fallback_files = discover_game_log_files(args.fallback_dir)
        if fallback_files:
            fallback_logs = load_game_logs(fallback_files)
            game_logs = merge_game_logs(game_logs, fallback_logs)

    calibrator = None
    if args.calibration:
        calibrator = ForecastDistributionCalibrator.load(args.calibration)

    params = ForecastParams(tau_short=args.tau_short, tau_long=args.tau_long)
    needed_stat_types = sorted({proj.stat_type for proj in projections})
    priors = LeaguePriors(game_logs, stat_types=needed_stat_types, minutes_prior=params.minutes_prior)
    predictor = StatForecastPredictor(
        game_logs,
        min_games=args.min_games,
        params=params,
        league_priors=priors,
        calibrator=calibrator,
    )

    predictions = []
    for projection in projections:
        pred = predictor.predict(projection)
        if pred is None:
            continue
        predictions.append(pred)

    if not predictions:
        print("No predictions generated. Check data coverage and filters.")
        return

    def _rank_score(pred) -> float:
        details = pred.details or {}
        status = str(details.get("calibration_status") or "raw")
        n_eff = details.get("n_eff")
        if args.rank == "confidence":
            return float(pred.confidence)
        p_adj = float(_risk_adjusted_confidence(p_over=float(pred.prob_over), n_eff=n_eff, status=status))
        if args.rank == "risk_adj":
            return p_adj
        p_be = float(args.break_even_prob) if args.break_even_prob is not None else (1.0 / float(args.decimal_odds))
        if args.rank == "edge":
            return p_adj - p_be
        if args.rank == "ev":
            decimal_odds = float(args.decimal_odds) if args.decimal_odds is not None else (1.0 / p_be)
            return (p_adj * decimal_odds) - 1.0
        return p_adj

    if args.exclude_raw:
        predictions = [
            pred
            for pred in predictions
            if str((pred.details or {}).get("calibration_status") or "raw") != "raw"
        ]
        if not predictions:
            print("All predictions were raw/uncalibrated after filtering. Check calibration artifact.")
            return

    predictions.sort(key=_rank_score, reverse=True)
    top_predictions = predictions[: args.top]

    # Summary
    status_counts: dict[str, int] = {"per_stat": 0, "fallback_global": 0, "raw": 0}
    raw_stat_types: set[str] = set()
    global_stat_types: set[str] = set()
    for pred in predictions:
        status = str((pred.details or {}).get("calibration_status") or "raw")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "raw":
            raw_stat_types.add(pred.projection.stat_type)
        elif status == "fallback_global":
            global_stat_types.add(pred.projection.stat_type)
    skipped = len(projections) - len(predictions)
    print(
        f"Scored {len(predictions)}/{len(projections)} projections (skipped {skipped}). "
        f"Calibration: per_stat={status_counts.get('per_stat', 0)}, "
        f"fallback_global={status_counts.get('fallback_global', 0)}, raw={status_counts.get('raw', 0)}."
    )
    if global_stat_types:
        listed = ", ".join(sorted(global_stat_types))
        print(f"Note: used global calibration for: {listed}")
    if raw_stat_types:
        listed = ", ".join(sorted(raw_stat_types))
        print(f"Warning: missing calibration for: {listed}")

    print(f"Top {len(top_predictions)} Forecast Picks")
    print("=" * 80)
    for rank, pred in enumerate(top_predictions, start=1):
        proj = pred.projection
        details = pred.details or {}
        status = str(details.get("calibration_status") or "raw")
        line = f"{proj.player_name} | {proj.stat_type} | line {proj.line_score:.2f}"
        score = _rank_score(pred)
        pick = (
            f"{pred.pick} ({pred.confidence:.2%} conf, P_over={pred.prob_over:.2%}, "
            f"rank={score:.3f}, cal={status})"
        )
        print(f"{rank:>2}. {line} -> {pick}")

    if args.log_decisions:
        decision_time = datetime.now(timezone.utc).isoformat()
        calibration_version = Path(args.calibration).name if args.calibration else None
        to_log = predictions if args.log_all else top_predictions

        rows = []
        for pred in to_log:
            proj = pred.projection
            details = pred.details or {}
            status = str(details.get("calibration_status") or "raw")
            score = _rank_score(pred)
            mu_hat = float(details.get("raw_mean", pred.mean or 0.0))
            sigma_hat = float(details.get("raw_std", pred.std or 0.0))
            p_over_raw = _p_over_raw(mu=mu_hat, sigma=sigma_hat, line=float(proj.line_score))
            rows.append(
                {
                    "projection_id": proj.projection_id,
                    "game_id": proj.game_id,
                    "player_id": proj.player_id,
                    "stat_type": proj.stat_type,
                    "decision_time": decision_time,
                    "line_score": float(proj.line_score),
                    "mu_hat": mu_hat,
                    "sigma_hat": sigma_hat,
                    "p_over_raw": float(p_over_raw),
                    "p_over_cal": float(pred.prob_over),
                    "model_version": pred.model_version,
                    "calibration_version": calibration_version,
                    "calibration_status": status,
                    "n_eff": details.get("n_eff"),
                    "rank_score": float(score),
                }
            )

        append_prediction_log(pd.DataFrame(rows), path=args.log_path)
        print(f"\nLogged {len(rows)} predictions -> {args.log_path}")


if __name__ == "__main__":
    main()
