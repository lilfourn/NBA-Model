import argparse
import sys
from datetime import date
from datetime import timedelta
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modeling.time_utils import central_today, central_yesterday  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build recent backtest dataset, run weekly calibration + report, then score top picks."
    )
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (Central). Defaults to today.")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--gap-days", type=int, default=2)
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--tau-days", type=float, default=60.0)
    ap.add_argument("--min-rows", type=int, default=20000)
    ap.add_argument("--knots", type=int, default=512)
    ap.add_argument("--dataset", default="data/calibration/forecast_backtest_recent.csv")
    ap.add_argument("--calibration", default=None, help="Defaults to dated path under data/calibration/")
    ap.add_argument("--report", default=None, help="Defaults to dated path under data/calibration/reports/")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--include-non-today", action="store_true")
    ap.add_argument(
        "--score",
        default="forecast",
        choices=["forecast", "ensemble"],
        help="Which scorer to run after calibration.",
    )
    ap.add_argument("--rank", default="risk_adj", choices=["risk_adj", "confidence", "edge", "ev"])
    ap.add_argument("--decimal-odds", type=float, default=None)
    ap.add_argument("--break-even-prob", type=float, default=None)
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-calibrate", action="store_true")
    ap.add_argument("--skip-top", action="store_true")
    ap.add_argument("--log-decisions", action="store_true")
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete older calibration artifacts under data/calibration/ after a successful run.",
    )
    ap.add_argument("--keep-calibrations", type=int, default=3)
    ap.add_argument("--keep-reports", type=int, default=3)
    ap.add_argument("--keep-backtests", type=int, default=1)
    args = ap.parse_args()

    load_env()

    asof = args.asof or str(central_today())
    asof_date = central_today() if args.asof is None else date.fromisoformat(asof)

    # Build only what we need for the time-split windows, but cap at completed games (yesterday).
    val_end = asof_date
    date_to = central_yesterday() if val_end == central_today() else val_end
    date_from = val_end - timedelta(days=args.train_days + args.gap_days + args.val_days)

    calibration_path = args.calibration or f"data/calibration/forecast_calibration_{asof}.json"
    report_path = args.report or f"data/calibration/reports/calibration_report_{asof}.csv"

    if not args.skip_dataset:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "calibration" / "build_forecast_backtest_dataset.py"),
                "--output",
                args.dataset,
                "--date-from",
                str(date_from),
                "--date-to",
                str(date_to),
                "--progress-every",
                "50000",
                "--all-supported",
            ],
            check=True,
        )

    if not args.skip_calibrate:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "calibration" / "run_weekly_calibration_report.py"),
                "--dataset",
                args.dataset,
                "--asof",
                asof,
                "--out-calibration",
                calibration_path,
                "--out-report",
                report_path,
                "--train-days",
                str(args.train_days),
                "--gap-days",
                str(args.gap_days),
                "--val-days",
                str(args.val_days),
                "--tau-days",
                str(args.tau_days),
                "--min-rows",
                str(args.min_rows),
                "--knots",
                str(args.knots),
            ],
            check=True,
        )

    if not args.skip_top:
        if args.score == "ensemble":
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "ml" / "run_top_picks_ensemble.py"),
                "--top",
                str(args.top),
                "--calibration",
                calibration_path,
                "--rank",
                args.rank,
            ]
            if args.decimal_odds is not None:
                cmd += ["--decimal-odds", str(args.decimal_odds)]
            if args.break_even_prob is not None:
                cmd += ["--break-even-prob", str(args.break_even_prob)]
            if args.include_non_today:
                cmd.append("--include-non-today")
            if args.log_decisions:
                cmd.append("--log-decisions")
            run(cmd, check=True)
        else:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "ml" / "run_top_picks_forecast.py"),
                "--use-db",
                "--top",
                str(args.top),
                "--calibration",
                calibration_path,
                "--rank",
                args.rank,
            ]
            if args.decimal_odds is not None:
                cmd += ["--decimal-odds", str(args.decimal_odds)]
            if args.break_even_prob is not None:
                cmd += ["--break-even-prob", str(args.break_even_prob)]
            if args.include_non_today:
                cmd.append("--include-non-today")
            if args.log_decisions:
                cmd.append("--log-decisions")
                cmd.append("--log-all")
            run(cmd, check=True)

    if args.cleanup:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "ops" / "cleanup_calibration_artifacts.py"),
                "--dir",
                "data/calibration",
                "--keep-calibrations",
                str(args.keep_calibrations),
                "--keep-reports",
                str(args.keep_reports),
                "--keep-backtests",
                str(args.keep_backtests),
                "--execute",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
