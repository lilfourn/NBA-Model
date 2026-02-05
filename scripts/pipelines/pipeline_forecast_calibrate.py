import argparse
import sys
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build backtest dataset, calibrate forecast distribution, then print top picks."
    )
    parser.add_argument("--dataset", default="data/calibration/forecast_backtest.csv")
    parser.add_argument("--calibration", default="data/calibration/forecast_calibration.json")
    parser.add_argument("--min-rows", type=int, default=5000)
    parser.add_argument("--knots", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--stat-types", nargs="*", default=None)
    parser.add_argument("--all-supported", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-calibrate", action="store_true")
    parser.add_argument("--skip-top", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    load_env()

    if not args.skip_dataset:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "calibration" / "build_forecast_backtest_dataset.py"),
            "--output",
            args.dataset,
        ]
        if args.max_rows:
            cmd += ["--max-rows", str(args.max_rows)]
        if args.all_supported:
            cmd.append("--all-supported")
        if args.stat_types:
            cmd += ["--stat-types", *args.stat_types]
        run(cmd, check=False)

    if not args.skip_calibrate:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "calibration" / "calibrate_forecast_distribution.py"),
                "--input",
                args.dataset,
                "--output",
                args.calibration,
                "--min-rows",
                str(args.min_rows),
                "--knots",
                str(args.knots),
            ],
            check=False,
        )

    if not args.skip_top:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "ml" / "run_top_picks_forecast.py"),
                "--use-db",
                "--include-non-today",
                "--top",
                str(args.top),
                "--calibration",
                args.calibration,
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
