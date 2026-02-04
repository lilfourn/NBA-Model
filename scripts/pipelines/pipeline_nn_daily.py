import argparse
import sys
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modeling.time_utils import central_date_range  # noqa: E402
from scripts.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NBA stats, train NN, and print top picks."
    )
    parser.add_argument("--date-from", default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="YYYY-MM-DD")
    parser.add_argument("--range-days", type=int, default=3)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-top", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    load_env()

    date_from = args.date_from
    date_to = args.date_to
    if not date_from and not date_to:
        start, end = central_date_range(args.range_days)
        date_from = start.isoformat()
        date_to = end.isoformat()

    if not args.skip_fetch:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "fetch_nba_stats.py"),
                "--date-from",
                date_from,
                "--date-to",
                date_to,
            ],
            check=False,
        )

    if not args.skip_train:
        run(
            [sys.executable, str(ROOT / "scripts" / "train_nn_model.py")],
            check=False,
        )

    if not args.skip_top:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_top_picks_nn.py"),
                "--top",
                str(args.top),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
