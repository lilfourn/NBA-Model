import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modeling.time_utils import central_today


def infer_season(today) -> str:
    year = today.year
    if today.month >= 7:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly refresh of player source cache (missing slugs only)."
    )
    parser.add_argument("--season", default=None, help="Season string, e.g. 2025-26")
    parser.add_argument(
        "--normalized-dir",
        default="data/normalized",
        help="Directory containing normalized PrizePicks tables.",
    )
    parser.add_argument(
        "--output",
        default="data/player_sources.jsonl",
        help="Output JSONL path for player sources.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit players.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    args = parser.parse_args()

    season = args.season or infer_season(central_today())

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ops" / "build_player_source_index.py"),
        "--season",
        season,
        "--normalized-dir",
        args.normalized_dir,
        "--output",
        args.output,
        "--missing-only",
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.sleep:
        cmd.extend(["--sleep", str(args.sleep)])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
