import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.db.prizepicks_loader import load_snapshot  # noqa: E402

DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")
DEFAULT_PREFIX = "prizepicks_nba_"


def find_latest_snapshot(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {directory}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load PrizePicks snapshot into Postgres.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to a specific snapshot JSON file",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Directory where snapshots are stored",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Snapshot filename prefix to search for",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL for this run",
    )
    parser.add_argument(
        "--league-id",
        default=None,
        help="Override league id recorded for snapshot",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=None,
        help="Override per_page recorded for snapshot",
    )
    args = parser.parse_args()

    snapshot_path = args.path or find_latest_snapshot(args.dir, args.prefix)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    engine = get_engine(args.database_url)
    counts = load_snapshot(
        payload,
        engine=engine,
        snapshot_path=str(snapshot_path),
        league_id=args.league_id,
        per_page=args.per_page,
    )

    print(f"Loaded snapshot: {snapshot_path}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
