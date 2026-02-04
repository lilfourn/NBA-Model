import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.prizepicks import fetch_projections
from app.collectors.snapshots import write_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA projections and save a snapshot.")
    parser.add_argument("--per-page", type=int, default=None, help="Override per_page for the API call")
    parser.add_argument(
        "--output-dir",
        default="data/snapshots",
        help="Directory to store snapshot files",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write JSON with indentation for readability",
    )
    args = parser.parse_args()

    payload = fetch_projections(per_page=args.per_page)
    snapshot_path = write_snapshot(payload, output_dir=args.output_dir, pretty=args.pretty)

    data_count = len(payload.get("data", []))
    included_count = len(payload.get("included", []))
    top_keys = list(payload.keys())

    print(f"Saved snapshot: {snapshot_path}")
    print(f"Top-level keys: {top_keys}")
    print(f"Counts -> data: {data_count}, included: {included_count}")


if __name__ == "__main__":
    main()
