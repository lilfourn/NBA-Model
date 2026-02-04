import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.collectors.inspector import summarize_snapshot  # noqa: E402

DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")
DEFAULT_PREFIX = "prizepicks_nba_"


def find_latest_snapshot(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {directory}")
    return candidates[-1]


def print_summary(summary: dict[str, Any], *, label: str, limit: int | None) -> None:
    if limit:
        print(f"{label} (limited to first {limit} items)")
    else:
        print(label)

    for section in ("data", "included"):
        section_summary = summary[section]
        print(f"{section.upper()}: total={section_summary['total']}")
        print(f"{section.upper()}: types={section_summary['types']}")
        for item_type, details in section_summary["by_type"].items():
            print(f"TYPE={item_type} count={details['count']} samples={details['sample_ids']}")
            print(
                f"TYPE={item_type} attribute_keys({len(details['attribute_keys'])})="
                f"{details['attribute_keys']}"
            )
            print(
                f"TYPE={item_type} relationship_keys({len(details['relationship_keys'])})="
                f"{details['relationship_keys']}"
            )
            if details["relationship_data_shapes"]:
                print(f"TYPE={item_type} relationship_data_shapes={details['relationship_data_shapes']}")
            print("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PrizePicks snapshot structure.")
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
        "--limit",
        type=int,
        default=None,
        help="Limit the number of items inspected per section",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print summary JSON when using --output",
    )
    args = parser.parse_args()

    snapshot_path = args.path or find_latest_snapshot(args.dir, args.prefix)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    summary = summarize_snapshot(payload, limit=args.limit)

    print(f"Snapshot: {snapshot_path}")
    print(f"Top-level keys: {summary['top_level_keys']}")
    print(f"Links keys: {summary['links_keys']}")
    print(f"Meta keys: {summary['meta_keys']}")
    print("-")

    print_summary(summary, label="SECTION SUMMARY", limit=args.limit)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            if args.pretty:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            else:
                json.dump(summary, handle, separators=(",", ":"), ensure_ascii=False)
        print(f"Summary written to: {args.output}")


if __name__ == "__main__":
    main()
