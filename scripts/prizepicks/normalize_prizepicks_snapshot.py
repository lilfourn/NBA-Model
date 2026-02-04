import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.collectors.normalizer import normalize_snapshot  # noqa: E402

DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")
DEFAULT_PREFIX = "prizepicks_nba_"
DEFAULT_OUTPUT_DIR = Path("data/normalized")


def find_latest_snapshot(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {directory}")
    return candidates[-1]


def _stringify_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return value


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify_for_csv(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize PrizePicks snapshot into tables.")
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store normalized tables",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format for tables",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Write snapshot meta + links to metadata.json",
    )
    parser.add_argument(
        "--all-odds-types",
        action="store_true",
        help="Include non-standard odds types (demon/goblin). Default filters to standard-only.",
    )
    args = parser.parse_args()

    snapshot_path = args.path or find_latest_snapshot(args.dir, args.prefix)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tables = normalize_snapshot(payload)

    if not args.all_odds_types and "projections" in tables:
        tables["projections"] = [
            row
            for row in tables["projections"]
            if str(row.get("odds_type") or "standard").strip().lower() == "standard"
        ]

    for table_name, rows in tables.items():
        output_path = args.output_dir / f"{table_name}.{args.format}"
        if args.format == "jsonl":
            write_jsonl(rows, output_path)
        else:
            write_csv(rows, output_path)
        print(f"Wrote {len(rows)} rows -> {output_path}")

    if args.include_meta:
        metadata_path = args.output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "links": payload.get("links") or {},
                    "meta": payload.get("meta") or {},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
