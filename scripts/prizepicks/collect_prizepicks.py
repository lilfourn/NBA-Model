import argparse
from uuid import UUID
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.logging import log_run_summary, log_validation, set_log_path  # noqa: E402
from app.clients.prizepicks import fetch_projections  # noqa: E402
from app.collectors.snapshots import write_snapshot  # noqa: E402
from app.collectors.validators import validate_prizepicks_response  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.db.feature_builder import build_projection_features  # noqa: E402
from app.db.prizepicks_loader import load_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch, snapshot, and load PrizePicks NBA data.")
    parser.add_argument("--per-page", type=int, default=None, help="Override per_page for API call")
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
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL for this run",
    )
    args = parser.parse_args()

    import time as _time
    _start = _time.monotonic()

    set_log_path("logs/collection.jsonl")

    payload = fetch_projections(per_page=args.per_page)

    vr = validate_prizepicks_response(payload)
    log_validation("prizepicks", valid=vr.valid, errors=vr.errors, warnings=vr.warnings)
    if not vr.valid:
        print(f"Validation failed: {vr.errors}")
        sys.exit(1)
    if vr.warnings:
        print(f"Validation warnings: {vr.warnings}")

    snapshot_path = write_snapshot(payload, output_dir=args.output_dir, pretty=args.pretty)

    engine = get_engine(args.database_url)
    counts = load_snapshot(payload, engine=engine, snapshot_path=str(snapshot_path))
    if counts.get("snapshots", 0) > 0 and counts.get("snapshot_id"):
        build_projection_features(engine, snapshot_id=UUID(counts["snapshot_id"]))

    _elapsed = _time.monotonic() - _start
    log_run_summary("prizepicks", duration_seconds=_elapsed, counts=counts)

    print(f"Saved snapshot: {snapshot_path}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
