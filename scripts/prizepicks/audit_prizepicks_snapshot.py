import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.collectors.audit import audit_snapshot  # noqa: E402

DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")
DEFAULT_PREFIX = "prizepicks_nba_"


def find_latest_snapshot(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {directory}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PrizePicks snapshot.")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    snapshot_path = args.path or find_latest_snapshot(args.dir, args.prefix)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    summary = audit_snapshot(payload)

    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
