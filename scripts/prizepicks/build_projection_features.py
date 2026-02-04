import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.db.feature_builder import build_projection_features  # noqa: E402


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_latest_snapshot_id(engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("select id from snapshots order by fetched_at desc limit 1")
        ).scalar()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build projection features for a snapshot.")
    parser.add_argument("--snapshot-id", default=None, help="Snapshot ID to build features for")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    snapshot_id = args.snapshot_id or get_latest_snapshot_id(engine)
    if not snapshot_id:
        raise SystemExit("No snapshots found")

    result = build_projection_features(engine, snapshot_id)
    print(result)


if __name__ == "__main__":
    main()
