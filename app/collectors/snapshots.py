from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_snapshot(
    payload: dict[str, Any],
    *,
    output_dir: str | Path = "data/snapshots",
    prefix: str = "prizepicks_nba",
    pretty: bool = False,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_path / f"{prefix}_{timestamp}.json"

    with snapshot_path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        else:
            json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)

    return snapshot_path
