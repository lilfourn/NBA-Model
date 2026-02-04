from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from app.modeling.name_utils import normalize_player_name
from app.modeling.types import PlayerGameLog


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def load_game_logs(paths: Iterable[str | Path]) -> list[PlayerGameLog]:
    logs: list[PlayerGameLog] = []
    for path in paths:
        for row in _load_jsonl(Path(path)):
            logs.append(
                PlayerGameLog(
                    player_id=(str(row.get("player_id")) if row.get("player_id") else None),
                    player_name=row.get("player_name"),
                    game_date=_parse_date(row.get("game_date")),
                    stats=row.get("stats") or {},
                )
            )
    return logs


def discover_game_log_files(official_dir: str | Path = "data/official") -> list[Path]:
    root = Path(official_dir)
    if not root.exists():
        return []
    return sorted(root.glob("nba_player_gamelogs_*.jsonl"))


def index_game_logs_by_player(logs: Iterable[PlayerGameLog]) -> dict[str, list[PlayerGameLog]]:
    indexed: dict[str, list[PlayerGameLog]] = {}
    for log in logs:
        key = normalize_player_name(log.player_name)
        if not key:
            continue
        indexed.setdefault(key, []).append(log)
    for key, values in indexed.items():
        indexed[key] = sorted(
            values,
            key=lambda entry: entry.game_date or date.min,
        )
    return indexed


def merge_game_logs(
    official: Iterable[PlayerGameLog],
    fallback: Iterable[PlayerGameLog],
) -> list[PlayerGameLog]:
    merged: dict[tuple[str, date | None], PlayerGameLog] = {}
    for log in official:
        name_key = normalize_player_name(log.player_name)
        if not name_key:
            continue
        key = (name_key, log.game_date)
        merged[key] = log
    for log in fallback:
        name_key = normalize_player_name(log.player_name)
        if not name_key:
            continue
        key = (name_key, log.game_date)
        if key in merged:
            continue
        merged[key] = log
    return list(merged.values())
