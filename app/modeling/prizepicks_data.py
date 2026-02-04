from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.modeling.name_utils import normalize_player_name
from app.modeling.types import Projection


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def load_projections(normalized_dir: str | Path = "data/normalized") -> list[Projection]:
    normalized_path = Path(normalized_dir)
    projections_rows = _load_jsonl(normalized_path / "projections.jsonl")
    players_rows = _load_jsonl(normalized_path / "new_players.jsonl")

    players_by_id = {row.get("id"): row for row in players_rows}

    projections: list[Projection] = []
    for row in projections_rows:
        odds_type = row.get("odds_type")
        if isinstance(odds_type, str):
            odds_key = odds_type.strip().lower()
            if odds_key and odds_key != "standard":
                # Exclude special PrizePicks lines (not comparable to standard markets).
                continue
        player_id = row.get("new_player_id") or row.get("player_id")
        if not player_id:
            continue
        player = players_by_id.get(player_id) or {}
        player_name = player.get("display_name") or player.get("name") or row.get("description")
        if not player_name:
            continue
        line_score = _parse_float(row.get("line_score"))
        if line_score is None:
            continue
        projections.append(
            Projection(
                projection_id=str(row.get("id")),
                player_id=str(player_id),
                player_name=str(player_name),
                stat_type=str(row.get("stat_type") or row.get("stat_display_name") or ""),
                line_score=line_score,
                start_time=_parse_datetime(row.get("start_time")),
                game_id=row.get("game_id"),
                event_type=row.get("event_type"),
                projection_type=row.get("projection_type"),
                trending_count=row.get("trending_count"),
                is_today=_parse_bool(row.get("today")),
                is_combo=bool(player.get("combo")),
            )
        )
    return projections


def projection_key(projection: Projection) -> str:
    return f"{normalize_player_name(projection.player_name)}::{projection.stat_type}::{projection.line_score}"
