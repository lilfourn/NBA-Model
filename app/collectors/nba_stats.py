from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _coerce_result_sets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "resultSets" in payload and isinstance(payload["resultSets"], list):
        return payload["resultSets"]
    if "resultSet" in payload and isinstance(payload["resultSet"], dict):
        return [payload["resultSet"]]
    return []


def parse_stats_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result_sets = _coerce_result_sets(payload)
    parsed: list[dict[str, Any]] = []
    for result in result_sets:
        headers = result.get("headers") or []
        rows = result.get("rowSet") or []
        name = result.get("name")
        parsed_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            parsed_rows.append({headers[idx]: row[idx] for idx in range(len(headers))})
        parsed.append({"name": name, "rows": parsed_rows})
    return parsed


def extract_result_rows(payload: dict[str, Any], result_name: str) -> list[dict[str, Any]]:
    for result in parse_stats_payload(payload):
        if result.get("name") == result_name:
            return result.get("rows") or []
    return []


def normalize_player_gamelogs(
    rows: list[dict[str, Any]],
    *,
    season: str | None = None,
    season_type: str | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        game_date = row.get("GAME_DATE")
        game_date_iso = None
        if isinstance(game_date, str):
            try:
                cleaned = game_date.replace("Z", "+00:00")
                game_date_iso = datetime.fromisoformat(cleaned).date().isoformat()
            except ValueError:
                try:
                    game_date_iso = datetime.strptime(game_date, "%b %d, %Y").date().isoformat()
                except ValueError:
                    try:
                        game_date_iso = datetime.strptime(game_date, "%Y-%m-%d").date().isoformat()
                    except ValueError:
                        game_date_iso = None
        normalized.append(
            {
                "player_id": row.get("PLAYER_ID"),
                "player_name": row.get("PLAYER_NAME"),
                "team_id": row.get("TEAM_ID"),
                "team_abbreviation": row.get("TEAM_ABBREVIATION"),
                "game_id": row.get("GAME_ID"),
                "game_date": game_date_iso,
                "matchup": row.get("MATCHUP"),
                "wl": row.get("WL"),
                "season": season,
                "season_type": season_type,
                "stats": row,
            }
        )
    return normalized


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
    return output_path
