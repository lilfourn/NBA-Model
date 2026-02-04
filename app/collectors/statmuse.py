from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from app.modeling.types import PlayerGameLog

STAT_COLUMN_MAP = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "FGM": "FGM",
    "FGA": "FGA",
    "3PM": "FG3M",
    "3PA": "FG3A",
    "FTM": "FTM",
    "FTA": "FTA",
    "OREB": "OREB",
    "DREB": "DREB",
    "TOV": "TOV",
    "PF": "PF",
}


def _parse_game_date(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def parse_first_stats_table(html: str) -> list[dict[str, Any]]:
    tables = pd.read_html(html)
    if not tables:
        return []
    # StatMuse tables often include a leading empty column.
    return tables[0].to_dict(orient="records")


def normalize_gamelogs(
    rows: list[dict[str, Any]],
    *,
    player_name: str,
) -> list[PlayerGameLog]:
    logs: list[PlayerGameLog] = []
    for row in rows:
        date_str = _parse_game_date(row.get("DATE") or row.get("Date"))
        stats: dict[str, Any] = {}
        for col, target in STAT_COLUMN_MAP.items():
            value = row.get(col)
            stats[target] = value
        stats["GAME_DATE"] = row.get("DATE") or row.get("Date")
        logs.append(
            PlayerGameLog(
                player_id=None,
                player_name=player_name,
                game_date=(datetime.fromisoformat(date_str).date() if date_str else None),
                stats=stats,
            )
        )
    return logs
