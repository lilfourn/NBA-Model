from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from app.modeling.types import PlayerGameLog

STAT_COLUMN_MAP = {
    "FG": "FGM",
    "FGA": "FGA",
    "3P": "FG3M",
    "3PA": "FG3A",
    "FT": "FTM",
    "FTA": "FTA",
    "ORB": "OREB",
    "DRB": "DREB",
    "TRB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
    "PF": "PF",
    "PTS": "PTS",
}


def _parse_game_date(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%b %d, %Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def parse_gamelog_table(html: str) -> list[dict[str, Any]]:
    tables = pd.read_html(html, attrs={"id": "pgl_basic"})
    if not tables:
        return []
    table = tables[0]
    table = table[table["Rk"] != "Rk"]
    return table.to_dict(orient="records")


def normalize_gamelogs(
    rows: list[dict[str, Any]],
    *,
    player_name: str,
    season: str,
    season_type: str,
) -> list[PlayerGameLog]:
    logs: list[PlayerGameLog] = []
    for row in rows:
        date_str = _parse_game_date(row.get("Date"))
        stats: dict[str, Any] = {}
        for col, target in STAT_COLUMN_MAP.items():
            value = row.get(col)
            if value == "Did Not Play" or value == "Inactive":
                value = None
            stats[target] = value
        stats["GAME_DATE"] = row.get("Date")
        logs.append(
            PlayerGameLog(
                player_id=None,
                player_name=player_name,
                game_date=(datetime.fromisoformat(date_str).date() if date_str else None),
                stats=stats,
            )
        )
    return logs
