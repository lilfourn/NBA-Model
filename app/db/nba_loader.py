from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from app.db import schema
from app.utils.names import normalize_name

MAX_QUERY_PARAMS = 60000


def _batch_size_for(table) -> int:
    column_count = max(1, len(table.c))
    return max(1, min(1000, MAX_QUERY_PARAMS // column_count))


def _chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _parse_date(value: Any) -> datetime.date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _merge_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if value is not None:
            existing[key] = value
    return existing


def _dedupe_rows_by_conflict(
    rows: list[dict[str, Any]], conflict_cols: list[str]
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(col) for col in conflict_cols)
        existing = deduped.get(key)
        if existing is not None:
            deduped[key] = _merge_row(existing, row)
        else:
            deduped[key] = row
    return list(deduped.values())


def _upsert_rows(conn: Connection, table, rows: list[dict[str, Any]], conflict_cols: list[str]) -> int:
    if not rows:
        return 0
    rows = _dedupe_rows_by_conflict(rows, conflict_cols)
    if not rows:
        return 0
    size = _batch_size_for(table)
    total = 0
    for batch in _chunk_rows(rows, size):
        stmt = pg_insert(table).values(batch)
        update_cols = {}
        for col in table.c:
            if col.name in conflict_cols:
                continue
            update_cols[col.name] = func.coalesce(stmt.excluded[col.name], col)
        stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
        result = conn.execute(stmt)
        batch_count = result.rowcount
        if batch_count is None or batch_count < 0:
            batch_count = len(batch)
        total += batch_count
    return total


def load_league_game_logs(rows: list[dict[str, Any]], *, engine: Engine) -> dict[str, int]:
    players_by_id: dict[str, dict[str, Any]] = {}
    games_by_id: dict[str, dict[str, Any]] = {}
    stats_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        player_id = str(row.get("PLAYER_ID")) if row.get("PLAYER_ID") is not None else None
        player_name = row.get("PLAYER_NAME")
        team_id = str(row.get("TEAM_ID")) if row.get("TEAM_ID") is not None else None
        team_abbrev = row.get("TEAM_ABBREVIATION")
        game_id = str(row.get("GAME_ID")) if row.get("GAME_ID") is not None else None
        matchup = row.get("MATCHUP") or ""

        home_abbrev = None
        away_abbrev = None
        if " vs. " in matchup:
            parts = matchup.split(" vs. ")
            if len(parts) == 2:
                home_abbrev = parts[0].strip()
                away_abbrev = parts[1].strip()
        elif " @ " in matchup:
            parts = matchup.split(" @ ")
            if len(parts) == 2:
                away_abbrev = parts[0].strip()
                home_abbrev = parts[1].strip()

        if player_id:
            incoming_player = {
                "id": player_id,
                "full_name": player_name,
                "name_key": normalize_name(player_name),
                "team_id": team_id,
                "team_abbreviation": team_abbrev,
            }
            existing_player = players_by_id.get(player_id)
            if existing_player:
                players_by_id[player_id] = _merge_row(existing_player, incoming_player)
            else:
                players_by_id[player_id] = incoming_player

        if game_id:
            incoming_game = {
                "id": game_id,
                "game_date": _parse_date(row.get("GAME_DATE")),
                "status_text": row.get("WL"),
                "home_team_id": None,
                "away_team_id": None,
                "home_team_abbreviation": home_abbrev,
                "away_team_abbreviation": away_abbrev,
            }
            existing_game = games_by_id.get(game_id)
            if existing_game:
                games_by_id[game_id] = _merge_row(existing_game, incoming_game)
            else:
                games_by_id[game_id] = incoming_game

        if game_id and player_id:
            stats_key = (game_id, player_id)
            incoming_stats = {
                "game_id": game_id,
                "player_id": player_id,
                "team_id": team_id,
                "team_abbreviation": team_abbrev,
                "minutes": _parse_decimal(row.get("MIN")),
                "points": _parse_int(row.get("PTS")),
                "rebounds": _parse_int(row.get("REB")),
                "assists": _parse_int(row.get("AST")),
                "steals": _parse_int(row.get("STL")),
                "blocks": _parse_int(row.get("BLK")),
                "turnovers": _parse_int(row.get("TOV")),
                "fg3m": _parse_int(row.get("FG3M")),
                "fg3a": _parse_int(row.get("FG3A")),
                "fg3_pct": _parse_decimal(row.get("FG3_PCT")),
                "fgm": _parse_int(row.get("FGM")),
                "fga": _parse_int(row.get("FGA")),
                "fg_pct": _parse_decimal(row.get("FG_PCT")),
                "ftm": _parse_int(row.get("FTM")),
                "fta": _parse_int(row.get("FTA")),
                "ft_pct": _parse_decimal(row.get("FT_PCT")),
                "plus_minus": _parse_decimal(row.get("PLUS_MINUS")),
                "stats_json": row,
            }
            existing_stats = stats_by_key.get(stats_key)
            if existing_stats:
                stats_by_key[stats_key] = _merge_row(existing_stats, incoming_stats)
            else:
                stats_by_key[stats_key] = incoming_stats

    with engine.begin() as conn:
        players = list(players_by_id.values())
        games = list(games_by_id.values())
        stats = list(stats_by_key.values())
        counts = {
            "nba_players": _upsert_rows(conn, schema.nba_players, players, ["id"]),
            "nba_games": _upsert_rows(conn, schema.nba_games, games, ["id"]),
            "nba_player_game_stats": _upsert_rows(
                conn, schema.nba_player_game_stats, stats, ["game_id", "player_id"]
            ),
        }

    return counts
