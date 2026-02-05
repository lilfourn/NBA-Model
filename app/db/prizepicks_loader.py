from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable
from uuid import UUID, uuid4, uuid5

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from app.clients.prizepicks import build_projections_url
from app.collectors.audit import audit_snapshot
from app.core.config import settings
from app.db import schema
from app.db.coerce import (
    parse_bool as _parse_bool,
    parse_datetime as _parse_datetime,
    parse_decimal as _parse_decimal,
    parse_int as _parse_int,
    to_str as _to_str,
)
from app.utils.names import normalize_name

TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{6})Z")
MAX_QUERY_PARAMS = 60000
SNAPSHOT_NAMESPACE = UUID("b1d0a6e3-2f1c-4a1b-9c35-9f9f6f0c7c8e")




ODDS_TYPE_CODES: dict[str, int] = {
    "standard": 0,
    "goblin": 1,
    "demon": 2,
}
ALLOWED_ODDS_TYPES = {"standard"}
EXCLUDED_ODDS_TYPES = {"goblin", "demon"}


def _parse_odds_type(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return ODDS_TYPE_CODES.get(value.strip().lower())
    return _parse_int(value)




def _relationship_id(relationships: dict[str, Any], key: str) -> str | None:
    rel = relationships.get(key)
    if not isinstance(rel, dict):
        return None
    data = rel.get("data")
    if isinstance(data, dict):
        return _to_str(data.get("id"))
    return None


def _parse_snapshot_timestamp(snapshot_path: str | None) -> datetime:
    if snapshot_path:
        match = TIMESTAMP_PATTERN.search(snapshot_path)
        if match:
            value = match.group(1)
            return datetime.strptime(value, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _snapshot_uuid(snapshot_path: str | None) -> UUID:
    if snapshot_path:
        return uuid5(SNAPSHOT_NAMESPACE, snapshot_path)
    return uuid4()


def _line_movement(current: Decimal | None, previous: Decimal | None) -> str | None:
    if current is None:
        return None
    if previous is None:
        return "new"
    delta = current - previous
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "same"


def _batch_size_for(table) -> int:
    column_count = max(1, len(table.c))
    return max(1, min(1000, MAX_QUERY_PARAMS // column_count))


def _chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _snapshot_row(
    *,
    snapshot_id: UUID,
    payload: dict[str, Any],
    snapshot_path: str | None,
    league_id: str,
    per_page: int | None,
) -> dict[str, Any]:
    fetched_at = _parse_snapshot_timestamp(snapshot_path)
    return {
        "id": snapshot_id,
        "fetched_at": fetched_at,
        "league_id": league_id,
        "per_page": per_page,
        "source_url": build_projections_url(),
        "snapshot_path": snapshot_path,
        "data_count": len(payload.get("data") or []),
        "included_count": len(payload.get("included") or []),
        "links": payload.get("links") or {},
        "meta": payload.get("meta") or {},
    }


def _projection_rows(items: Iterable[dict[str, Any]], snapshot_id: UUID) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        relationships = item.get("relationships") or {}
        odds_type_raw = attributes.get("odds_type")
        odds_type_code = _parse_odds_type(odds_type_raw)
        odds_type_key: str | None
        if isinstance(odds_type_raw, str):
            odds_type_key = odds_type_raw.strip().lower() or "standard"
        elif odds_type_code is not None:
            odds_type_key = next((key for key, value in ODDS_TYPE_CODES.items() if value == odds_type_code), None)
        else:
            # PrizePicks omits odds_type for standard lines sometimes; treat missing as standard.
            odds_type_key = "standard"

        if odds_type_key in EXCLUDED_ODDS_TYPES:
            continue
        if odds_type_key not in ALLOWED_ODDS_TYPES:
            continue
        if odds_type_code is None and odds_type_key == "standard":
            odds_type_code = ODDS_TYPE_CODES["standard"]
        line_score = _parse_decimal(attributes.get("line_score"))
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "projection_id": _to_str(item.get("id")),
                "league_id": _relationship_id(relationships, "league"),
                "player_id": _relationship_id(relationships, "new_player"),
                "stat_type_id": _relationship_id(relationships, "stat_type"),
                "projection_type_id": _relationship_id(relationships, "projection_type"),
                "game_id": _relationship_id(relationships, "game"),
                "duration_id": _relationship_id(relationships, "duration"),
                "line_score": line_score,
                "line_score_prev": None,
                "line_score_delta": None,
                "line_movement": None,
                "adjusted_odds": _parse_decimal(attributes.get("adjusted_odds")),
                "discount_percentage": _parse_decimal(attributes.get("discount_percentage")),
                "flash_sale_line_score": _parse_decimal(attributes.get("flash_sale_line_score")),
                "odds_type": odds_type_code,
                "rank": _parse_int(attributes.get("rank")),
                "trending_count": _parse_int(attributes.get("trending_count")),
                "status": attributes.get("status"),
                "stat_type": attributes.get("stat_type"),
                "stat_display_name": attributes.get("stat_display_name"),
                "projection_type": attributes.get("projection_type"),
                "description": attributes.get("description"),
                "event_type": attributes.get("event_type"),
                "group_key": attributes.get("group_key"),
                "tv_channel": attributes.get("tv_channel"),
                "start_time": _parse_datetime(attributes.get("start_time")),
                "board_time": _parse_datetime(attributes.get("board_time")),
                "end_time": _parse_datetime(attributes.get("end_time")),
                "updated_at": _parse_datetime(attributes.get("updated_at")),
                "is_promo": _parse_bool(attributes.get("is_promo")),
                "is_live": _parse_bool(attributes.get("is_live")),
                "is_live_scored": _parse_bool(attributes.get("is_live_scored")),
                "in_game": _parse_bool(attributes.get("in_game")),
                "today": _parse_bool(attributes.get("today")),
                "refundable": _parse_bool(attributes.get("refundable")),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def _player_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        relationships = item.get("relationships") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "name": attributes.get("name"),
                "name_key": normalize_name(attributes.get("name")),
                "display_name": attributes.get("display_name"),
                "team": attributes.get("team"),
                "team_name": attributes.get("team_name"),
                "position": attributes.get("position"),
                "market": attributes.get("market"),
                "jersey_number": attributes.get("jersey_number"),
                "image_url": attributes.get("image_url"),
                "league_id": _to_str(attributes.get("league_id")),
                "league": attributes.get("league"),
                "combo": _parse_bool(attributes.get("combo")),
                "team_id": _relationship_id(relationships, "team_data"),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def _team_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "abbreviation": attributes.get("abbreviation"),
                "market": attributes.get("market"),
                "name": attributes.get("name"),
                "primary_color": attributes.get("primary_color"),
                "secondary_color": attributes.get("secondary_color"),
                "tertiary_color": attributes.get("tertiary_color"),
                "attributes": attributes,
            }
        )
    return rows


def _stat_type_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "name": attributes.get("name"),
                "rank": _parse_int(attributes.get("rank")),
                "lfg_ignored_leagues": attributes.get("lfg_ignored_leagues"),
                "attributes": attributes,
            }
        )
    return rows


def _game_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        relationships = item.get("relationships") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "start_time": _parse_datetime(attributes.get("start_time")),
                "end_time": _parse_datetime(attributes.get("end_time")),
                "status": attributes.get("status"),
                "is_live": _parse_bool(attributes.get("is_live")),
                "external_game_id": attributes.get("external_game_id"),
                "created_at": _parse_datetime(attributes.get("created_at")),
                "updated_at": _parse_datetime(attributes.get("updated_at")),
                "metadata": attributes.get("metadata"),
                "home_team_id": _relationship_id(relationships, "home_team_data"),
                "away_team_id": _relationship_id(relationships, "away_team_data"),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def _projection_type_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "name": attributes.get("name"),
                "attributes": attributes,
            }
        )
    return rows


def _league_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        relationships = item.get("relationships") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "name": attributes.get("name"),
                "rank": _parse_int(attributes.get("rank")),
                "active": _parse_bool(attributes.get("active")),
                "projections_count": _parse_int(attributes.get("projections_count")),
                "icon": attributes.get("icon"),
                "image_url": attributes.get("image_url"),
                "parent_id": attributes.get("parent_id"),
                "parent_name": attributes.get("parent_name"),
                "f2p_enabled": _parse_bool(attributes.get("f2p_enabled")),
                "has_live_projections": _parse_bool(attributes.get("has_live_projections")),
                "last_five_games_enabled": _parse_bool(attributes.get("last_five_games_enabled")),
                "league_icon_id": attributes.get("league_icon_id"),
                "show_trending": _parse_bool(attributes.get("show_trending")),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def _duration_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") or {}
        rows.append(
            {
                "id": _to_str(item.get("id")),
                "name": attributes.get("name"),
                "attributes": attributes,
            }
        )
    return rows


def _upsert_rows(
    conn: Connection,
    table,
    rows: list[dict[str, Any]],
    conflict_cols: list[str],
    *,
    batch_size: int | None = None,
) -> int:
    if not rows:
        return 0
    size = batch_size or _batch_size_for(table)
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


def _insert_rows(
    conn: Connection,
    table,
    rows: list[dict[str, Any]],
    conflict_cols: list[str],
    *,
    batch_size: int | None = None,
) -> int:
    if not rows:
        return 0
    size = batch_size or _batch_size_for(table)
    total = 0
    for batch in _chunk_rows(rows, size):
        stmt = pg_insert(table).values(batch)
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        result = conn.execute(stmt)
        batch_count = result.rowcount
        if batch_count is None or batch_count < 0:
            batch_count = len(batch)
        total += batch_count
    return total


def load_snapshot(
    payload: dict[str, Any],
    *,
    engine: Engine,
    snapshot_path: str | None = None,
    snapshot_id: UUID | None = None,
    league_id: str | None = None,
    per_page: int | None = None,
) -> dict[str, int]:
    snapshot_uuid = snapshot_id or _snapshot_uuid(snapshot_path)
    league = _to_str(league_id) or str(settings.prizepicks_league_id)
    per_page_value = per_page or settings.prizepicks_per_page

    included_items = payload.get("included") or []
    included_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in included_items:
        included_by_type[item.get("type", "unknown")].append(item)

    result_counts: dict[str, int] = {}

    with engine.begin() as conn:
        snapshot_row = _snapshot_row(
            snapshot_id=snapshot_uuid,
            payload=payload,
            snapshot_path=snapshot_path,
            league_id=league,
            per_page=per_page_value,
        )
        snapshot_insert = pg_insert(schema.snapshots).values(snapshot_row)
        if snapshot_path:
            snapshot_insert = snapshot_insert.on_conflict_do_nothing(
                index_elements=[schema.snapshots.c.snapshot_path],
                index_where=schema.snapshots.c.snapshot_path.isnot(None),
            )
        else:
            snapshot_insert = snapshot_insert.on_conflict_do_nothing(index_elements=[schema.snapshots.c.id])
        snapshot_result = conn.execute(snapshot_insert)
        if snapshot_result.rowcount == 0:
            return {"snapshots": 0, "skipped": 1, "snapshot_id": str(snapshot_uuid)}

        result_counts["players"] = _upsert_rows(
            conn, schema.players, _player_rows(included_by_type.get("new_player", [])), ["id"]
        )
        result_counts["teams"] = _upsert_rows(
            conn, schema.teams, _team_rows(included_by_type.get("team", [])), ["id"]
        )
        result_counts["stat_types"] = _upsert_rows(
            conn, schema.stat_types, _stat_type_rows(included_by_type.get("stat_type", [])), ["id"]
        )
        result_counts["games"] = _upsert_rows(
            conn, schema.games, _game_rows(included_by_type.get("game", [])), ["id"]
        )
        result_counts["projection_types"] = _upsert_rows(
            conn,
            schema.projection_types,
            _projection_type_rows(included_by_type.get("projection_type", [])),
            ["id"],
        )
        result_counts["leagues"] = _upsert_rows(
            conn, schema.leagues, _league_rows(included_by_type.get("league", [])), ["id"]
        )
        result_counts["durations"] = _upsert_rows(
            conn, schema.durations, _duration_rows(included_by_type.get("duration", [])), ["id"]
        )

        projections_rows = _projection_rows(payload.get("data") or [], snapshot_uuid)
        fetched_at = snapshot_row["fetched_at"]
        prev_snapshot_id = conn.execute(
            select(schema.snapshots.c.id)
            .where(schema.snapshots.c.league_id == league)
            .where(schema.snapshots.c.fetched_at < fetched_at)
            .order_by(schema.snapshots.c.fetched_at.desc())
            .limit(1)
        ).scalar()

        prev_lines = {}
        if prev_snapshot_id:
            prev_rows = conn.execute(
                text(
                    """
                    select
                        player_id,
                        stat_type_id,
                        projection_type_id,
                        game_id,
                        duration_id,
                        league_id,
                        line_score
                    from projections
                    where snapshot_id = :snapshot_id
                      and coalesce(odds_type, 0) = 0
                    """
                ),
                {"snapshot_id": prev_snapshot_id},
            ).all()
            for row in prev_rows:
                prev_lines[(
                    row.player_id,
                    row.stat_type_id,
                    row.projection_type_id,
                    row.game_id,
                    row.duration_id,
                    row.league_id,
                )] = row.line_score

        for row in projections_rows:
            key = (
                row.get("player_id"),
                row.get("stat_type_id"),
                row.get("projection_type_id"),
                row.get("game_id"),
                row.get("duration_id"),
                row.get("league_id"),
            )
            prev_line = prev_lines.get(key)
            current_line = row.get("line_score")
            row["line_score_prev"] = prev_line
            if current_line is not None and prev_line is not None:
                row["line_score_delta"] = current_line - prev_line
            else:
                row["line_score_delta"] = None
            row["line_movement"] = _line_movement(current_line, prev_line)
        result_counts["projections"] = _insert_rows(
            conn, schema.projections, projections_rows, ["snapshot_id", "projection_id"]
        )

        audit_summary = audit_snapshot(payload)
        audit_row = {
            "id": uuid4(),
            "snapshot_id": snapshot_uuid,
            "created_at": datetime.now(timezone.utc),
            "summary": audit_summary,
        }
        audit_stmt = pg_insert(schema.snapshot_audits).values(audit_row)
        audit_stmt = audit_stmt.on_conflict_do_nothing(index_elements=[schema.snapshot_audits.c.snapshot_id])
        conn.execute(audit_stmt)

    result_counts["snapshots"] = 1
    result_counts["snapshot_id"] = str(snapshot_uuid)
    return result_counts
