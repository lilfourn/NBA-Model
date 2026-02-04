from __future__ import annotations

import json
from typing import Any, Iterable

TABLE_NAME_MAP: dict[str, str] = {
    "projection": "projections",
    "new_player": "new_players",
    "stat_type": "stat_types",
    "team": "teams",
    "game": "games",
    "projection_type": "projection_types",
    "league": "leagues",
    "duration": "durations",
}


def _pluralize(value: str) -> str:
    if value.endswith("s"):
        return value
    return f"{value}s"


def _normalize_relationship(rel_name: str, rel_value: Any, row: dict[str, Any]) -> None:
    if not isinstance(rel_value, dict):
        return

    rel_data = rel_value.get("data")
    if isinstance(rel_data, dict):
        row[f"{rel_name}_id"] = rel_data.get("id")
        row[f"{rel_name}_type"] = rel_data.get("type")
        return

    if isinstance(rel_data, list):
        row[f"{rel_name}_ids"] = [entry.get("id") for entry in rel_data if isinstance(entry, dict)]
        row[f"{rel_name}_types"] = [
            entry.get("type") for entry in rel_data if isinstance(entry, dict)
        ]
        return

    row[f"{rel_name}_id"] = None
    row[f"{rel_name}_type"] = None


def normalize_item(item: dict[str, Any], *, source: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": item.get("id"),
        "type": item.get("type"),
        "source": source,
    }

    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        row.update(attributes)

    relationships = item.get("relationships")
    if isinstance(relationships, dict):
        for rel_name, rel_value in relationships.items():
            _normalize_relationship(rel_name, rel_value, row)

    return row


def normalize_snapshot(
    payload: dict[str, Any],
    *,
    include_data: bool = True,
    include_included: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {}

    def upsert_row(table_name: str, row: dict[str, Any], key: str) -> None:
        table = tables.setdefault(table_name, {})
        existing = table.get(key)
        if existing:
            for field, value in row.items():
                if value is not None:
                    existing[field] = value
        else:
            table[key] = row

    def process_items(items: Iterable[dict[str, Any]], source: str) -> None:
        for index, item in enumerate(items):
            item_type = item.get("type", "unknown")
            table_name = TABLE_NAME_MAP.get(item_type, _pluralize(item_type))
            row = normalize_item(item, source=source)
            key = str(row.get("id") or f"{source}_{item_type}_{index}")
            upsert_row(table_name, row, key)

    if include_data:
        process_items(payload.get("data") or [], source="data")
    if include_included:
        process_items(payload.get("included") or [], source="included")

    return {table_name: list(rows.values()) for table_name, rows in tables.items()}


def to_jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) for row in rows)
