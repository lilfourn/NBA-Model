from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

REQUIRED_DATA_KEYS = ("id", "type", "attributes", "relationships")
REQUIRED_INCLUDED_KEYS = ("id", "type", "attributes")

PROJECTION_ATTRIBUTE_KEYS = (
    "line_score",
    "stat_type",
    "projection_type",
    "status",
    "start_time",
)

PROJECTION_RELATIONSHIP_KEYS = (
    "new_player",
    "stat_type",
    "game",
    "league",
    "duration",
    "projection_type",
)


def _missing_required(items: list[dict[str, Any]], required: tuple[str, ...]) -> dict[str, int]:
    counts = Counter()
    for item in items:
        for key in required:
            if key not in item or item.get(key) is None:
                counts[key] += 1
    return dict(counts)


def _attribute_nulls(items: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    counts = Counter()
    for item in items:
        attributes = item.get("attributes") or {}
        for key in keys:
            if attributes.get(key) is None:
                counts[key] += 1
    return dict(counts)


def _relationship_missing(items: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    counts = Counter()
    for item in items:
        relationships = item.get("relationships") or {}
        for key in keys:
            if key not in relationships:
                counts[key] += 1
                continue
            rel = relationships.get(key)
            if not isinstance(rel, dict) or rel.get("data") is None:
                counts[key] += 1
    return dict(counts)


def _duplicates(items: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, Any]:
    seen = set()
    dupes = 0
    for item in items:
        key = tuple(item.get(field) for field in key_fields)
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    return {"total": len(items), "duplicates": dupes}


def audit_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    data_items = payload.get("data") or []
    included_items = payload.get("included") or []

    included_types = Counter(item.get("type", "unknown") for item in included_items)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_count": len(data_items),
        "included_count": len(included_items),
        "links_keys": sorted((payload.get("links") or {}).keys()),
        "meta_keys": sorted((payload.get("meta") or {}).keys()),
        "data_required_missing": _missing_required(data_items, REQUIRED_DATA_KEYS),
        "included_required_missing": _missing_required(included_items, REQUIRED_INCLUDED_KEYS),
        "projection_attribute_nulls": _attribute_nulls(data_items, PROJECTION_ATTRIBUTE_KEYS),
        "projection_relationship_missing": _relationship_missing(data_items, PROJECTION_RELATIONSHIP_KEYS),
        "data_duplicate_ids": _duplicates(data_items, ("id", "type")),
        "included_duplicate_ids": _duplicates(included_items, ("id", "type")),
        "included_type_counts": dict(included_types),
    }

    return summary
