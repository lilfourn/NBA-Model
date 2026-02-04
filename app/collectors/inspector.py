from __future__ import annotations

from collections import Counter
from typing import Any


def _normalize_relationship_data_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def summarize_items(items: list[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
    if limit is not None:
        items = items[:limit]

    total = len(items)
    type_counts: Counter[str] = Counter()
    by_type: dict[str, dict[str, Any]] = {}

    for item in items:
        item_type = item.get("type", "unknown")
        type_counts[item_type] += 1

        entry = by_type.setdefault(
            item_type,
            {
                "count": 0,
                "attribute_keys": set(),
                "relationship_keys": set(),
                "attribute_key_counts": Counter(),
                "relationship_key_counts": Counter(),
                "relationship_data_shapes": {},
                "sample_ids": [],
            },
        )
        entry["count"] += 1

        item_id = item.get("id")
        if item_id is not None and len(entry["sample_ids"]) < 3:
            entry["sample_ids"].append(item_id)

        attributes = item.get("attributes") or {}
        if isinstance(attributes, dict):
            entry["attribute_keys"].update(attributes.keys())
            entry["attribute_key_counts"].update(attributes.keys())

        relationships = item.get("relationships") or {}
        if isinstance(relationships, dict):
            entry["relationship_keys"].update(relationships.keys())
            entry["relationship_key_counts"].update(relationships.keys())

            for rel_name, rel_value in relationships.items():
                rel_data = None
                if isinstance(rel_value, dict):
                    rel_data = rel_value.get("data")
                shape = _normalize_relationship_data_shape(rel_data)
                rel_shapes = entry["relationship_data_shapes"].setdefault(rel_name, Counter())
                rel_shapes[shape] += 1

    for entry in by_type.values():
        entry["attribute_keys"] = sorted(entry["attribute_keys"])
        entry["relationship_keys"] = sorted(entry["relationship_keys"])
        entry["attribute_key_counts"] = dict(entry["attribute_key_counts"].most_common())
        entry["relationship_key_counts"] = dict(entry["relationship_key_counts"].most_common())
        entry["relationship_data_shapes"] = {
            key: dict(counts) for key, counts in entry["relationship_data_shapes"].items()
        }

    return {
        "total": total,
        "types": dict(type_counts.most_common()),
        "by_type": by_type,
    }


def summarize_snapshot(payload: dict[str, Any], *, limit: int | None = None) -> dict[str, Any]:
    data_items = payload.get("data") or []
    included_items = payload.get("included") or []

    summary: dict[str, Any] = {
        "top_level_keys": sorted(payload.keys()),
        "links_keys": sorted((payload.get("links") or {}).keys()),
        "meta_keys": sorted((payload.get("meta") or {}).keys()),
        "data": summarize_items(data_items, limit=limit),
        "included": summarize_items(included_items, limit=limit),
    }
    return summary
