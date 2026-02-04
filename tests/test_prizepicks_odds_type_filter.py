from uuid import uuid4

from app.db.prizepicks_loader import _projection_rows


def test_prizepicks_loader_filters_non_standard_odds_types() -> None:
    snapshot_id = uuid4()
    items = [
        {"id": "standard", "attributes": {"odds_type": "standard", "line_score": 1.0}, "relationships": {}},
        {"id": "demon", "attributes": {"odds_type": "demon", "line_score": 2.0}, "relationships": {}},
        {"id": "goblin", "attributes": {"odds_type": "goblin", "line_score": 3.0}, "relationships": {}},
        {"id": "missing", "attributes": {"line_score": 4.0}, "relationships": {}},
        {"id": "unknown", "attributes": {"odds_type": "boost", "line_score": 5.0}, "relationships": {}},
    ]

    rows = _projection_rows(items, snapshot_id)
    ids = {row["projection_id"] for row in rows}

    assert ids == {"standard", "missing"}
    for row in rows:
        assert row["odds_type"] == 0
