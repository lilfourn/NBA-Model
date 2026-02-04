from app.collectors.normalizer import normalize_snapshot


def test_normalize_snapshot_projections() -> None:
    payload = {
        "data": [
            {
                "id": "1",
                "type": "projection",
                "attributes": {"line_score": 1.0},
                "relationships": {
                    "league": {"data": {"id": "7", "type": "league"}},
                    "projection_type": {"data": {"id": "2", "type": "projection_type"}},
                },
            }
        ],
        "included": [
            {
                "id": "7",
                "type": "league",
                "attributes": {"name": "NBA"},
                "relationships": {"projection_filters": {"data": []}},
            }
        ],
    }

    tables = normalize_snapshot(payload)

    projections = tables["projections"]
    assert projections[0]["id"] == "1"
    assert projections[0]["league_id"] == "7"
    assert projections[0]["projection_type_id"] == "2"
    assert projections[0]["line_score"] == 1.0

    leagues = tables["leagues"]
    assert leagues[0]["id"] == "7"
    assert leagues[0]["name"] == "NBA"
    assert leagues[0]["projection_filters_ids"] == []
