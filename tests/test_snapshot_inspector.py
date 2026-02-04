from app.collectors.inspector import summarize_snapshot


def test_summarize_snapshot_basic() -> None:
    payload = {
        "data": [
            {
                "id": "1",
                "type": "projection",
                "attributes": {"line_score": 1.0},
                "relationships": {
                    "league": {"data": {"id": "7", "type": "league"}},
                },
            },
            {
                "id": "2",
                "type": "projection",
                "attributes": {"line_score": 2.0, "status": "pre_game"},
                "relationships": {
                    "league": {"data": None},
                    "new_player": {"data": {"id": "9", "type": "new_player"}},
                },
            },
        ],
        "included": [
            {
                "id": "9",
                "type": "new_player",
                "attributes": {"name": "Test Player"},
                "relationships": {
                    "league": {"data": {"id": "7", "type": "league"}},
                },
            }
        ],
        "links": {"self": "https://example.test"},
        "meta": {"page": 1},
    }

    summary = summarize_snapshot(payload)

    assert summary["top_level_keys"] == ["data", "included", "links", "meta"]
    assert summary["data"]["total"] == 2
    assert summary["data"]["types"] == {"projection": 2}

    projection_summary = summary["data"]["by_type"]["projection"]
    assert set(projection_summary["attribute_keys"]) == {"line_score", "status"}
    assert set(projection_summary["relationship_keys"]) == {"league", "new_player"}
    assert projection_summary["relationship_data_shapes"]["league"] == {"object": 1, "null": 1}
    assert projection_summary["relationship_data_shapes"]["new_player"] == {"object": 1}

    included_summary = summary["included"]
    assert included_summary["total"] == 1
    assert included_summary["types"] == {"new_player": 1}
