from app.db.nba_loader import _dedupe_rows_by_conflict


def test_dedupe_rows_by_conflict_merges_non_null() -> None:
    rows = [
        {"id": "1", "full_name": "Alice", "team_id": None},
        {"id": "1", "full_name": None, "team_id": "10"},
        {"id": "2", "full_name": "Bob", "team_id": "20"},
    ]

    deduped = _dedupe_rows_by_conflict(rows, ["id"])

    assert len(deduped) == 2
    row1 = next(row for row in deduped if row["id"] == "1")
    assert row1["full_name"] == "Alice"
    assert row1["team_id"] == "10"
