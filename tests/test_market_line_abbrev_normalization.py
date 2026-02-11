from __future__ import annotations

from scripts.nba.fetch_market_game_lines import (
    _event_game_date,
    _normalize_abbreviation,
    _parse_espn_rows,
)


def test_normalize_abbreviation_maps_espn_variants() -> None:
    assert _normalize_abbreviation("GS") == "GSW"
    assert _normalize_abbreviation("NO") == "NOP"
    assert _normalize_abbreviation("SA") == "SAS"
    assert _normalize_abbreviation("NY") == "NYK"
    assert _normalize_abbreviation("UTAH") == "UTA"


def test_parse_espn_rows_normalizes_team_abbreviations() -> None:
    events = [
        {
            "id": "evt-1",
            "competitions": [
                {
                    "date": "2026-02-11T23:00:00Z",
                    "competitors": [
                        {"homeAway": "home", "team": {"abbreviation": "GS"}},
                        {"homeAway": "away", "team": {"abbreviation": "NO"}},
                    ],
                    "odds": [
                        {
                            "provider": {"displayName": "ESPN Bet"},
                            "spread": "-4.5",
                            "overUnder": "230.5",
                            "moneyline": {
                                "home": {"close": {"odds": "-140"}},
                                "away": {"close": {"odds": "+120"}},
                            },
                        }
                    ],
                }
            ],
        }
    ]

    rows = _parse_espn_rows(events, provider="espn_public")

    assert len(rows) == 1
    row = rows[0]
    assert row["home_team_abbreviation"] == "GSW"
    assert row["away_team_abbreviation"] == "NOP"
    assert row["home_spread"] == -4.5
    assert row["away_spread"] == 4.5


def test_event_game_date_uses_new_york_calendar_day() -> None:
    # 01:00Z is still prior calendar day in America/New_York.
    assert str(_event_game_date("2026-02-12T01:00:00Z")) == "2026-02-11"
