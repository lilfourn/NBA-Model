"""Tests for app.collectors.validators — per-source response validation."""
from __future__ import annotations

import pytest

from app.collectors.validators import (
    ValidationResult,
    validate_basketball_reference_html,
    validate_nba_stats_response,
    validate_prizepicks_response,
    validate_statmuse_html,
)


# ---------------------------------------------------------------------------
# PrizePicks
# ---------------------------------------------------------------------------
class TestValidatePrizePicks:
    def test_valid_response(self):
        payload = {
            "data": [
                {
                    "id": "1",
                    "type": "projection",
                    "attributes": {"line_score": 25.5},
                    "relationships": {"new_player": {"data": {"id": "p1"}}},
                }
            ]
        }
        result = validate_prizepicks_response(payload)
        assert result.valid
        assert not result.errors

    def test_not_dict(self):
        result = validate_prizepicks_response("not a dict")
        assert not result.valid

    def test_empty_data(self):
        result = validate_prizepicks_response({"data": []})
        assert not result.valid
        assert any("empty" in e for e in result.errors)

    def test_missing_data_key(self):
        result = validate_prizepicks_response({"included": []})
        assert not result.valid

    def test_all_missing_line_score(self):
        payload = {
            "data": [
                {"attributes": {}, "relationships": {"new_player": {"data": {"id": "p1"}}}},
            ]
        }
        result = validate_prizepicks_response(payload)
        assert not result.valid
        assert any("line_score" in e for e in result.errors)

    def test_partial_missing_warns(self):
        payload = {
            "data": [
                {"attributes": {"line_score": 25.5}, "relationships": {"new_player": {"data": {"id": "p1"}}}},
                {"attributes": {}, "relationships": {"new_player": {"data": {"id": "p2"}}}},
            ]
        }
        result = validate_prizepicks_response(payload)
        assert result.valid
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# NBA Stats
# ---------------------------------------------------------------------------
class TestValidateNbaStats:
    def test_valid_response(self):
        payload = {
            "resultSets": [
                {"rowSet": [["row1"], ["row2"]], "headers": ["COL"]}
            ]
        }
        result = validate_nba_stats_response(payload)
        assert result.valid

    def test_not_dict(self):
        result = validate_nba_stats_response([])
        assert not result.valid

    def test_missing_result_sets(self):
        result = validate_nba_stats_response({"parameters": {}})
        assert not result.valid

    def test_empty_row_set_warns(self):
        payload = {"resultSets": [{"rowSet": []}]}
        result = validate_nba_stats_response(payload)
        assert result.valid
        assert any("No rows" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Basketball Reference
# ---------------------------------------------------------------------------
class TestValidateBasketballReference:
    def test_valid_html(self):
        html = "x" * 600 + '<table id="pgl_basic">...</table>'
        result = validate_basketball_reference_html(html)
        assert result.valid

    def test_empty(self):
        result = validate_basketball_reference_html("")
        assert not result.valid

    def test_too_short(self):
        result = validate_basketball_reference_html("<html>short</html>")
        assert not result.valid

    def test_no_table(self):
        html = "x" * 600 + "<html>no table here</html>"
        result = validate_basketball_reference_html(html)
        assert not result.valid

    def test_rate_limited(self):
        html = "x" * 600 + "Rate limit exceeded 429"
        result = validate_basketball_reference_html(html)
        assert not result.valid


# ---------------------------------------------------------------------------
# StatMuse
# ---------------------------------------------------------------------------
class TestValidateStatMuse:
    def test_valid_html(self):
        html = "x" * 600 + "<table><tr><td>data</td></tr></table>"
        result = validate_statmuse_html(html)
        assert result.valid

    def test_empty(self):
        result = validate_statmuse_html("")
        assert not result.valid

    def test_no_table(self):
        html = "x" * 600 + "<div>just text</div>"
        result = validate_statmuse_html(html)
        assert not result.valid

    def test_no_results_warns(self):
        html = "x" * 600 + "<table></table> No results found"
        result = validate_statmuse_html(html)
        assert result.valid
        assert any("no results" in w.lower() for w in result.warnings)
