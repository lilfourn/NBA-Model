"""Tests for unified name normalization, team abbreviation normalization, and type coercion."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest


# ---------------------------------------------------------------------------
# Name normalization (app/utils/names.py)
# ---------------------------------------------------------------------------
class TestNormalizeName:
    def test_basic_name(self):
        from app.utils.names import normalize_name
        assert normalize_name("LeBron James") == "lebronjames"

    def test_none_returns_none(self):
        from app.utils.names import normalize_name
        assert normalize_name(None) is None

    def test_empty_returns_none(self):
        from app.utils.names import normalize_name
        assert normalize_name("") is None
        assert normalize_name("   ") is None

    def test_strips_suffix_jr(self):
        from app.utils.names import normalize_name
        # "Kelly Oubre Jr." → suffix stripped → "kelly oubre" → key "kellyoubre"
        assert normalize_name("Kelly Oubre Jr.") == "kellyoubre"

    def test_strips_suffix_iii(self):
        from app.utils.names import normalize_name
        assert normalize_name("Robert Williams III") == "robertwilliams"

    def test_unicode_diacritics(self):
        from app.utils.names import normalize_name
        # Luka Dončić → strip diacritics → "luka doncic" → "lukadoncic"
        assert normalize_name("Luka Dončić") == "lukadoncic"

    def test_numeric_input(self):
        from app.utils.names import normalize_name
        assert normalize_name(12345) == "12345"


class TestNormalizePlayerName:
    def test_returns_human_readable(self):
        from app.utils.names import normalize_player_name
        assert normalize_player_name("LeBron James") == "lebron james"

    def test_strips_suffix(self):
        from app.utils.names import normalize_player_name
        assert normalize_player_name("Gary Trent Jr.") == "gary trent"

    def test_unicode(self):
        from app.utils.names import normalize_player_name
        assert normalize_player_name("Nikola Jokić") == "nikola jokic"

    def test_empty(self):
        from app.utils.names import normalize_player_name
        assert normalize_player_name(None) == ""
        assert normalize_player_name("") == ""


class TestNameUtilsReexport:
    def test_modeling_name_utils_delegates(self):
        from app.modeling.name_utils import normalize_player_name
        assert normalize_player_name("LeBron James") == "lebron james"


# ---------------------------------------------------------------------------
# Team abbreviation normalization (app/utils/teams.py)
# ---------------------------------------------------------------------------
class TestNormalizeTeamAbbrev:
    def test_known_override(self):
        from app.utils.teams import normalize_team_abbrev
        assert normalize_team_abbrev("BRK") == "BKN"
        assert normalize_team_abbrev("PHO") == "PHX"

    def test_passthrough(self):
        from app.utils.teams import normalize_team_abbrev
        assert normalize_team_abbrev("LAL") == "LAL"

    def test_case_insensitive(self):
        from app.utils.teams import normalize_team_abbrev
        assert normalize_team_abbrev("brk") == "BKN"
        assert normalize_team_abbrev("pho") == "PHX"

    def test_none_returns_none(self):
        from app.utils.teams import normalize_team_abbrev
        assert normalize_team_abbrev(None) is None

    def test_empty_returns_none(self):
        from app.utils.teams import normalize_team_abbrev
        assert normalize_team_abbrev("") is None

    def test_all_overrides(self):
        from app.utils.teams import normalize_team_abbrev
        expected = {
            "BRK": "BKN", "BK": "BKN", "NJN": "BKN",
            "PHO": "PHX", "NO": "NOP", "NOK": "NOP",
            "NY": "NYK", "SA": "SAS", "GS": "GSW",
            "WSH": "WAS", "CHO": "CHA", "MINN": "MIN",
            "UTAH": "UTA",
        }
        for src, target in expected.items():
            assert normalize_team_abbrev(src) == target, f"{src} -> expected {target}"


# ---------------------------------------------------------------------------
# Type coercion (app/db/coerce.py)
# ---------------------------------------------------------------------------
class TestParseBool:
    def test_none(self):
        from app.db.coerce import parse_bool
        assert parse_bool(None) is None

    def test_bool(self):
        from app.db.coerce import parse_bool
        assert parse_bool(True) is True
        assert parse_bool(False) is False

    def test_int(self):
        from app.db.coerce import parse_bool
        assert parse_bool(1) is True
        assert parse_bool(0) is False

    def test_string(self):
        from app.db.coerce import parse_bool
        assert parse_bool("true") is True
        assert parse_bool("false") is False
        assert parse_bool("yes") is True
        assert parse_bool("no") is False
        assert parse_bool("maybe") is None

    def test_nan(self):
        from app.db.coerce import parse_bool
        assert parse_bool(float("nan")) is None


class TestParseInt:
    def test_none(self):
        from app.db.coerce import parse_int
        assert parse_int(None) is None

    def test_int(self):
        from app.db.coerce import parse_int
        assert parse_int(42) == 42

    def test_float(self):
        from app.db.coerce import parse_int
        assert parse_int(42.7) == 42

    def test_string(self):
        from app.db.coerce import parse_int
        assert parse_int("42") == 42
        assert parse_int("42.9") == 42

    def test_nan(self):
        from app.db.coerce import parse_int
        assert parse_int(float("nan")) is None

    def test_inf(self):
        from app.db.coerce import parse_int
        assert parse_int(float("inf")) is None


class TestParseDecimal:
    def test_none(self):
        from app.db.coerce import parse_decimal
        assert parse_decimal(None) is None

    def test_bool_returns_none(self):
        from app.db.coerce import parse_decimal
        assert parse_decimal(True) is None
        assert parse_decimal(False) is None

    def test_decimal(self):
        from app.db.coerce import parse_decimal
        assert parse_decimal(Decimal("3.14")) == Decimal("3.14")

    def test_float(self):
        from app.db.coerce import parse_decimal
        result = parse_decimal(3.14)
        assert isinstance(result, Decimal)
        assert float(result) == pytest.approx(3.14)

    def test_nan_returns_none(self):
        from app.db.coerce import parse_decimal
        assert parse_decimal(float("nan")) is None

    def test_inf_returns_none(self):
        from app.db.coerce import parse_decimal
        assert parse_decimal(float("inf")) is None
        assert parse_decimal(Decimal("Infinity")) is None

    def test_string(self):
        from app.db.coerce import parse_decimal
        assert parse_decimal("3.14") == Decimal("3.14")
        assert parse_decimal("") is None
        assert parse_decimal("abc") is None


class TestParseDatetime:
    def test_none(self):
        from app.db.coerce import parse_datetime
        assert parse_datetime(None) is None

    def test_datetime_with_tz(self):
        from app.db.coerce import parse_datetime
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        assert parse_datetime(dt) == dt

    def test_datetime_naive_gets_utc(self):
        from app.db.coerce import parse_datetime
        dt = datetime(2024, 1, 15, 14, 30)
        result = parse_datetime(dt)
        assert result.tzinfo == timezone.utc

    def test_iso_string(self):
        from app.db.coerce import parse_datetime
        result = parse_datetime("2024-01-15T14:30:00+00:00")
        assert result is not None
        assert result.year == 2024

    def test_z_suffix(self):
        from app.db.coerce import parse_datetime
        result = parse_datetime("2024-01-15T14:30:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_empty_string(self):
        from app.db.coerce import parse_datetime
        assert parse_datetime("") is None
        assert parse_datetime("   ") is None


class TestParseDate:
    def test_none(self):
        from app.db.coerce import parse_date
        assert parse_date(None) is None

    def test_date_string(self):
        from app.db.coerce import parse_date
        assert parse_date("2024-01-15") == date(2024, 1, 15)

    def test_datetime_extracts_date(self):
        from app.db.coerce import parse_date
        dt = datetime(2024, 1, 15, 14, 30)
        assert parse_date(dt) == date(2024, 1, 15)

    def test_iso_with_time(self):
        from app.db.coerce import parse_date
        assert parse_date("2024-01-15T14:30:00") == date(2024, 1, 15)

    def test_empty(self):
        from app.db.coerce import parse_date
        assert parse_date("") is None


class TestNormalizeId:
    def test_none(self):
        from app.db.coerce import normalize_id
        assert normalize_id(None) is None

    def test_int(self):
        from app.db.coerce import normalize_id
        assert normalize_id(1610612737) == "1610612737"

    def test_float_integer(self):
        from app.db.coerce import normalize_id
        assert normalize_id(1610612737.0) == "1610612737"

    def test_string_with_trailing_dot_zero(self):
        from app.db.coerce import normalize_id
        assert normalize_id("1610612737.0") == "1610612737"

    def test_bool_returns_none(self):
        from app.db.coerce import normalize_id
        assert normalize_id(True) is None

    def test_nan_returns_none(self):
        from app.db.coerce import normalize_id
        assert normalize_id(float("nan")) is None


class TestJsonSafe:
    def test_decimal(self):
        from app.db.coerce import json_safe
        assert json_safe(Decimal("3.14")) == 3.14

    def test_datetime(self):
        from app.db.coerce import json_safe
        dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        assert json_safe(dt) == dt.isoformat()

    def test_nan_float(self):
        from app.db.coerce import json_safe
        assert json_safe(float("nan")) is None

    def test_nested_dict(self):
        from app.db.coerce import json_safe
        result = json_safe({"a": Decimal("1.5"), "b": [Decimal("2.5")]})
        assert result == {"a": 1.5, "b": [2.5]}


# ---------------------------------------------------------------------------
# Stats JSON normalization (app/db/nba_loader.py)
# ---------------------------------------------------------------------------
class TestNormalizeStatsJson:
    def test_uppercases_keys(self):
        from app.db.nba_loader import _normalize_stats_json
        result = _normalize_stats_json({"pts": 25, "reb": 10})
        assert result == {"PTS": 25, "REB": 10}

    def test_non_dict_returns_none(self):
        from app.db.nba_loader import _normalize_stats_json
        assert _normalize_stats_json("not a dict") is None
        assert _normalize_stats_json(None) is None

    def test_already_uppercase(self):
        from app.db.nba_loader import _normalize_stats_json
        result = _normalize_stats_json({"PTS": 25, "OREB": 3})
        assert result == {"PTS": 25, "OREB": 3}
