"""Per-source response validation rules.

Each validator returns a ValidationResult. Invalid responses should not be loaded.
Warnings are logged but data proceeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PrizePicks
# ---------------------------------------------------------------------------
def validate_prizepicks_response(payload: Any) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return ValidationResult(valid=False, errors=["Response is not a dict"])

    data = payload.get("data")
    if not isinstance(data, list):
        return ValidationResult(valid=False, errors=["'data' key missing or not a list"])
    if len(data) == 0:
        return ValidationResult(valid=False, errors=["'data' array is empty"])

    missing_line = 0
    missing_player = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes") or {}
        rels = item.get("relationships") or {}
        if attrs.get("line_score") is None:
            missing_line += 1
        player_rel = rels.get("new_player") or rels.get("player")
        if not player_rel or not isinstance(player_rel, dict):
            missing_player += 1
        elif not (player_rel.get("data") or {}).get("id"):
            missing_player += 1

    if missing_line > 0:
        warnings.append(f"{missing_line}/{len(data)} projections missing line_score")
    if missing_player > 0:
        warnings.append(f"{missing_player}/{len(data)} projections missing player relationship")
    if missing_line == len(data):
        errors.append("All projections missing line_score")
    if missing_player == len(data):
        errors.append("All projections missing player relationship")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# NBA Stats API
# ---------------------------------------------------------------------------
def validate_nba_stats_response(payload: Any) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return ValidationResult(valid=False, errors=["Response is not a dict"])

    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if result_sets is None:
        return ValidationResult(valid=False, errors=["'resultSets'/'resultSet' key missing"])

    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    if not isinstance(result_sets, list) or len(result_sets) == 0:
        return ValidationResult(valid=False, errors=["resultSets is empty or invalid"])

    total_rows = 0
    for rs in result_sets:
        if not isinstance(rs, dict):
            continue
        row_set = rs.get("rowSet")
        if isinstance(row_set, list):
            total_rows += len(row_set)

    if total_rows == 0:
        warnings.append("No rows in any resultSet (may be off-season or no games)")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Basketball Reference (HTML)
# ---------------------------------------------------------------------------
def validate_basketball_reference_html(html: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not html or not isinstance(html, str):
        return ValidationResult(valid=False, errors=["Empty or non-string HTML"])

    if len(html) < 500:
        return ValidationResult(valid=False, errors=["HTML too short (likely error page)"])

    if "pgl_basic" not in html and "pgl_basic_playoffs" not in html:
        errors.append("No game log table (pgl_basic) found in HTML")

    if "Rate limit" in html or "429" in html:
        errors.append("Rate limited by Basketball Reference")

    if "did not play" in html.lower():
        warnings.append("Page contains 'did not play' entries")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# StatMuse (HTML)
# ---------------------------------------------------------------------------
def validate_statmuse_html(html: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not html or not isinstance(html, str):
        return ValidationResult(valid=False, errors=["Empty or non-string HTML"])

    if len(html) < 500:
        return ValidationResult(valid=False, errors=["HTML too short (likely error page)"])

    if "<table" not in html.lower():
        errors.append("No <table> element found in HTML")

    if "no results" in html.lower() or "couldn't find" in html.lower():
        warnings.append("StatMuse returned no results for query")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
