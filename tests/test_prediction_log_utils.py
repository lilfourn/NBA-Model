from __future__ import annotations

from decimal import Decimal

from app.db.prediction_logs import _normalize_pick, _resolve_over_under_outcome


def test_resolve_over_under_outcome_over() -> None:
    over_label, outcome = _resolve_over_under_outcome(line_score=20.5, actual_value=21.0)
    assert over_label == 1
    assert outcome == "over"


def test_resolve_over_under_outcome_under() -> None:
    over_label, outcome = _resolve_over_under_outcome(line_score=20.5, actual_value=19.0)
    assert over_label == 0
    assert outcome == "under"


def test_resolve_over_under_outcome_push() -> None:
    over_label, outcome = _resolve_over_under_outcome(line_score=20.5, actual_value=20.5)
    assert over_label == 0
    assert outcome == "push"


def test_normalize_pick_prefers_valid_input() -> None:
    assert _normalize_pick("under", prob_over=Decimal("0.9")) == "UNDER"


def test_normalize_pick_falls_back_to_probability() -> None:
    assert _normalize_pick(None, prob_over=Decimal("0.2")) == "UNDER"
    assert _normalize_pick(None, prob_over=Decimal("0.8")) == "OVER"
