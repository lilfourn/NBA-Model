from decimal import Decimal

from app.db.prizepicks_loader import _line_movement


def test_line_movement_new() -> None:
    assert _line_movement(Decimal("1.0"), None) == "new"


def test_line_movement_up() -> None:
    assert _line_movement(Decimal("2.5"), Decimal("2.0")) == "up"


def test_line_movement_down() -> None:
    assert _line_movement(Decimal("1.5"), Decimal("2.0")) == "down"


def test_line_movement_same() -> None:
    assert _line_movement(Decimal("2.0"), Decimal("2.0")) == "same"


def test_line_movement_none() -> None:
    assert _line_movement(None, Decimal("2.0")) is None
