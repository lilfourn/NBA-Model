from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

CENTRAL_TZ = ZoneInfo("America/Chicago")


def central_now() -> datetime:
    return datetime.now(tz=CENTRAL_TZ)


def central_today() -> date:
    return central_now().date()


def central_yesterday() -> date:
    return central_today() - timedelta(days=1)


def central_date_range(days: int) -> tuple[date, date]:
    if days <= 0:
        raise ValueError("days must be positive")
    end = central_yesterday()
    start = end - timedelta(days=days - 1)
    return start, end
