from __future__ import annotations

import re
from typing import Any

NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = NON_ALNUM.sub("", text)
    return normalized or None
