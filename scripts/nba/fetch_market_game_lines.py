"""Fetch external sportsbook game lines and load `market_game_lines`.

Default provider: The Odds API (basketball_nba endpoint).
Requires env var: MARKET_ODDS_API_KEY
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.db.engine import get_engine  # noqa: E402

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

ABBREV_NORMALIZATION_MAP: dict[str, str] = {
    "GS": "GSW",
    "NO": "NOP",
    "SA": "SAS",
    "NY": "NYK",
    "UTAH": "UTA",
}


def _get_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 25) -> Any:
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    full_url = f"{url}?{query}" if query else url
    request = Request(full_url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _team_name_to_abbr(engine) -> dict[str, str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select upper(trim(coalesce(market, '') || ' ' || coalesce(name, ''))) as full_name,
                       upper(trim(coalesce(abbreviation, ''))) as abbreviation
                from teams
                where abbreviation is not null
                """
            )
        ).all()

    mapping: dict[str, str] = {}
    for row in rows:
        full_name = str(row.full_name or "").strip().upper()
        abbr = str(row.abbreviation or "").strip().upper()
        if full_name and abbr:
            mapping[full_name] = abbr

    mapping.update(
        {
            "LA CLIPPERS": "LAC",
            "LOS ANGELES CLIPPERS": "LAC",
            "LOS ANGELES LAKERS": "LAL",
            "NEW ORLEANS PELICANS": "NOP",
            "NEW YORK KNICKS": "NYK",
            "SAN ANTONIO SPURS": "SAS",
            "GOLDEN STATE WARRIORS": "GSW",
            "PHOENIX SUNS": "PHX",
            "BROOKLYN NETS": "BKN",
        }
    )
    return mapping


def _to_abbr(mapping: dict[str, str], raw_name: str | None) -> str | None:
    if not raw_name:
        return None
    key = str(raw_name).strip().upper()
    if key in mapping:
        return mapping[key]
    return mapping.get(" ".join(key.split()))


def _normalize_abbreviation(raw_abbr: str | None) -> str | None:
    if not raw_abbr:
        return None
    key = str(raw_abbr).strip().upper()
    if not key:
        return None
    return ABBREV_NORMALIZATION_MAP.get(key, key)


def _extract_market_values(markets: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "home_spread": None,
        "away_spread": None,
        "total_points": None,
        "home_moneyline": None,
        "away_moneyline": None,
    }

    for market in markets:
        key = str(market.get("key") or "")
        outcomes = market.get("outcomes") or []
        if not isinstance(outcomes, list):
            continue

        if key == "totals":
            points = [o.get("point") for o in outcomes if o.get("point") is not None]
            if points:
                try:
                    out["total_points"] = float(points[0])
                except (TypeError, ValueError):
                    pass
        elif key == "spreads":
            for o in outcomes:
                name = str(o.get("name") or "").strip().upper()
                point = o.get("point")
                if point is None:
                    continue
                try:
                    point_val = float(point)
                except (TypeError, ValueError):
                    continue
                if "HOME" in name:
                    out["home_spread"] = point_val
                elif "AWAY" in name:
                    out["away_spread"] = point_val
        elif key == "h2h":
            # Keep raw h2h values; home/away mapping happens at row assembly using team names.
            pass

    return out


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _fetch_espn_events(*, days_ahead: int = 2) -> list[dict[str, Any]]:
    events_by_id: dict[str, dict[str, Any]] = {}
    today = datetime.now(timezone.utc).date()
    horizon = max(0, int(days_ahead))
    for offset in range(horizon + 1):
        game_day = today + timedelta(days=offset)
        params = {"dates": game_day.strftime("%Y%m%d")}
        payload = _get_json(ESPN_SCOREBOARD_URL, params=params, timeout=25)
        events = payload.get("events") or []
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("id") or "").strip()
            if event_id:
                events_by_id[event_id] = event
    return list(events_by_id.values())


def _parse_espn_rows(events: list[dict[str, Any]], *, provider: str) -> list[dict[str, Any]]:
    captured_at = datetime.now(timezone.utc)
    captured_at_iso = captured_at.replace(microsecond=0).isoformat()
    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id") or "").strip()
        competitions = event.get("competitions") or []
        if not event_id or not isinstance(competitions, list) or not competitions:
            continue
        comp = competitions[0] if isinstance(competitions[0], dict) else {}
        event_time = str(comp.get("date") or event.get("date") or "")
        if not event_time:
            continue
        try:
            game_date = datetime.fromisoformat(event_time.replace("Z", "+00:00")).date()
        except ValueError:
            continue

        home_abbr: str | None = None
        away_abbr: str | None = None
        competitors = comp.get("competitors") or []
        if isinstance(competitors, list):
            for competitor in competitors:
                if not isinstance(competitor, dict):
                    continue
                team = competitor.get("team") or {}
                if not isinstance(team, dict):
                    continue
                abbr = _normalize_abbreviation(str(team.get("abbreviation") or ""))
                if not abbr:
                    continue
                side = str(competitor.get("homeAway") or "").strip().lower()
                if side == "home":
                    home_abbr = abbr
                elif side == "away":
                    away_abbr = abbr
        if not home_abbr or not away_abbr:
            continue

        odds_entries = comp.get("odds") or []
        if not isinstance(odds_entries, list):
            continue
        for odds in odds_entries:
            if not isinstance(odds, dict):
                continue
            provider_obj = odds.get("provider") or {}
            if not isinstance(provider_obj, dict):
                provider_obj = {}
            book_display = str(
                provider_obj.get("displayName") or provider_obj.get("name") or "espn"
            ).strip()
            book = re.sub(r"[^a-z0-9]+", "_", book_display.lower()).strip("_") or "espn"

            spread = None
            try:
                if odds.get("spread") is not None:
                    spread = float(odds.get("spread"))
            except (TypeError, ValueError):
                spread = None
            point_spread = odds.get("pointSpread") or {}
            if spread is None and isinstance(point_spread, dict):
                home_close = ((point_spread.get("home") or {}).get("close") or {}).get("line")
                try:
                    spread = float(str(home_close).strip()) if home_close is not None else None
                except (TypeError, ValueError):
                    spread = None

            total_points = None
            try:
                if odds.get("overUnder") is not None:
                    total_points = float(odds.get("overUnder"))
            except (TypeError, ValueError):
                total_points = None
            total_obj = odds.get("total") or {}
            if total_points is None and isinstance(total_obj, dict):
                over_line = ((total_obj.get("over") or {}).get("close") or {}).get("line")
                if over_line is not None:
                    digits = re.sub(r"^[^0-9.-]+", "", str(over_line).strip())
                    try:
                        total_points = float(digits)
                    except (TypeError, ValueError):
                        total_points = None

            moneyline_obj = odds.get("moneyline") or {}
            home_moneyline = None
            away_moneyline = None
            if isinstance(moneyline_obj, dict):
                home_moneyline = _safe_int(
                    ((moneyline_obj.get("home") or {}).get("close") or {}).get("odds")
                )
                away_moneyline = _safe_int(
                    ((moneyline_obj.get("away") or {}).get("close") or {}).get("odds")
                )

            row_id = _stable_line_id(provider, book, event_id, captured_at_iso)
            rows.append(
                {
                    "id": row_id,
                    "provider": provider,
                    "book": book,
                    "captured_at": captured_at,
                    "game_date": game_date,
                    "home_team_abbreviation": home_abbr,
                    "away_team_abbreviation": away_abbr,
                    "home_spread": spread,
                    "away_spread": (-spread if spread is not None else None),
                    "total_points": total_points,
                    "home_moneyline": home_moneyline,
                    "away_moneyline": away_moneyline,
                    "source_payload": json.dumps(odds, ensure_ascii=True),
                }
            )
    return rows


def _stable_line_id(
    provider: str,
    book: str,
    event_id: str,
    captured_at_iso: str,
) -> str:
    digest = hashlib.sha1(
        f"{provider}|{book}|{event_id}|{captured_at_iso}".encode("utf-8")
    ).hexdigest()
    return digest[:40]


def _fetch_odds_events(api_key: str, *, regions: str, bookmakers: str | None = None) -> list[dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    payload = _get_json(ODDS_API_URL, params=params, timeout=25)
    return payload if isinstance(payload, list) else []


def _parse_rows(events: list[dict[str, Any]], team_map: dict[str, str], *, provider: str) -> list[dict[str, Any]]:
    captured_at = datetime.now(timezone.utc)
    captured_at_iso = captured_at.replace(microsecond=0).isoformat()
    rows: list[dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("id") or "").strip()
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        commence_time = str(event.get("commence_time") or "")
        if not event_id or not home_team or not away_team or not commence_time:
            continue

        home_abbr = _normalize_abbreviation(_to_abbr(team_map, home_team))
        away_abbr = _normalize_abbreviation(_to_abbr(team_map, away_team))
        if not home_abbr or not away_abbr:
            continue

        try:
            game_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).date()
        except ValueError:
            continue

        bookmakers = event.get("bookmakers") or []
        if not isinstance(bookmakers, list):
            continue

        for book in bookmakers:
            book_key = str(book.get("key") or "").strip().lower()
            if not book_key:
                continue
            markets = book.get("markets") or []
            if not isinstance(markets, list):
                continue

            parsed = _extract_market_values(markets)

            # H2H prices by explicit team names.
            for market in markets:
                if str(market.get("key") or "") != "h2h":
                    continue
                for outcome in market.get("outcomes") or []:
                    name = str(outcome.get("name") or "").strip()
                    price = outcome.get("price")
                    if price is None:
                        continue
                    try:
                        price_val = int(price)
                    except (TypeError, ValueError):
                        continue
                    if name.upper() == home_team.upper():
                        parsed["home_moneyline"] = price_val
                    elif name.upper() == away_team.upper():
                        parsed["away_moneyline"] = price_val

            row_id = _stable_line_id(provider, book_key, event_id, captured_at_iso)
            rows.append(
                {
                    "id": row_id,
                    "provider": provider,
                    "book": book_key,
                    "captured_at": captured_at,
                    "game_date": game_date,
                    "home_team_abbreviation": home_abbr,
                    "away_team_abbreviation": away_abbr,
                    "home_spread": parsed.get("home_spread"),
                    "away_spread": parsed.get("away_spread"),
                    "total_points": parsed.get("total_points"),
                    "home_moneyline": parsed.get("home_moneyline"),
                    "away_moneyline": parsed.get("away_moneyline"),
                    "source_payload": json.dumps(book, ensure_ascii=True),
                }
            )

    return rows


def _upsert_rows(engine, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    stmt = text(
        """
        insert into market_game_lines (
            id, provider, book, captured_at, game_date,
            home_team_abbreviation, away_team_abbreviation,
            home_spread, away_spread, total_points,
            home_moneyline, away_moneyline, source_payload
        ) values (
            :id, :provider, :book, :captured_at, :game_date,
            :home_team_abbreviation, :away_team_abbreviation,
            :home_spread, :away_spread, :total_points,
            :home_moneyline, :away_moneyline, cast(:source_payload as jsonb)
        )
        on conflict (id) do update set
            provider = excluded.provider,
            book = excluded.book,
            captured_at = excluded.captured_at,
            game_date = excluded.game_date,
            home_team_abbreviation = excluded.home_team_abbreviation,
            away_team_abbreviation = excluded.away_team_abbreviation,
            home_spread = excluded.home_spread,
            away_spread = excluded.away_spread,
            total_points = excluded.total_points,
            home_moneyline = excluded.home_moneyline,
            away_moneyline = excluded.away_moneyline,
            source_payload = excluded.source_payload
        """
    )

    with engine.begin() as conn:
        conn.execute(stmt, rows)
    return len(rows)


def _rows_matching_nba_games(engine, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    keys: list[tuple[object, str, str]] = []
    for row in rows:
        game_date = row.get("game_date")
        home = _normalize_abbreviation(row.get("home_team_abbreviation"))
        away = _normalize_abbreviation(row.get("away_team_abbreviation"))
        if game_date is None or not home or not away:
            continue
        keys.append((game_date, home, away))
    if not keys:
        return 0

    game_dates = sorted({game_date for game_date, _, _ in keys})
    with engine.connect() as conn:
        matched = conn.execute(
            text(
                """
                select game_date, home_team_abbreviation, away_team_abbreviation
                from nba_games
                where game_date = any(:game_dates)
                """
            ),
            {"game_dates": game_dates},
        ).all()

    matched_keys = {
        (
            row.game_date,
            _normalize_abbreviation(row.home_team_abbreviation),
            _normalize_abbreviation(row.away_team_abbreviation),
        )
        for row in matched
    }
    return sum(1 for key in keys if key in matched_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and load external market game lines.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "the_odds_api", "espn"],
        help="Market line source. 'auto' uses The Odds API when key exists, else ESPN public odds feed.",
    )
    parser.add_argument("--provider", default="market_feed")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", default=None)
    parser.add_argument("--days-ahead", type=int, default=2)
    args = parser.parse_args()

    api_key = os.getenv("MARKET_ODDS_API_KEY", "").strip()
    engine = get_engine(args.database_url)
    source = str(args.source).strip().lower()
    if source == "auto":
        source = "the_odds_api" if api_key else "espn"
    provider = str(args.provider).strip() or source

    if source == "the_odds_api":
        if not api_key:
            print("MARKET_ODDS_API_KEY missing; falling back to ESPN public odds feed.")
            source = "espn"
        else:
            team_map = _team_name_to_abbr(engine)
            events = _fetch_odds_events(
                api_key, regions=args.regions, bookmakers=args.bookmakers
            )
            rows = _parse_rows(events, team_map, provider=provider or "the_odds_api")
            upserted = _upsert_rows(engine, rows)
            matched_rows = _rows_matching_nba_games(engine, rows)
            print(
                {
                    "source": source,
                    "provider": provider,
                    "events": len(events),
                    "rows_parsed": len(rows),
                    "rows_upserted": upserted,
                    "rows_matching_nba_games": matched_rows,
                    "join_match_rate": round(matched_rows / len(rows), 4) if rows else 0.0,
                    "database": bool(settings.database_url or args.database_url),
                }
            )
            return

    events = _fetch_espn_events(days_ahead=int(args.days_ahead))
    rows = _parse_espn_rows(events, provider=provider or "espn_public")
    upserted = _upsert_rows(engine, rows)
    matched_rows = _rows_matching_nba_games(engine, rows)

    print(
        {
            "source": source,
            "provider": provider,
            "events": len(events),
            "rows_parsed": len(rows),
            "rows_upserted": upserted,
            "rows_matching_nba_games": matched_rows,
            "join_match_rate": round(matched_rows / len(rows), 4) if rows else 0.0,
            "database": bool(settings.database_url or args.database_url),
        }
    )


if __name__ == "__main__":
    main()
