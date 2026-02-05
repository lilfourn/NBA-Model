from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from curl_cffi import requests as curl_requests

from app.core.config import settings

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": settings.nba_stats_origin,
    "Referer": settings.nba_stats_referer,
    "User-Agent": settings.nba_stats_user_agent,
}


def _request(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{settings.nba_stats_api_url.rstrip('/')}/{endpoint}"
    last_error: Exception | None = None

    for attempt in range(settings.nba_stats_max_retries):
        try:
            response = curl_requests.get(
                url,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=settings.nba_stats_timeout_seconds,
                impersonate=settings.nba_stats_impersonate,
                proxy=settings.nba_stats_proxy or None,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_for = settings.nba_stats_backoff_seconds * (attempt + 1)
            time.sleep(sleep_for)

    if last_error:
        raise last_error
    raise RuntimeError("NBA stats request failed")


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    if not result_sets:
        return []

    rows: list[dict[str, Any]] = []
    for result in result_sets:
        headers = result.get("headers") or []
        for row in result.get("rowSet") or []:
            rows.append({header: row[idx] for idx, header in enumerate(headers)})
    return rows


def _format_date(value: str | None) -> str | None:
    if not value:
        return value
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return value
    return parsed.strftime("%m/%d/%Y")


def fetch_player_gamelogs(
    *,
    season: str,
    season_type: str = "Regular Season",
    league_id: str = "00",
    date_from: str | None = None,
    date_to: str | None = None,
    player_id: str | None = None,
    team_id: str | None = None,
    per_mode: str = "Totals",
    measure_type: str = "Base",
    last_n_games: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "Season": season,
        "SeasonType": season_type,
        "LeagueID": league_id,
        "PerMode": per_mode,
        "MeasureType": measure_type,
    }
    if date_from:
        params["DateFrom"] = _format_date(date_from)
    if date_to:
        params["DateTo"] = _format_date(date_to)
    if player_id:
        params["PlayerID"] = player_id
    if team_id:
        params["TeamID"] = team_id
    if last_n_games is not None:
        params["LastNGames"] = int(last_n_games)

    return _request("playergamelogs", params)


def fetch_league_game_log(
    *,
    season: str,
    season_type: str = "Regular Season",
    date_from: str = "",
    date_to: str = "",
) -> list[dict[str, Any]]:
    payload = _request(
        "leaguegamelog",
        {
            "Counter": "0",
            "DateFrom": _format_date(date_from) or "",
            "DateTo": _format_date(date_to) or "",
            "Direction": "DESC",
            "LeagueID": "00",
            "PlayerOrTeam": "P",
            "Season": season,
            "SeasonType": season_type,
            "Sorter": "DATE",
        },
    )
    return _extract_rows(payload)


def fetch_shot_chart_detail(
    *,
    game_id: str,
    season: str,
    season_type: str = "Regular Season",
    player_id: str = "0",
    team_id: str = "0",
    league_id: str = "00",
    context_measure: str = "FGA",
) -> list[dict[str, Any]]:
    """
    Fetch shot chart rows for a single game/player (player_id=0 returns all players).

    This is used for derived markets like Dunks, where we count made dunk attempts
    from ACTION_TYPE.
    """
    params: dict[str, Any] = {
        "AheadBehind": "",
        "CFID": "33",
        "CFPARAMS": season,
        "ClutchTime": "",
        "Conference": "",
        "ContextFilter": "",
        "ContextMeasure": context_measure,
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "EndPeriod": "10",
        "EndRange": "28800",
        "GROUP_ID": "",
        "GameEventID": "",
        "GameID": game_id,
        "GameSegment": "",
        "LastNGames": "0",
        "LeagueID": league_id,
        "Location": "",
        "Month": "0",
        "OnOff": "",
        "OpponentTeamID": "0",
        "Outcome": "",
        "Period": "0",
        "PlayerID": player_id,
        "PlayerPosition": "",
        "PointDiff": "",
        "Position": "",
        "RangeType": "0",
        "RookieYear": "",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": season_type,
        "StartPeriod": "1",
        "StartRange": "0",
        "TeamID": team_id,
        "VsConference": "",
        "VsDivision": "",
        "VsPlayerID1": "",
        "VsPlayerID2": "",
        "VsPlayerID3": "",
        "VsPlayerID4": "",
        "VsPlayerID5": "",
        "VsTeamID": "",
    }
    payload = _request("shotchartdetail", params)
    result_sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    for result in result_sets:
        if (result.get("name") or "").lower() == "shot_chart_detail":
            headers = result.get("headers") or []
            rows = result.get("rowSet") or []
            out: list[dict[str, Any]] = []
            for row in rows:
                out.append({header: row[idx] for idx, header in enumerate(headers)})
            return out
    return []
