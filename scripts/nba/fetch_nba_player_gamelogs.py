import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from threading import Lock
from datetime import date, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.basketball_reference import fetch_player_gamelog_html, search_player_slug
from app.clients.nba_stats import fetch_player_gamelogs
from app.clients.statmuse import build_player_gamelog_query, fetch_ask_html
from app.collectors.basketball_reference import normalize_gamelogs as normalize_bref
from app.collectors.basketball_reference import parse_gamelog_table
from app.collectors.nba_stats import extract_result_rows, normalize_player_gamelogs, write_jsonl
from app.collectors.statmuse import normalize_gamelogs as normalize_statmuse
from app.collectors.statmuse import parse_first_stats_table
from app.modeling.types import PlayerGameLog
from app.modeling.time_utils import central_date_range, central_today, central_yesterday
from app.modeling.name_utils import normalize_player_name

DEFAULT_OUTPUT_DIR = Path("data/official")


def _today_iso() -> str:
    return central_today().isoformat()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _apply_date_range(logs: list[PlayerGameLog], start: date, end: date) -> list[PlayerGameLog]:
    filtered: list[PlayerGameLog] = []
    for log in logs:
        if not log.game_date:
            continue
        if start <= log.game_date <= end:
            filtered.append(log)
    return filtered


def _season_end_year(season: str) -> int:
    parts = season.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid season format: {season}")
    start = int(parts[0])
    end = int(parts[1])
    if end < 100:
        end += (start // 100) * 100
    if end < start:
        end += 100
    return end


def _load_players(players_file: str | None) -> list[dict[str, str]]:
    if players_file:
        path = Path(players_file)
        if not path.exists():
            raise FileNotFoundError(f"Players file not found: {players_file}")
        if path.suffix == ".jsonl":
            players: list[dict[str, str]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("combo"):
                        continue
                    name = row.get("display_name") or row.get("name")
                    if name:
                        player_id = row.get("id")
                        entry = {"name": str(name)}
                        if player_id:
                            entry["id"] = str(player_id)
                        players.append(entry)
            return players
        players: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if name:
                    players.append({"name": name})
        return players

    default_players = Path("data/normalized/new_players.jsonl")
    if not default_players.exists():
        return []
    return _load_players(str(default_players))


def _load_players_from_nba_file(path: str) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"NBA stats file not found: {path}")
    seen: set[str] = set()
    players: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row.get("player_name")
            if not name:
                continue
            normalized = normalize_player_name(name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entry = {"name": str(name)}
            player_id = row.get("player_id")
            if player_id:
                entry["id"] = str(player_id)
            players.append(entry)
    return players


def _serialize_logs(
    logs: list[PlayerGameLog],
    season: str,
    season_type: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log in logs:
        rows.append(
            {
                "player_id": log.player_id,
                "player_name": log.player_name,
                "game_date": log.game_date.isoformat() if log.game_date else None,
                "season": season,
                "season_type": season_type,
                "stats": log.stats,
            }
        )
    return rows


def _load_sources(path: str | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    sources_path = Path(path)
    if not sources_path.exists():
        return {}
    mapping: dict[str, dict[str, object]] = {}
    with sources_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            player_id = row.get("player_id")
            if player_id:
                mapping[str(player_id)] = row
    return mapping


def _load_sources_by_name(path: str | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    sources_path = Path(path)
    if not sources_path.exists():
        return {}
    mapping: dict[str, dict[str, object]] = {}
    with sources_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row.get("normalized_name") or row.get("player_name")
            if name:
                mapping[normalize_player_name(str(name))] = row
    return mapping


def _run_with_retries(
    fn,
    *,
    retries: int,
    backoff_seconds: float,
) -> list[PlayerGameLog]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            sleep(max(0.0, float(backoff_seconds)) * (2**attempt))
    if last_error is not None:
        raise last_error
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA player game logs.")
    parser.add_argument("--season", required=True, help="Season string, e.g. 2025-26")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Season type, e.g. Regular Season or Playoffs",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "nba_stats", "basketball_reference", "statmuse"),
        default="auto",
        help="Primary data source.",
    )
    parser.add_argument(
        "--players-file",
        default=None,
        help="Optional players file (jsonl from PrizePicks or newline list) for fallback sources.",
    )
    parser.add_argument(
        "--players-from-nba-file",
        default=None,
        help="Use players from an NBA stats JSONL file (recommended for validation).",
    )
    parser.add_argument(
        "--sources-file",
        default="data/player_sources.jsonl",
        help="Optional cached player sources file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of players to fetch (fallback sources).",
    )
    parser.add_argument("--date-from", default=None, help="Date from (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Date to (YYYY-MM-DD)")
    parser.add_argument(
        "--range-days",
        type=int,
        default=3,
        help="When no dates are supplied, use the last N completed days (Central time).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--sleep", type=float, default=0.0, help="Minimum seconds between fallback requests.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Max parallel player fetch workers for fallback sources.",
    )
    parser.add_argument(
        "--fallback-retries",
        type=int,
        default=2,
        help="Retries per fallback source request.",
    )
    parser.add_argument(
        "--fallback-backoff",
        type=float,
        default=1.0,
        help="Base exponential backoff seconds for fallback retries.",
    )
    args = parser.parse_args()

    source = args.source

    date_from = args.date_from
    date_to = args.date_to
    if not date_from and not date_to:
        start, end = central_date_range(args.range_days)
        date_from = start.isoformat()
        date_to = end.isoformat()
    elif date_from and not date_to:
        date_to = central_yesterday().isoformat()
    elif date_to and not date_from:
        start = _parse_date(date_to)
        if start:
            date_from = start.isoformat()
    if source in ("auto", "nba_stats"):
        try:
            payload = fetch_player_gamelogs(
                season=args.season,
                season_type=args.season_type,
                date_from=date_from,
                date_to=date_to,
            )
            rows = extract_result_rows(payload, "PlayerGameLogs")
            normalized = normalize_player_gamelogs(
                rows, season=args.season, season_type=args.season_type
            )
            date_from = date_from or "start"
            date_to = date_to or _today_iso()
            output_path = (
                Path(args.output_dir)
                / f"nba_player_gamelogs_{args.season}_{date_from}_{date_to}.jsonl"
            )
            write_jsonl(normalized, output_path)
            print(f"Wrote {len(normalized)} game logs -> {output_path}")
            if args.sleep:
                sleep(args.sleep)
            return
        except Exception as exc:  # noqa: BLE001
            if source == "nba_stats":
                raise
            print(f"NBA Stats fetch failed ({exc}); falling back to alternate sources.")

    if args.players_from_nba_file:
        players = _load_players_from_nba_file(args.players_from_nba_file)
    else:
        players = _load_players(args.players_file)
    if args.limit:
        players = players[: args.limit]
    if not players:
        raise RuntimeError("No players available for fallback sources.")

    season_end_year = _season_end_year(args.season)
    sources_by_id = _load_sources(args.sources_file)
    sources_by_name = _load_sources_by_name(args.sources_file)
    max_workers = max(1, int(args.max_workers))
    fallback_retries = max(0, int(args.fallback_retries))
    fallback_backoff = max(0.0, float(args.fallback_backoff))

    request_lock = Lock()
    next_allowed: dict[str, float] = {"ts": 0.0}

    def throttle_request() -> None:
        interval = max(0.0, float(args.sleep))
        if interval <= 0:
            return
        while True:
            with request_lock:
                now = monotonic()
                if now >= next_allowed["ts"]:
                    next_allowed["ts"] = now + interval
                    return
                wait_for = next_allowed["ts"] - now
            if wait_for > 0:
                sleep(wait_for)

    def fetch_player_logs(player: dict[str, str]) -> list[PlayerGameLog]:
        player_name = player.get("name")
        if not player_name:
            return []
        player_id = player.get("id")
        normalized = normalize_player_name(player_name)
        source_row = None
        if player_id:
            source_row = sources_by_id.get(player_id)
        if not source_row:
            source_row = sources_by_name.get(normalized)
        if source in ("auto", "basketball_reference"):
            try:
                def fetch_bref() -> list[PlayerGameLog]:
                    slug = None
                    if source_row:
                        slug = source_row.get("basketball_reference_slug")
                    if not slug:
                        throttle_request()
                        slug = search_player_slug(player_name)
                    if not slug:
                        return []
                    throttle_request()
                    html = fetch_player_gamelog_html(slug, season_end_year)
                    rows = parse_gamelog_table(html)
                    return normalize_bref(
                        rows,
                        player_name=player_name,
                        season=args.season,
                        season_type=args.season_type,
                    )

                bref_logs = _run_with_retries(
                    fetch_bref,
                    retries=fallback_retries,
                    backoff_seconds=fallback_backoff,
                )
                if bref_logs:
                    return bref_logs
            except Exception:
                if source == "basketball_reference":
                    raise
        if source in ("auto", "statmuse"):
            try:
                def fetch_statmuse() -> list[PlayerGameLog]:
                    query = None
                    if source_row:
                        query = source_row.get("statmuse_query")
                    if not query:
                        query = build_player_gamelog_query(player_name, season_end_year)
                    throttle_request()
                    html = fetch_ask_html(query)
                    rows = parse_first_stats_table(html)
                    return normalize_statmuse(rows, player_name=player_name)

                return _run_with_retries(
                    fetch_statmuse,
                    retries=fallback_retries,
                    backoff_seconds=fallback_backoff,
                )
            except Exception:
                if source == "statmuse":
                    raise
        return []

    all_logs: list[PlayerGameLog] = []
    if max_workers <= 1 or len(players) <= 1:
        for player in players:
            all_logs.extend(fetch_player_logs(player))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_player_logs, player) for player in players]
            for future in as_completed(futures):
                all_logs.extend(future.result())

    date_from = date_from or "start"
    date_to = date_to or _today_iso()
    start_date = _parse_date(date_from)
    end_date = _parse_date(date_to)
    if start_date and end_date:
        all_logs = _apply_date_range(all_logs, start_date, end_date)
    output_path = (
        Path(args.output_dir) / f"nba_player_gamelogs_{args.season}_{date_from}_{date_to}.jsonl"
    )
    write_jsonl(_serialize_logs(all_logs, args.season, args.season_type), output_path)
    print(f"Wrote {len(all_logs)} game logs -> {output_path}")


if __name__ == "__main__":
    main()
