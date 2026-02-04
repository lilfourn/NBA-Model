import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.basketball_reference import search_player_slug
from app.clients.statmuse import build_ask_url, build_player_gamelog_query
from app.modeling.name_utils import normalize_player_name


DEFAULT_OUTPUT = Path("data/player_sources.jsonl")


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


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


def _load_projection_players(normalized_dir: Path) -> list[dict[str, object]]:
    projections_path = normalized_dir / "projections.jsonl"
    players_path = normalized_dir / "new_players.jsonl"
    projections = _load_jsonl(projections_path)
    players = _load_jsonl(players_path)
    players_by_id = {str(row.get("id")): row for row in players}

    projection_player_ids = {
        str(row.get("new_player_id"))
        for row in projections
        if row.get("new_player_id")
        and str(row.get("odds_type") or "standard").strip().lower() == "standard"
        and str(row.get("event_type") or "").strip().lower() != "combo"
    }
    output: list[dict[str, object]] = []
    for player_id in projection_player_ids:
        player = players_by_id.get(player_id)
        if not player:
            continue
        if player.get("combo"):
            continue
        name = player.get("display_name") or player.get("name")
        if not name:
            continue
        output.append(
            {
                "player_id": player_id,
                "player_name": name,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-player source URLs for stats fetching.",
    )
    parser.add_argument("--season", required=True, help="Season string, e.g. 2025-26")
    parser.add_argument(
        "--normalized-dir",
        default="data/normalized",
        help="Directory containing normalized PrizePicks tables.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL path for player sources.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit players.")
    parser.add_argument("--refresh", action="store_true", help="Recompute URLs.")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only re-query players missing Basketball-Reference slugs.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    args = parser.parse_args()

    normalized_dir = Path(args.normalized_dir)
    output_path = Path(args.output)
    season_end_year = _season_end_year(args.season)

    existing_rows = _load_jsonl(output_path)
    existing_by_player = {row.get("player_id"): row for row in existing_rows}

    players = _load_projection_players(normalized_dir)
    if args.limit:
        players = players[: args.limit]

    updated: list[dict[str, object]] = []
    for player in players:
        player_id = player["player_id"]
        player_name = player["player_name"]
        normalized_name = normalize_player_name(str(player_name))
        existing = existing_by_player.get(player_id)

        if existing and not args.refresh:
            if existing.get("season") == args.season:
                if args.missing_only:
                    if existing.get("basketball_reference_slug"):
                        updated.append(existing)
                        continue
                else:
                    updated.append(existing)
                    continue

        bref_slug = None
        try:
            bref_slug = search_player_slug(str(player_name))
        except Exception:
            bref_slug = None
        bref_url = None
        if bref_slug:
            bref_url = f"https://www.basketball-reference.com{bref_slug.replace('.html','')}/gamelog/{season_end_year}"

        statmuse_query = build_player_gamelog_query(str(player_name), season_end_year)
        statmuse_url = build_ask_url(statmuse_query)

        updated.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "normalized_name": normalized_name,
                "season": args.season,
                "season_end_year": season_end_year,
                "basketball_reference_slug": bref_slug,
                "basketball_reference_gamelog_url": bref_url,
                "statmuse_query": statmuse_query,
                "statmuse_url": statmuse_url,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        if args.sleep:
            sleep(args.sleep)

    _write_jsonl(updated, output_path)
    print(f"Wrote {len(updated)} player sources -> {output_path}")


if __name__ == "__main__":
    main()
