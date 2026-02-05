import argparse
import sys
import time
from datetime import date
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clients.nba_stats import fetch_shot_chart_detail  # noqa: E402
from app.db.engine import get_engine  # noqa: E402
from app.ml.feature_engineering import clear_gamelog_frame_cache  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _season_for_game_date(game_date: date) -> str:
    start_year = game_date.year if game_date.month >= 10 else (game_date.year - 1)
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def _count_dunks_made(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("ACTION_TYPE") or "")
        if "dunk" not in action.lower():
            continue
        made = row.get("SHOT_MADE_FLAG")
        try:
            made_int = int(made)
        except (TypeError, ValueError):
            made_int = 0
        if made_int != 1:
            continue
        pid = row.get("PLAYER_ID")
        if pid is None:
            continue
        key = str(pid)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill DUNKS into nba_player_game_stats.stats_json using NBA shotchartdetail.",
    )
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--date-from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--date-to", required=True, help="YYYY-MM-DD")
    ap.add_argument("--season-type", default="Regular Season")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--force", action="store_true", help="Recompute even if DUNKS already present.")
    ap.add_argument("--execute", action="store_true", help="Actually update the DB (default: dry-run).")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)

    with engine.connect() as conn:
        games = conn.execute(
            text(
                """
                select ng.id, ng.game_date
                from nba_games ng
                where ng.game_date >= :date_from
                  and ng.game_date <= :date_to
                  and exists (
                      select 1 from nba_player_game_stats s where s.game_id = ng.id
                  )
                  and (
                      :force = true
                      or not exists (
                          select 1
                          from nba_player_game_stats s2
                          where s2.game_id = ng.id
                            and (s2.stats_json ? 'DUNKS')
                      )
                  )
                order by ng.game_date asc, ng.id asc
                """
            ),
            {"date_from": date_from, "date_to": date_to, "force": bool(args.force)},
        ).all()

    if args.max_games is not None:
        games = games[: int(args.max_games)]

    print({"games": len(games), "execute": bool(args.execute), "force": bool(args.force)})
    if not games:
        return

    updated_games = 0
    updated_rows = 0

    for idx, (game_id, game_date) in enumerate(games, start=1):
        season = _season_for_game_date(game_date)
        rows = fetch_shot_chart_detail(
            game_id=str(game_id),
            season=season,
            season_type=str(args.season_type),
            player_id="0",
            team_id="0",
        )
        dunk_counts = _count_dunks_made(rows)

        if not args.execute:
            print(
                {
                    "idx": idx,
                    "game_id": str(game_id),
                    "game_date": str(game_date),
                    "season": season,
                    "players_with_dunks": len(dunk_counts),
                    "total_dunks": int(sum(dunk_counts.values())),
                }
            )
            continue

        with engine.begin() as conn:
            # Mark all players in the game with DUNKS=0, then overwrite non-zero counts.
            result = conn.execute(
                text(
                    """
                    update nba_player_game_stats
                    set stats_json = jsonb_set(coalesce(stats_json, '{}'::jsonb), '{DUNKS}', '0'::jsonb, true)
                    where game_id = :game_id
                    """
                ),
                {"game_id": str(game_id)},
            )
            updated_rows += int(result.rowcount or 0)

            if dunk_counts:
                value_rows = []
                params: dict[str, object] = {"game_id": str(game_id)}
                for idx_value, (player_id, dunks) in enumerate(dunk_counts.items()):
                    player_key = f"player_id_{idx_value}"
                    dunks_key = f"dunks_{idx_value}"
                    value_rows.append(f"(:{player_key}, :{dunks_key})")
                    params[player_key] = str(player_id)
                    params[dunks_key] = int(dunks)
                conn.execute(
                    text(
                        f"""
                        with counts(player_id, dunks) as (
                            values {", ".join(value_rows)}
                        )
                        update nba_player_game_stats s
                        set stats_json = jsonb_set(
                            coalesce(s.stats_json, '{{}}'::jsonb),
                            '{{DUNKS}}',
                            to_jsonb(cast(c.dunks as int)),
                            true
                        )
                        from counts c
                        where s.game_id = :game_id
                          and s.player_id = c.player_id
                        """
                    ),
                    params,
                )

        updated_games += 1
        print(
            {
                "idx": idx,
                "game_id": str(game_id),
                "game_date": str(game_date),
                "season": season,
                "players_with_dunks": len(dunk_counts),
                "total_dunks": int(sum(dunk_counts.values())),
            }
        )

        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    if args.execute and updated_games > 0:
        clear_gamelog_frame_cache(str(engine.url))

    print({"updated_games": updated_games, "updated_rows": updated_rows})


if __name__ == "__main__":
    main()
