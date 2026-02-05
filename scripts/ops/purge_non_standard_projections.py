import argparse
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _print_counts(conn) -> None:
    proj_counts = conn.execute(
        text(
            """
            select
                case
                    when coalesce(odds_type, 0) = 0 then 'standard'
                    when odds_type = 1 then 'goblin'
                    when odds_type = 2 then 'demon'
                    else lower(coalesce(attributes->>'odds_type', 'unknown'))
                end as odds_type,
                count(*)
            from projections
            group by 1
            order by 2 desc
            """
        )
    ).all()
    pf_counts = conn.execute(
        text(
            """
            select
                case
                    when coalesce(p.odds_type, 0) = 0 then 'standard'
                    when p.odds_type = 1 then 'goblin'
                    when p.odds_type = 2 then 'demon'
                    else lower(coalesce(p.attributes->>'odds_type', 'unknown'))
                end as odds_type,
                count(*)
            from projection_features pf
            join projections p
              on p.snapshot_id = pf.snapshot_id
             and p.projection_id = pf.projection_id
            group by 1
            order by 2 desc
            """
        )
    ).all()
    print({"projections_by_odds_type": [(r[0], int(r[1])) for r in proj_counts]})
    print({"projection_features_by_odds_type": [(r[0], int(r[1])) for r in pf_counts]})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Purge non-standard PrizePicks projections (goblin/demon/etc) from Postgres.",
    )
    ap.add_argument("--database-url", default=None)
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows (default: dry-run).",
    )
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    with engine.begin() as conn:
        _print_counts(conn)
        if not args.execute:
            print("Dry-run only. Re-run with --execute to delete non-standard rows.")
            return

        deleted_pf = conn.execute(
            text(
                """
                delete from projection_features pf
                using projections p
                where p.snapshot_id = pf.snapshot_id
                  and p.projection_id = pf.projection_id
                  and coalesce(
                        p.odds_type,
                        case lower(coalesce(p.attributes->>'odds_type', 'standard'))
                            when 'goblin' then 1
                            when 'demon' then 2
                            else 0
                        end
                    ) <> 0
                """
            )
        ).rowcount

        deleted_proj = conn.execute(
            text(
                """
                delete from projections p
                where coalesce(
                    p.odds_type,
                    case lower(coalesce(p.attributes->>'odds_type', 'standard'))
                        when 'goblin' then 1
                        when 'demon' then 2
                        else 0
                    end
                ) <> 0
                """
            )
        ).rowcount

        # Hardening: after purge we only retain standard rows, so store odds_type=0 explicitly.
        conn.execute(text("update projections set odds_type = 0 where odds_type is null"))
        conn.execute(text("update projection_features set odds_type = 0 where odds_type is null"))

        print(
            {
                "deleted_projection_features": int(deleted_pf or 0),
                "deleted_projections": int(deleted_proj or 0),
            }
        )

        _print_counts(conn)


if __name__ == "__main__":
    main()
