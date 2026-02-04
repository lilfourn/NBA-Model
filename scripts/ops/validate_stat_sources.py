import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modeling.name_utils import normalize_player_name
from app.modeling.stat_mappings import SPECIAL_STATS, STAT_TYPE_MAP, stat_value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_game_date(row: dict[str, Any]) -> str | None:
    game_date = row.get("game_date")
    if isinstance(game_date, str) and game_date:
        return game_date
    stats = row.get("stats") or {}
    stats_date = stats.get("GAME_DATE")
    if isinstance(stats_date, str) and stats_date:
        if "T" in stats_date:
            return stats_date.split("T")[0]
        return stats_date
    return None


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        name = normalize_player_name(row.get("player_name"))
        date = _extract_game_date(row)
        if not name or not date:
            continue
        key = (name, date)
        if key in indexed:
            duplicates += 1
            continue
        indexed[key] = row
    if duplicates:
        print(f"Warning: skipped {duplicates} duplicate rows during indexing.")
    return indexed


def _filter_by_date(rows: list[dict[str, Any]], start: date | None, end: date | None) -> list[dict[str, Any]]:
    if not start and not end:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        game_date = _extract_game_date(row)
        if not game_date:
            continue
        try:
            parsed = date.fromisoformat(game_date)
        except ValueError:
            continue
        if start and parsed < start:
            continue
        if end and parsed > end:
            continue
        filtered.append(row)
    return filtered


def _compare_values(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NBA Stats vs fallback source game logs.",
    )
    parser.add_argument("--nba-stats-file", required=True, help="NBA Stats JSONL file.")
    parser.add_argument("--fallback-file", required=True, help="Fallback JSONL file.")
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="Max mismatches to show per stat.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output for mismatch details.",
    )
    parser.add_argument("--date-from", default=None, help="Filter from date (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Filter to date (YYYY-MM-DD)")
    args = parser.parse_args()

    nba_rows = _load_jsonl(Path(args.nba_stats_file))
    fallback_rows = _load_jsonl(Path(args.fallback_file))

    start_date = date.fromisoformat(args.date_from) if args.date_from else None
    end_date = date.fromisoformat(args.date_to) if args.date_to else None
    nba_rows = _filter_by_date(nba_rows, start_date, end_date)
    fallback_rows = _filter_by_date(fallback_rows, start_date, end_date)

    nba_index = _index_rows(nba_rows)
    fallback_index = _index_rows(fallback_rows)

    shared_keys = sorted(set(nba_index.keys()) & set(fallback_index.keys()))
    if not shared_keys:
        print("No overlapping player/date rows found between sources.")
        nba_dates = sorted(
            {_extract_game_date(row) for row in nba_rows if _extract_game_date(row)}
        )
        fallback_dates = sorted(
            {_extract_game_date(row) for row in fallback_rows if _extract_game_date(row)}
        )
        if nba_dates:
            print(f"NBA Stats date range: {nba_dates[0]} -> {nba_dates[-1]}")
        if fallback_dates:
            print(f"Fallback date range: {fallback_dates[0]} -> {fallback_dates[-1]}")
        return

    stat_types = list(STAT_TYPE_MAP.keys()) + list(SPECIAL_STATS.keys())

    summary: dict[str, dict[str, float | int]] = {}
    mismatch_details: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for stat_type in stat_types:
        total = 0
        matches = 0
        for key in shared_keys:
            nba_stats = nba_index[key].get("stats") or {}
            fallback_stats = fallback_index[key].get("stats") or {}
            nba_value = stat_value(stat_type, nba_stats)
            fallback_value = stat_value(stat_type, fallback_stats)
            if nba_value is None or fallback_value is None:
                continue
            total += 1
            if _compare_values(nba_value, fallback_value):
                matches += 1
            else:
                if len(mismatch_details[stat_type]) < args.max_mismatches:
                    mismatch_details[stat_type].append(
                        {
                            "player": key[0],
                            "game_date": key[1],
                            "stat_type": stat_type,
                            "nba_value": nba_value,
                            "fallback_value": fallback_value,
                        }
                    )
        accuracy = (matches / total) if total else 0.0
        summary[stat_type] = {
            "compared": total,
            "matches": matches,
            "accuracy": round(accuracy, 4),
        }

    print("Per-Stat Accuracy")
    print("=" * 80)
    for stat_type, metrics in sorted(
        summary.items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        compared = metrics["compared"]
        matches = metrics["matches"]
        accuracy = metrics["accuracy"]
        print(f"{stat_type:20s} | compared {compared:4d} | matches {matches:4d} | acc {accuracy:.2%}")

    print("\nMismatches (sample)")
    print("=" * 80)
    for stat_type, entries in mismatch_details.items():
        if not entries:
            continue
        print(f"{stat_type} ({len(entries)})")
        for entry in entries:
            print(
                f"- {entry['player']} {entry['game_date']}: "
                f"NBA={entry['nba_value']} vs Fallback={entry['fallback_value']}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps({"summary": summary, "mismatches": mismatch_details}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote mismatch report -> {output_path}")


if __name__ == "__main__":
    main()
