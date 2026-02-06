from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.db import schema
from app.db.coerce import (
    json_safe as _json_safe,
    normalize_id as _normalize_id,
    parse_datetime as _parse_datetime,
    parse_decimal as _parse_decimal,
)
from app.ml.stat_mappings import stat_value_from_row
from app.utils.names import normalize_name


def _parse_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return UUID(text_value)
    except ValueError:
        return None


def _normalize_pick(value: Any, *, prob_over: Decimal | None = None) -> str:
    if value is not None:
        text_value = str(value).strip().upper()
        if text_value in {"OVER", "UNDER"}:
            return text_value
    if prob_over is not None:
        return "OVER" if float(prob_over) >= 0.5 else "UNDER"
    return "OVER"


def _normalize_team_abbreviation(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip().upper()
    return text_value or None


def _load_name_overrides() -> dict[str, str]:
    path = Path(settings.player_name_overrides_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for source_name, target_name in data.items():
        source_key = normalize_name(source_name)
        target_key = normalize_name(target_name)
        if source_key and target_key:
            out[source_key] = target_key
    return out


def append_prediction_rows(engine: Engine, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    now = datetime.now(timezone.utc)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        projection_id = _normalize_id(row.get("projection_id"))
        if not projection_id:
            continue

        prob_over = _parse_decimal(row.get("prob_over"))
        if prob_over is None:
            prob_over = _parse_decimal(row.get("p_final"))

        decision_time = _parse_datetime(row.get("decision_time"))
        if decision_time is None:
            decision_time = _parse_datetime(row.get("created_at"))
        if decision_time is None:
            decision_time = now

        snapshot_id = _parse_uuid(row.get("snapshot_id"))
        prepared.append(
            {
                "id": uuid4(),
                "snapshot_id": snapshot_id,
                "projection_id": projection_id,
                "model_version": str(row.get("model_version") or "ensemble"),
                "decision_time": decision_time,
                "player_id": _normalize_id(row.get("player_id")),
                "game_id": _normalize_id(row.get("game_id")),
                "stat_type": str(row.get("stat_type") or "") or None,
                "line_score": _parse_decimal(row.get("line_score")),
                "pick": _normalize_pick(row.get("pick"), prob_over=prob_over),
                "prob_over": prob_over,
                "p_raw": _parse_decimal(row.get("p_raw")),
                "confidence": _parse_decimal(row.get("confidence")),
                "p_forecast_cal": _parse_decimal(row.get("p_forecast_cal")),
                "p_nn": _parse_decimal(row.get("p_nn")),
                "p_tabdl": _parse_decimal(row.get("p_tabdl")),
                "p_lr": _parse_decimal(row.get("p_lr")),
                "p_xgb": _parse_decimal(row.get("p_xgb")),
                "p_lgbm": _parse_decimal(row.get("p_lgbm")),
                "rank_score": _parse_decimal(row.get("rank_score")),
                "n_eff": _parse_decimal(row.get("n_eff")),
                "mean": _parse_decimal(
                    row.get("mu_hat")
                    if row.get("mu_hat") is not None
                    else row.get("mean")
                ),
                "std": _parse_decimal(
                    row.get("sigma_hat")
                    if row.get("sigma_hat") is not None
                    else row.get("std")
                ),
                "actual_value": None,
                "over_label": None,
                "outcome": None,
                "is_correct": None,
                "resolved_at": None,
                "created_at": now,
                "details": _json_safe(row),
            }
        )

    if not prepared:
        return 0

    batch_size = 100
    inserted = 0
    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        try:
            with engine.begin() as conn:
                conn.execute(pg_insert(schema.projection_predictions).values(batch))
            inserted += len(batch)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[prediction_logs] batch {i//batch_size} failed ({len(batch)} rows): {exc}"
            )
            for row in batch:
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            pg_insert(schema.projection_predictions).values([row])
                        )
                    inserted += 1
                except Exception as row_exc:  # noqa: BLE001
                    print(
                        f"[prediction_logs] row {row.get('projection_id')} failed: {row_exc}"
                    )
    return inserted


def _resolve_over_under_outcome(
    *, line_score: float, actual_value: float
) -> tuple[int, str]:
    if actual_value > line_score:
        return 1, "over"
    if actual_value < line_score:
        return 0, "under"
    return 0, "push"


def resolve_prediction_outcomes(
    engine: Engine,
    *,
    days_back: int = 21,
    decision_lag_hours: int = 3,
    limit: int = 10000,
) -> dict[str, int]:
    now = datetime.now(timezone.utc)
    start_at = now - timedelta(days=max(1, int(days_back)))
    decision_cutoff = now - timedelta(hours=max(0, int(decision_lag_hours)))

    candidates = pd.read_sql(
        text(
            """
            select
                id,
                pick,
                player_id,
                game_id,
                stat_type,
                line_score,
                coalesce(decision_time, created_at) as decision_time
            from projection_predictions
            where actual_value is null
              and player_id is not null
              and game_id is not null
              and stat_type is not null
              and line_score is not null
              and coalesce(decision_time, created_at) >= :start_at
              and coalesce(decision_time, created_at) <= :decision_cutoff
            order by coalesce(decision_time, created_at) asc
            limit :limit
            """
        ),
        engine,
        params={
            "start_at": start_at,
            "decision_cutoff": decision_cutoff,
            "limit": int(limit),
        },
    )
    if candidates.empty:
        return {
            "candidates": 0,
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    candidates = candidates.copy()
    candidates["player_id"] = candidates["player_id"].apply(_normalize_id)
    candidates["game_id"] = candidates["game_id"].apply(_normalize_id)
    candidates = candidates.dropna(
        subset=["player_id", "game_id", "stat_type", "line_score"]
    )
    if candidates.empty:
        return {
            "candidates": 0,
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    player_ids = sorted(
        {str(v) for v in candidates["player_id"].dropna().unique().tolist()}
    )
    game_ids = sorted(
        {str(v) for v in candidates["game_id"].dropna().unique().tolist()}
    )

    players = pd.read_sql(
        text(
            "select id as player_id, name_key, display_name from players where id = any(:ids)"
        ),
        engine,
        params={"ids": player_ids},
    )
    games = pd.read_sql(
        text(
            """
            select
                g.id as game_id,
                (g.start_time at time zone 'America/New_York')::date as game_date,
                ht.abbreviation as home_team_abbreviation,
                at.abbreviation as away_team_abbreviation
            from games g
            left join teams ht on ht.id = g.home_team_id
            left join teams at on at.id = g.away_team_id
            where g.id = any(:ids)
            """
        ),
        engine,
        params={"ids": game_ids},
    )
    if players.empty or games.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    # Use unified normalize_name which now includes NFKD + suffix stripping + overrides.
    players = players.copy()
    players["normalized_name_key"] = players.apply(
        lambda r: normalize_name(r.get("display_name") or r.get("name_key")), axis=1
    )
    keys = sorted(
        {str(v) for v in players["normalized_name_key"].dropna().unique().tolist()}
    )
    if not keys:
        print(f"[resolve] 0/{len(candidates)} candidates: no player name_keys resolved")
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "no_name_key": int(len(candidates)),
        }

    nba_players = pd.read_sql(
        text(
            "select id as nba_player_id, name_key from nba_players where name_key = any(:keys)"
        ),
        engine,
        params={"keys": keys},
    )
    unmatched_names = (
        set(keys) - set(nba_players["name_key"].tolist())
        if not nba_players.empty
        else set(keys)
    )
    if unmatched_names:
        print(
            f"[resolve] {len(unmatched_names)} player name_keys not found in nba_players: {sorted(unmatched_names)[:10]}"
        )
    if nba_players.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "no_nba_player_match": int(len(candidates)),
        }

    mapped = players.merge(
        nba_players,
        left_on="normalized_name_key",
        right_on="name_key",
        how="left",
    )[["player_id", "nba_player_id"]]

    # Ensure consistent str types for join columns to prevent silent dtype mismatches.
    mapped["player_id"] = mapped["player_id"].astype(str)
    candidates["player_id"] = candidates["player_id"].astype(str)
    candidates["game_id"] = candidates["game_id"].astype(str)
    games["game_id"] = games["game_id"].astype(str)
    games["home_team_abbreviation"] = games["home_team_abbreviation"].apply(
        _normalize_team_abbreviation
    )
    games["away_team_abbreviation"] = games["away_team_abbreviation"].apply(
        _normalize_team_abbreviation
    )

    merged = candidates.merge(mapped, on="player_id", how="left").merge(
        games, on="game_id", how="left"
    )
    no_nba_id = int(merged["nba_player_id"].isna().sum())
    no_game_date = int(merged["game_date"].isna().sum())
    merged = merged.dropna(subset=["nba_player_id", "game_date"])
    if merged.empty:
        print(
            f"[resolve] 0/{len(candidates)} matched: {no_nba_id} missing nba_player_id, {no_game_date} missing game_date"
        )
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "no_nba_player_match": no_nba_id,
            "no_game_date": no_game_date,
        }

    date_from = merged["game_date"].min()
    date_to = merged["game_date"].max()

    # Map PP game -> NBA game by date + matchup abbreviations so we can detect
    # "player has no boxscore row but game data exists" and void those props.
    nba_games = pd.read_sql(
        text(
            """
            select
                id as nba_game_id,
                game_date,
                upper(home_team_abbreviation) as home_team_abbreviation,
                upper(away_team_abbreviation) as away_team_abbreviation
            from nba_games
            where game_date >= :date_from
              and game_date <= :date_to
            """
        ),
        engine,
        params={"date_from": date_from, "date_to": date_to},
    )
    if nba_games.empty:
        merged["nba_game_id"] = None
    else:
        nba_games = nba_games.drop_duplicates(
            subset=["game_date", "home_team_abbreviation", "away_team_abbreviation"]
        )
        merged = merged.merge(
            nba_games,
            on=["game_date", "home_team_abbreviation", "away_team_abbreviation"],
            how="left",
        )

    nba_game_ids = sorted(
        {str(v) for v in merged["nba_game_id"].dropna().unique().tolist()}
    )
    if nba_game_ids:
        game_boxscore_counts = pd.read_sql(
            text(
                """
                select
                    game_id as nba_game_id,
                    count(*)::int as nba_game_boxscore_rows
                from nba_player_game_stats
                where game_id = any(:game_ids)
                group by game_id
                """
            ),
            engine,
            params={"game_ids": nba_game_ids},
        )
    else:
        game_boxscore_counts = pd.DataFrame(
            columns=["nba_game_id", "nba_game_boxscore_rows"]
        )
    if game_boxscore_counts.empty:
        merged["nba_game_boxscore_rows"] = 0
    else:
        game_boxscore_counts["nba_game_id"] = game_boxscore_counts[
            "nba_game_id"
        ].astype(str)
        merged = merged.merge(game_boxscore_counts, on="nba_game_id", how="left")
        merged["nba_game_boxscore_rows"] = pd.to_numeric(
            merged["nba_game_boxscore_rows"], errors="coerce"
        ).fillna(0)

    nba_ids = sorted({str(v) for v in merged["nba_player_id"].unique().tolist()})
    stats = pd.read_sql(
        text(
            """
            select
                s.player_id as nba_player_id,
                ng.game_date as game_date,
                s.points,
                s.rebounds,
                s.assists,
                s.steals,
                s.blocks,
                s.turnovers,
                s.fg3m,
                s.fg3a,
                s.fgm,
                s.fga,
                s.ftm,
                s.fta,
                s.minutes,
                coalesce(nullif(s.stats_json->>'OREB', '')::numeric, 0) as oreb,
                coalesce(nullif(s.stats_json->>'DREB', '')::numeric, 0) as dreb,
                coalesce(nullif(s.stats_json->>'PF', '')::numeric, 0) as pf,
                coalesce(nullif(s.stats_json->>'DUNKS', '')::numeric, 0) as dunks
            from nba_player_game_stats s
            join nba_games ng on ng.id = s.game_id
            where s.player_id = any(:player_ids)
              and ng.game_date >= :date_from
              and ng.game_date <= :date_to
            """
        ),
        engine,
        params={"player_ids": nba_ids, "date_from": date_from, "date_to": date_to},
    )
    # Ensure nba_player_id is str for merge consistency.
    if not stats.empty:
        stats["nba_player_id"] = stats["nba_player_id"].astype(str)

    # Ensure consistent str type for merge key.
    merged["nba_player_id"] = merged["nba_player_id"].astype(str)
    resolved = merged.merge(
        stats, on=["nba_player_id", "game_date"], how="left", indicator="_stats_merge"
    )
    matched_boxscores = int((resolved["_stats_merge"] == "both").sum())
    resolved["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in resolved.itertuples(index=False)
    ]

    # If game boxscore data exists for the matched NBA game but this player has
    # no row, treat as no-appearance and grade as push (void) instead of leaving
    # it unresolved forever.
    forced_push_mask = (
        resolved["actual_value"].isna()
        & resolved["line_score"].notna()
        & resolved["_stats_merge"].eq("left_only")
        & pd.to_numeric(resolved["nba_game_boxscore_rows"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    resolved["forced_push_no_boxscore"] = forced_push_mask
    resolved.loc[forced_push_mask, "actual_value"] = resolved.loc[
        forced_push_mask, "line_score"
    ].astype(float)
    forced_push_no_boxscore = int(forced_push_mask.sum())

    resolved = resolved.dropna(subset=["actual_value", "line_score"])
    if resolved.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": matched_boxscores,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "forced_push_no_boxscore": forced_push_no_boxscore,
        }

    updates: list[dict[str, Any]] = []
    wins = 0
    losses = 0
    pushes = 0
    for row in resolved.itertuples(index=False):
        line_score = float(getattr(row, "line_score"))
        forced_push = bool(getattr(row, "forced_push_no_boxscore", False))
        actual_value = float(getattr(row, "actual_value"))
        if forced_push:
            over_label: int | None = None
            outcome = "push"
        else:
            over_label_raw, outcome = _resolve_over_under_outcome(
                line_score=line_score, actual_value=actual_value
            )
            over_label = None if outcome == "push" else int(over_label_raw)
        pick = str(getattr(row, "pick") or "").strip().lower()
        is_correct: bool | None
        if outcome == "push":
            is_correct = None
            pushes += 1
        elif pick in {"over", "under"}:
            is_correct = pick == outcome
            if is_correct:
                wins += 1
            else:
                losses += 1
        else:
            is_correct = None

        updates.append(
            {
                "id": str(getattr(row, "id")),
                "actual_value": Decimal(str(actual_value)),
                "over_label": over_label,
                "outcome": outcome,
                "is_correct": is_correct,
                "resolved_at": now,
            }
        )

    if not updates:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": matched_boxscores,
            "updated": 0,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "forced_push_no_boxscore": forced_push_no_boxscore,
        }

    deduped: dict[str, dict[str, Any]] = {row["id"]: row for row in updates}
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                update projection_predictions
                set
                    actual_value = :actual_value,
                    over_label = :over_label,
                    outcome = :outcome,
                    is_correct = :is_correct,
                    resolved_at = :resolved_at
                where id = :id
                  and actual_value is null
                """
            ),
            list(deduped.values()),
        )
        updated_rows = int(result.rowcount or 0)

    return {
        "candidates": int(len(candidates)),
        "matched_boxscores": matched_boxscores,
        "updated": updated_rows,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "forced_push_no_boxscore": forced_push_no_boxscore,
    }
