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
from app.ml.stat_mappings import stat_value_from_row
from app.utils.names import normalize_name


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return None
        try:
            parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


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


def _normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        head = text_value[:-2]
        if head.isdigit():
            return head
    return text_value


def _normalize_pick(value: Any, *, prob_over: Decimal | None = None) -> str:
    if value is not None:
        text_value = str(value).strip().upper()
        if text_value in {"OVER", "UNDER"}:
            return text_value
    if prob_over is not None:
        return "OVER" if float(prob_over) >= 0.5 else "UNDER"
    return "OVER"


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
                "confidence": _parse_decimal(row.get("confidence")),
                "p_forecast_cal": _parse_decimal(row.get("p_forecast_cal")),
                "p_nn": _parse_decimal(row.get("p_nn")),
                "p_lr": _parse_decimal(row.get("p_lr")),
                "p_xgb": _parse_decimal(row.get("p_xgb")),
                "p_lgbm": _parse_decimal(row.get("p_lgbm")),
                "rank_score": _parse_decimal(row.get("rank_score")),
                "n_eff": _parse_decimal(row.get("n_eff")),
                "mean": _parse_decimal(row.get("mu_hat") if row.get("mu_hat") is not None else row.get("mean")),
                "std": _parse_decimal(row.get("sigma_hat") if row.get("sigma_hat") is not None else row.get("std")),
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
            print(f"[prediction_logs] batch {i//batch_size} failed ({len(batch)} rows): {exc}")
            for row in batch:
                try:
                    with engine.begin() as conn:
                        conn.execute(pg_insert(schema.projection_predictions).values([row]))
                    inserted += 1
                except Exception as row_exc:  # noqa: BLE001
                    print(f"[prediction_logs] row {row.get('projection_id')} failed: {row_exc}")
    return inserted


def _resolve_over_under_outcome(*, line_score: float, actual_value: float) -> tuple[int, str]:
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
    candidates = candidates.dropna(subset=["player_id", "game_id", "stat_type", "line_score"])
    if candidates.empty:
        return {
            "candidates": 0,
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    player_ids = sorted({str(v) for v in candidates["player_id"].dropna().unique().tolist()})
    game_ids = sorted({str(v) for v in candidates["game_id"].dropna().unique().tolist()})

    players = pd.read_sql(
        text("select id as player_id, name_key, display_name from players where id = any(:ids)"),
        engine,
        params={"ids": player_ids},
    )
    games = pd.read_sql(
        text(
            """
            select
                id as game_id,
                (start_time at time zone 'America/New_York')::date as game_date
            from games
            where id = any(:ids)
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

    name_overrides = _load_name_overrides()

    def to_name_key(row: pd.Series) -> str | None:
        raw = row.get("display_name") or row.get("name_key")
        key = normalize_name(raw)
        if not key:
            return None
        return name_overrides.get(key, key)

    players = players.copy()
    players["normalized_name_key"] = [to_name_key(row) for row in players.to_dict(orient="records")]
    keys = sorted({str(v) for v in players["normalized_name_key"].dropna().unique().tolist()})
    if not keys:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players where name_key = any(:keys)"),
        engine,
        params={"keys": keys},
    )
    if nba_players.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    mapped = players.merge(
        nba_players,
        left_on="normalized_name_key",
        right_on="name_key",
        how="left",
    )[["player_id", "nba_player_id"]]

    merged = candidates.merge(mapped, on="player_id", how="left").merge(games, on="game_id", how="left")
    merged = merged.dropna(subset=["nba_player_id", "game_date"])
    if merged.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    date_from = merged["game_date"].min()
    date_to = merged["game_date"].max()
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
                s.fta
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
    if stats.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    resolved = merged.merge(stats, on=["nba_player_id", "game_date"], how="left")
    resolved["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in resolved.itertuples(index=False)
    ]
    resolved = resolved.dropna(subset=["actual_value", "line_score"])
    if resolved.empty:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": 0,
            "updated": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        }

    updates: list[dict[str, Any]] = []
    wins = 0
    losses = 0
    pushes = 0
    for row in resolved.itertuples(index=False):
        line_score = float(getattr(row, "line_score"))
        actual_value = float(getattr(row, "actual_value"))
        over_label, outcome = _resolve_over_under_outcome(line_score=line_score, actual_value=actual_value)
        pick = str(getattr(row, "pick") or "").strip().lower()
        is_correct: bool | None
        if outcome == "push":
            is_correct = None
            pushes += 1
        elif pick in {"over", "under"}:
            is_correct = (pick == outcome)
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
                "over_label": int(over_label),
                "outcome": outcome,
                "is_correct": is_correct,
                "resolved_at": now,
            }
        )

    if not updates:
        return {
            "candidates": int(len(candidates)),
            "matched_boxscores": int(len(resolved)),
            "updated": 0,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
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
        "matched_boxscores": int(len(resolved)),
        "updated": updated_rows,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
    }
