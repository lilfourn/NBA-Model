from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.utils.names import normalize_name
from app.ml.feature_engineering import (
    build_league_means_timeline,
    build_logs_by_player,
    compute_history_features,
    estimate_team_travel_context,
    league_mean_before_cutoff,
    load_gamelogs_frame,
    prepare_gamelogs,
    slice_player_logs_before_cutoff,
)
from app.ml.opponent_features import (
    build_opponent_defensive_averages,
    build_opponent_defensive_ranks,
    compute_opponent_features,
    compute_team_pace,
    _load_team_game_stats,
)
from app.ml.feature_engineering import compute_player_usage
from app.ml.stat_mappings import STAT_COLUMNS, normalize_stat_type, stat_value_from_row

COMBO_SPLIT = re.compile(r"\s*\+\s*")


def compute_actual_value(row: pd.Series) -> float | None:
    return stat_value_from_row(row.get("stat_type"), row)


def _load_team_abbrev_overrides() -> dict[str, str]:
    path = Path(settings.team_abbrev_overrides_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return {str(k).upper(): str(v).upper() for k, v in data.items() if k and v}
    if isinstance(data, list):
        overrides: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or item.get("source_abbr")
            target = (
                item.get("normalized")
                or item.get("normalized_abbr")
                or item.get("target")
            )
            if source and target:
                overrides[str(source).upper()] = str(target).upper()
        return overrides
    return {}


def _load_name_overrides() -> dict[str, str]:
    path = Path(settings.player_name_overrides_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        overrides: dict[str, str] = {}
        for raw_name, normalized in data.items():
            key = normalize_name(raw_name)
            value = normalize_name(normalized)
            if key and value:
                overrides[key] = value
        return overrides
    return {}


def _build_team_abbrev_cte(overrides: dict[str, str]) -> tuple[str, dict[str, Any]]:
    if not overrides:
        return (
            "team_abbrev_map as (select null::text as source_abbr, null::text as normalized_abbr where false)",
            {},
        )

    params: dict[str, Any] = {}
    values_sql: list[str] = []
    for idx, (source, normalized) in enumerate(sorted(overrides.items())):
        src_key = f"src_{idx}"
        dst_key = f"dst_{idx}"
        params[src_key] = source
        params[dst_key] = normalized
        values_sql.append(f"(:{src_key}, :{dst_key})")

    cte = (
        "team_abbrev_map as (select * from (values "
        + ", ".join(values_sql)
        + ") as m(source_abbr, normalized_abbr))"
    )
    return cte, params


def load_training_data(engine: Engine) -> pd.DataFrame:
    overrides = _load_team_abbrev_overrides()
    cte, params = _build_team_abbrev_cte(overrides)
    name_overrides = _load_name_overrides()
    query = text(
        f"""
        with {cte}
        select
            pf.snapshot_id,
            pf.projection_id,
            pf.player_id,
            pf.game_id,
            pf.line_score,
            pf.line_score_prev,
            pf.line_score_delta,
            pf.line_movement,
            pf.stat_type,
            pf.projection_type,
            pf.odds_type,
            pf.trending_count,
            pf.is_promo,
            pf.is_live,
            pf.in_game,
            pf.today,
            pf.minutes_to_start,
            pf.fetched_at,
            pl.name as player_name,
            pl.combo as is_combo,
            pl.name_key as prizepicks_name_key,
            ng.id as nba_game_id
        from projection_features pf
        join projections p
            on p.snapshot_id = pf.snapshot_id
            and p.projection_id = pf.projection_id
        join players pl on pl.id = pf.player_id
        join games g on g.id = pf.game_id
        left join teams ht on ht.id = g.home_team_id
        left join teams at on at.id = g.away_team_id
        left join team_abbrev_map htm on htm.source_abbr = upper(ht.abbreviation)
        left join team_abbrev_map atm on atm.source_abbr = upper(at.abbreviation)
        join nba_games ng
            on ng.game_date = (g.start_time at time zone 'America/New_York')::date
            and ng.home_team_abbreviation = upper(coalesce(htm.normalized_abbr, ht.abbreviation))
            and ng.away_team_abbreviation = upper(coalesce(atm.normalized_abbr, at.abbreviation))
        where coalesce(p.odds_type, 0) = 0
          and lower(coalesce(p.event_type, p.attributes->>'event_type', '')) <> 'combo'
        """
    )
    base_df = pd.read_sql(query, engine, params=params)
    if base_df.empty:
        return base_df

    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players"),
        engine,
    )
    if nba_players.empty:
        return base_df

    def normalize_component(name: str) -> str | None:
        key = normalize_name(name)
        if not key:
            return None
        return name_overrides.get(key, key)

    component_rows: list[dict[str, Any]] = []
    for row in base_df.itertuples(index=False):
        raw_name = row.player_name or ""
        is_combo = bool(row.is_combo) or ("+" in raw_name)
        parts = (
            [raw_name]
            if not is_combo
            else [p for p in COMBO_SPLIT.split(raw_name) if p.strip()]
        )
        if not parts:
            continue
        for part in parts:
            component_rows.append(
                {
                    "snapshot_id": row.snapshot_id,
                    "projection_id": row.projection_id,
                    "nba_game_id": row.nba_game_id,
                    "component_name_key": normalize_component(part),
                }
            )

    components_df = pd.DataFrame(component_rows)
    if components_df.empty:
        return base_df

    components_df = components_df.dropna(subset=["component_name_key", "nba_game_id"])
    if components_df.empty:
        return base_df

    expected_counts = components_df.groupby(["snapshot_id", "projection_id"])[
        "component_name_key"
    ].count()

    merged = components_df.merge(
        nba_players,
        left_on="component_name_key",
        right_on="name_key",
        how="left",
    )
    merged = merged.dropna(subset=["nba_player_id"])
    if merged.empty:
        return base_df

    game_ids = sorted(merged["nba_game_id"].unique().tolist())
    stats = pd.read_sql(
        text(
            """
            select
                game_id as nba_game_id,
                player_id as nba_player_id,
                points,
                rebounds,
                assists,
                steals,
                blocks,
                turnovers,
                fg3m,
                fg3a,
                fgm,
                fga,
                ftm,
                fta,
                cast(stats_json->>'OREB' as float) as oreb,
                cast(stats_json->>'DREB' as float) as dreb,
                cast(stats_json->>'PF' as float) as pf,
                cast(stats_json->>'DUNKS' as float) as dunks
            from nba_player_game_stats
            where game_id = any(:game_ids)
            """
        ),
        engine,
        params={"game_ids": game_ids},
    )
    if stats.empty:
        return base_df

    merged = merged.merge(
        stats,
        on=["nba_player_id", "nba_game_id"],
        how="left",
    )

    stat_cols = STAT_COLUMNS
    merged["has_stats"] = merged[stat_cols].notna().any(axis=1)
    merged = merged[merged["has_stats"]]
    if merged.empty:
        return base_df

    matched_counts = merged.groupby(["snapshot_id", "projection_id"])[
        "nba_player_id"
    ].count()
    expected_aligned = expected_counts.reindex(matched_counts.index)
    valid_ids = matched_counts[matched_counts == expected_aligned].index
    if valid_ids.empty:
        return base_df

    aggregated = (
        merged.set_index(["snapshot_id", "projection_id"])
        .loc[valid_ids]
        .reset_index()
        .groupby(["snapshot_id", "projection_id"], as_index=False)[stat_cols]
        .sum(min_count=1)
    )

    full_df = base_df.merge(
        aggregated, on=["snapshot_id", "projection_id"], how="inner"
    )
    full_df = _add_history_features(full_df, engine)
    return full_df


def _map_nba_player_ids(frame: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players"),
        engine,
    )
    if nba_players.empty:
        frame["nba_player_id"] = None
        return frame

    name_overrides = _load_name_overrides()

    def normalize_component(name: str) -> str | None:
        key = normalize_name(name)
        if not key:
            return None
        return name_overrides.get(key, key)

    frame = frame.copy()
    frame["normalized_name_key"] = frame["player_name"].apply(normalize_component)
    merged = frame.merge(
        nba_players,
        left_on="normalized_name_key",
        right_on="name_key",
        how="left",
    )
    return merged


def _load_gamelogs(engine: Engine) -> pd.DataFrame:
    return load_gamelogs_frame(engine)


OPPONENT_FEATURE_COLS = [
    "is_home",
    "opp_def_stat_avg",
    "opp_def_points_avg",
    "opp_def_rebounds_avg",
    "opp_def_assists_avg",
    "opp_def_rank",
]

MARKET_FEATURE_COLS = [
    "market_total_points",
    "market_spread_abs",
    "market_total_move",
    "market_volatility",
    "market_home_implied_total",
    "market_away_implied_total",
]


def _resolve_nba_game_ids(frame: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Resolve PrizePicks game_id -> nba_game_id when not already on the frame."""
    if "nba_game_id" in frame.columns and frame["nba_game_id"].notna().any():
        return frame
    if "game_id" not in frame.columns:
        frame["nba_game_id"] = None
        return frame
    game_ids = sorted({str(v) for v in frame["game_id"].dropna().unique()})
    if not game_ids:
        frame["nba_game_id"] = None
        return frame
    overrides = _load_team_abbrev_overrides()
    cte, params = _build_team_abbrev_cte(overrides)
    params["game_ids"] = game_ids
    mapping = pd.read_sql(
        text(
            f"""
            with {cte}
            select
                g.id as game_id,
                ng.id as nba_game_id
            from games g
            left join teams ht on ht.id = g.home_team_id
            left join teams at on at.id = g.away_team_id
            left join team_abbrev_map htm on htm.source_abbr = upper(ht.abbreviation)
            left join team_abbrev_map atm on atm.source_abbr = upper(at.abbreviation)
            join nba_games ng
                on ng.game_date = (g.start_time at time zone 'America/New_York')::date
                and ng.home_team_abbreviation = upper(coalesce(htm.normalized_abbr, ht.abbreviation))
                and ng.away_team_abbreviation = upper(coalesce(atm.normalized_abbr, at.abbreviation))
            where g.id = any(:game_ids)
            """
        ),
        engine,
        params=params,
    )
    if mapping.empty:
        frame["nba_game_id"] = None
        return frame
    frame = frame.merge(mapping, on="game_id", how="left")
    return frame


def _load_market_lines_by_game(engine: Engine, nba_game_ids: list[str]) -> pd.DataFrame:
    """Load market lines joined to nba_game_id for provided games."""
    if not nba_game_ids:
        return pd.DataFrame()
    try:
        return pd.read_sql(
            text(
                """
                select
                    ng.id as nba_game_id,
                    m.captured_at,
                    m.home_spread,
                    m.away_spread,
                    m.total_points
                from market_game_lines m
                join nba_games ng
                  on ng.game_date = m.game_date
                 and upper(ng.home_team_abbreviation) = upper(m.home_team_abbreviation)
                 and upper(ng.away_team_abbreviation) = upper(m.away_team_abbreviation)
                where ng.id = any(:game_ids)
                """
            ),
            engine,
            params={"game_ids": nba_game_ids},
        )
    except Exception:
        # Table might not exist yet in dev/local; features will default to zeros.
        return pd.DataFrame()


def _add_history_features(frame: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame = _map_nba_player_ids(frame, engine)
    frame = _resolve_nba_game_ids(frame, engine)
    gamelogs = prepare_gamelogs(_load_gamelogs(engine))
    league_means_timeline = build_league_means_timeline(gamelogs)
    logs_by_player, log_dates_by_player = build_logs_by_player(gamelogs)
    empty_logs = gamelogs.iloc[0:0]
    empty_dates = np.array([], dtype="datetime64[ns]")

    # Opponent defensive context
    team_game_stats = _load_team_game_stats(engine)
    opp_def_avgs = build_opponent_defensive_averages(team_game_stats)
    opp_def_ranks = build_opponent_defensive_ranks(opp_def_avgs)
    team_pace_data = compute_team_pace(team_game_stats)
    market_lines = _load_market_lines_by_game(
        engine,
        sorted(
            {
                str(v)
                for v in frame.get("nba_game_id", pd.Series(dtype=object))
                .dropna()
                .unique()
            }
        ),
    )

    # Build player -> team abbreviation lookup from nba_players
    nba_player_teams = pd.read_sql(
        text("select id as nba_player_id, team_abbreviation from nba_players"),
        engine,
    )
    player_team_map: dict[str, str] = {}
    for row in nba_player_teams.itertuples(index=False):
        if row.team_abbreviation:
            player_team_map[str(row.nba_player_id)] = str(row.team_abbreviation).upper()

    # Build nba_game_id -> (home_abbr, away_abbr) lookup
    game_teams: dict[str, tuple[str, str]] = {}
    for row in team_game_stats.drop_duplicates("game_id").itertuples(index=False):
        game_teams[str(row.game_id)] = (
            str(row.home_team_abbreviation).upper(),
            str(row.away_team_abbreviation).upper(),
        )

    # Fallback: PrizePicks game_id -> (home_abbr, away_abbr) for upcoming games
    # where nba_game_id is null.
    pp_game_teams: dict[str, tuple[str, str]] = {}
    pp_game_ids = sorted(
        {
            str(v)
            for v in frame["game_id"].dropna().unique()
            if "nba_game_id" not in frame.columns
            or frame.loc[frame["game_id"] == v, "nba_game_id"].isna().all()
        }
    )
    if pp_game_ids:
        overrides = _load_team_abbrev_overrides()
        override_map = {k.upper(): v.upper() for k, v in overrides.items()}
        pp_games = pd.read_sql(
            text(
                """
                select g.id as game_id, ht.abbreviation as home_abbr, at.abbreviation as away_abbr
                from games g
                left join teams ht on ht.id = g.home_team_id
                left join teams at on at.id = g.away_team_id
                where g.id = any(:game_ids)
                """
            ),
            engine,
            params={"game_ids": pp_game_ids},
        )
        for r in pp_games.itertuples(index=False):
            h = override_map.get(
                str(r.home_abbr or "").upper(), str(r.home_abbr or "").upper()
            )
            a = override_map.get(
                str(r.away_abbr or "").upper(), str(r.away_abbr or "").upper()
            )
            if h and a:
                pp_game_teams[str(r.game_id)] = (h, a)

    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce")
    start_time = None
    if "start_time" in frame.columns:
        start_time = pd.to_datetime(frame["start_time"], errors="coerce")
    cutoff_times = []
    for idx in range(len(frame)):
        fetch = fetched_at.iloc[idx] if fetched_at is not None else None
        start = start_time.iloc[idx] if start_time is not None else None
        if pd.notna(fetch) and pd.notna(start):
            cutoff_times.append(min(fetch, start))
        else:
            cutoff_times.append(fetch if pd.notna(fetch) else start)
    frame["cutoff_time"] = cutoff_times

    extras_rows: list[dict[str, float]] = []
    travel_cache: dict[tuple[str, str], dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        nba_player_id = getattr(row, "nba_player_id", None)
        if nba_player_id is None:
            extras_rows.append({})
            continue
        stat_type = getattr(row, "stat_type", None)
        stat_key = normalize_stat_type(stat_type)
        line_score = getattr(row, "line_score", None)
        cutoff = getattr(row, "cutoff_time", None)
        league_mean = league_mean_before_cutoff(
            league_means_timeline,
            str(stat_key or ""),
            cutoff,
            default=0.0,
        )
        player_key = str(nba_player_id)
        logs = logs_by_player.get(player_key, empty_logs)
        log_dates = log_dates_by_player.get(player_key, empty_dates)
        logs = slice_player_logs_before_cutoff(logs, log_dates, cutoff)
        hist_feats = compute_history_features(
            stat_type=str(stat_type) if stat_type is not None else "",
            line_score=float(line_score) if line_score is not None else 0.0,
            cutoff=cutoff,
            player_logs=logs,
            league_mean=float(league_mean),
            logs_prefiltered=True,
        )

        # Opponent features
        nba_game_id = getattr(row, "nba_game_id", None)
        pp_game_id = getattr(row, "game_id", None)
        player_team = player_team_map.get(player_key)
        opp_team = None
        is_home = None
        # Try nba_game_id first, fall back to PrizePicks game_id for upcoming games
        resolved = False
        if nba_game_id and str(nba_game_id) in game_teams:
            home_abbr, away_abbr = game_teams[str(nba_game_id)]
            if player_team:
                if player_team == home_abbr:
                    opp_team = away_abbr
                    is_home = 1
                    resolved = True
                elif player_team == away_abbr:
                    opp_team = home_abbr
                    is_home = 0
                    resolved = True
        if not resolved and pp_game_id and str(pp_game_id) in pp_game_teams:
            home_abbr, away_abbr = pp_game_teams[str(pp_game_id)]
            if player_team:
                if player_team == home_abbr:
                    opp_team = away_abbr
                    is_home = 1
                elif player_team == away_abbr:
                    opp_team = home_abbr
                    is_home = 0
        opp_feats = compute_opponent_features(
            player_team_abbr=player_team,
            opp_team_abbr=opp_team,
            game_date=cutoff,
            stat_type=str(stat_type) if stat_type is not None else "",
            opp_def_avgs=opp_def_avgs,
            is_home=is_home,
            opp_def_ranks=opp_def_ranks,
        )

        # Team pace + player usage
        cutoff_ts = (
            pd.to_datetime(cutoff, errors="coerce") if cutoff is not None else None
        )
        if isinstance(cutoff_ts, pd.Timestamp) and cutoff_ts.tz is not None:
            cutoff_ts = cutoff_ts.tz_localize(None)
        if isinstance(cutoff_ts, pd.Timestamp):
            cutoff_ts = cutoff_ts.normalize()

        def _latest_pace(team_abbr: str | None) -> float:
            if not team_abbr or team_abbr not in team_pace_data:
                return 0.0
            tdf = team_pace_data[team_abbr]
            if cutoff_ts is not None:
                tdf = tdf[tdf["game_date"] < cutoff_ts]
            if tdf.empty:
                return 0.0
            val = tdf.iloc[-1]["team_pace"]
            return float(val) if pd.notna(val) else 0.0

        t_pace = _latest_pace(player_team)
        o_pace = _latest_pace(opp_team)
        g_pace = (t_pace + o_pace) / 2.0 if (t_pace and o_pace) else t_pace or o_pace

        # Player usage from recent gamelogs
        p_usage = 0.0
        if not logs.empty and player_team:
            recent = logs.tail(10)
            p_fga = (
                float(recent["fga"].fillna(0).mean())
                if "fga" in recent.columns
                else 0.0
            )
            p_fta = (
                float(recent["fta"].fillna(0).mean())
                if "fta" in recent.columns
                else 0.0
            )
            p_to = (
                float(recent["turnovers"].fillna(0).mean())
                if "turnovers" in recent.columns
                else 0.0
            )
            tp = team_pace_data.get(player_team)
            if tp is not None:
                tp_hist = (
                    tp[tp["game_date"] < cutoff_ts] if cutoff_ts is not None else tp
                )
                if not tp_hist.empty:
                    tgs = team_game_stats[
                        team_game_stats["team_abbreviation"] == player_team
                    ].sort_values("game_date")
                    if cutoff_ts is not None:
                        tgs = tgs[tgs["game_date"] < cutoff_ts]
                    tgs_recent = tgs.tail(10)
                    if not tgs_recent.empty:
                        t_fga = float(tgs_recent["fga"].fillna(0).mean())
                        t_fta = float(tgs_recent["fta"].fillna(0).mean())
                        t_to = float(tgs_recent["turnovers"].fillna(0).mean())
                        p_usage = compute_player_usage(
                            p_fga, p_fta, p_to, t_fga, t_fta, t_to
                        )

        pace_usage_feats = {
            "team_pace": t_pace,
            "opp_pace": o_pace,
            "game_pace": g_pace,
            "player_usage": p_usage,
        }
        travel_key = (
            str(player_team or ""),
            str(pd.to_datetime(cutoff, errors="coerce").date())
            if cutoff is not None
            else "",
        )
        if travel_key not in travel_cache:
            travel_cache[travel_key] = estimate_team_travel_context(
                team_game_stats,
                team_abbreviation=player_team,
                cutoff=cutoff,
                lookback_games=7,
            )
        travel_feats = travel_cache[travel_key]

        market_feats = {
            "market_total_points": 0.0,
            "market_spread_abs": 0.0,
            "market_total_move": 0.0,
            "market_volatility": 0.0,
            "market_home_implied_total": 0.0,
            "market_away_implied_total": 0.0,
        }
        if (
            nba_game_id is not None
            and not market_lines.empty
            and "nba_game_id" in market_lines.columns
        ):
            ml = market_lines[market_lines["nba_game_id"].astype(str) == str(nba_game_id)].copy()
            if not ml.empty:
                ml["captured_at"] = pd.to_datetime(ml["captured_at"], errors="coerce", utc=True)
                if cutoff is not None:
                    cutoff_utc = pd.to_datetime(cutoff, errors="coerce", utc=True)
                    ml = ml[ml["captured_at"] <= cutoff_utc]
                if not ml.empty:
                    ml = ml.sort_values("captured_at")
                    latest = ml.iloc[-1]
                    total = pd.to_numeric(latest.get("total_points"), errors="coerce")
                    hs = pd.to_numeric(latest.get("home_spread"), errors="coerce")
                    aspr = pd.to_numeric(latest.get("away_spread"), errors="coerce")
                    spread = hs if pd.notna(hs) else aspr
                    spread_abs = abs(float(spread)) if pd.notna(spread) else 0.0
                    market_feats["market_total_points"] = (
                        float(total) if pd.notna(total) else 0.0
                    )
                    market_feats["market_spread_abs"] = spread_abs
                    if len(ml) >= 2:
                        first_total = pd.to_numeric(
                            ml.iloc[0].get("total_points"), errors="coerce"
                        )
                        if pd.notna(total) and pd.notna(first_total):
                            market_feats["market_total_move"] = float(total - first_total)
                        spread_series = pd.to_numeric(
                            ml["home_spread"].fillna(ml["away_spread"]), errors="coerce"
                        ).dropna()
                        if len(spread_series) >= 2:
                            market_feats["market_volatility"] = float(spread_series.std(ddof=0))
                    if pd.notna(total) and pd.notna(hs):
                        home_implied = float(total) / 2.0 - float(hs) / 2.0
                        market_feats["market_home_implied_total"] = home_implied
                        market_feats["market_away_implied_total"] = float(total) - home_implied

        extras_rows.append(
            {
                **hist_feats,
                **opp_feats,
                **pace_usage_feats,
                **travel_feats,
                **market_feats,
            }
        )

    extras_df = pd.DataFrame(extras_rows)
    for key in extras_df.columns:
        frame[key] = extras_df[key]

    # Derived line movement features
    line_score = pd.to_numeric(frame.get("line_score"), errors="coerce").fillna(0.0)
    line_delta = pd.to_numeric(frame.get("line_score_delta"), errors="coerce").fillna(
        0.0
    )
    mins_to_start = pd.to_numeric(
        frame.get("minutes_to_start"), errors="coerce"
    ).fillna(360.0)
    frame["line_move_pct"] = np.where(line_score > 0, line_delta / line_score, 0.0)
    # Late movement signal: movement magnitude weighted by proximity to game time
    # Closer to game time (lower minutes) = sharper signal
    time_weight = np.clip(1.0 - mins_to_start / 360.0, 0.0, 1.0)
    frame["line_move_late"] = line_delta.abs() * time_weight

    # Differential features derived from history + opponent context
    opp_def = pd.to_numeric(frame.get("opp_def_stat_avg"), errors="coerce").fillna(0.0)
    lm = pd.to_numeric(frame.get("league_mean"), errors="coerce").fillna(0.0)
    ls = pd.to_numeric(frame.get("line_score"), errors="coerce").fillna(0.0)
    frame["line_vs_opp_def"] = ls - opp_def
    frame["opp_def_ratio"] = np.where(lm > 0, opp_def / lm, 1.0)

    frame = frame.replace([float("inf"), float("-inf")], pd.NA)
    for col in [
        "hist_n",
        "hist_mean",
        "hist_std",
        "league_mean",
        "mu_stab",
        "p_hist_over",
        "z_line",
        "rest_days",
        "is_back_to_back",
        "stat_mean_3",
        "stat_mean_5",
        "stat_mean_10",
        "minutes_mean_3",
        "minutes_mean_5",
        "minutes_mean_10",
        "trend_slope",
        "stat_cv",
        "recent_vs_season",
        "minutes_trend",
        "minutes_std_10",
        "recent_load_3",
        "role_stability",
        "stat_std_5",
        "line_move_pct",
        "line_move_late",
        "stat_rate_per_min",
        "line_vs_mean_ratio",
        "hot_streak_count",
        "cold_streak_count",
        "season_game_number",
        "forecast_edge",
        "line_vs_opp_def",
        "opp_def_ratio",
        "team_pace",
        "opp_pace",
        "game_pace",
        "player_usage",
        "travel_km_7d",
        "circadian_shift_hours",
        *MARKET_FEATURE_COLS,
        *OPPONENT_FEATURE_COLS,
    ]:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame
