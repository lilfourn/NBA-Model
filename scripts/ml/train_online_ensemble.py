from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import _load_name_overrides  # noqa: E402
from app.ml.stat_mappings import stat_value_from_row  # noqa: E402
from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler, logloss  # noqa: E402
from app.utils.names import normalize_name  # noqa: E402
from scripts.ops.log_decisions import PRED_LOG_DEFAULT  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


EXPERT_COLS_DEFAULT = ["p_forecast_cal", "p_nn", "p_lr"]


def _parse_timestamp(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.fillna(pd.NaT)

def _normalize_id(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
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


def _load_outcomes(engine, df: pd.DataFrame) -> pd.DataFrame:
    need = df.dropna(subset=["player_id", "game_id"]).copy()
    if need.empty:
        return need

    need["player_id"] = need["player_id"].apply(_normalize_id)
    need["game_id"] = need["game_id"].apply(_normalize_id)
    need = need.dropna(subset=["player_id", "game_id"])
    if need.empty:
        return need

    player_ids = sorted({str(v) for v in need["player_id"].dropna().unique().tolist()})
    game_ids = sorted({str(v) for v in need["game_id"].dropna().unique().tolist()})

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

    name_overrides = _load_name_overrides()

    def to_name_key(row: pd.Series) -> str | None:
        raw = row.get("display_name") or row.get("name_key")
        key = normalize_name(raw)
        if not key:
            return None
        return name_overrides.get(key, key)

    players = players.copy()
    players["normalized_name_key"] = [to_name_key(row) for row in players.to_dict(orient="records")]

    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players where name_key = any(:keys)"),
        engine,
        params={"keys": sorted({str(v) for v in players["normalized_name_key"].dropna().unique().tolist()})},
    )

    mapped = players.merge(
        nba_players,
        left_on="normalized_name_key",
        right_on="name_key",
        how="left",
    )[["player_id", "nba_player_id"]]

    merged = need.merge(mapped, on="player_id", how="left").merge(games, on="game_id", how="left")
    merged = merged.dropna(subset=["nba_player_id", "game_date"])
    if merged.empty:
        return merged

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
        # No boxscore rows yet for the date window (likely games not completed).
        return merged.head(0)

    out = merged.merge(stats, on=["nba_player_id", "game_date"], how="left")
    out["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in out.itertuples(index=False)
    ]
    out = out.dropna(subset=["actual_value", "line_score"])
    out["over_label"] = (out["actual_value"].astype(float) > out["line_score"].astype(float)).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an online contextual Hedge ensemble from logged predictions.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--log-path", default=PRED_LOG_DEFAULT)
    ap.add_argument("--out", default="models/ensemble_weights.json")
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--shrink", type=float, default=0.01)
    ap.add_argument("--experts", nargs="*", default=None, help="Expert probability columns.")
    ap.add_argument(
        "--min-experts",
        type=int,
        default=2,
        help="Minimum number of non-null expert probs required per row.",
    )
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Prediction log not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("Prediction log is empty.")
        return

    df["player_id"] = df.get("player_id").apply(_normalize_id)
    df["game_id"] = df.get("game_id").apply(_normalize_id)

    experts = args.experts or EXPERT_COLS_DEFAULT
    missing = [col for col in experts if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing expert columns in log: {missing}")

    required = {"player_id", "game_id", "stat_type", "line_score"}
    req_missing = required - set(df.columns)
    if req_missing:
        raise SystemExit(f"Prediction log missing required columns: {sorted(req_missing)}")

    df["decision_time_parsed"] = _parse_timestamp(df.get("decision_time"))
    if df["decision_time_parsed"].isna().all():
        df["decision_time_parsed"] = _parse_timestamp(df.get("created_at"))

    for col in experts:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with enough expert probs.
    expert_non_null = df[experts].notna().sum(axis=1)
    df = df[expert_non_null >= int(args.min_experts)].copy()
    if df.empty:
        print("No rows with enough expert probabilities to train.")
        return

    df = df.dropna(subset=["stat_type", "line_score"])
    df["is_live"] = df.get("is_live", False).fillna(False).astype(bool)
    df["n_eff"] = pd.to_numeric(df.get("n_eff"), errors="coerce")

    joined = _load_outcomes(engine, df)
    if joined.empty:
        print("No outcomes available yet for logged rows.")
        return

    joined = joined.sort_values("decision_time_parsed")
    ens = ContextualHedgeEnsembler(experts=list(experts), eta=float(args.eta), shrink_to_uniform=float(args.shrink))

    losses: dict[str, float] = {name: 0.0 for name in ["ensemble", *experts]}
    counts: dict[str, int] = {name: 0 for name in ["ensemble", *experts]}

    for row in joined.itertuples(index=False):
        expert_probs = {col: getattr(row, col) for col in experts}
        ctx = Context(
            stat_type=str(getattr(row, "stat_type") or ""),
            is_live=bool(getattr(row, "is_live") or False),
            n_eff=float(getattr(row, "n_eff")) if getattr(row, "n_eff") is not None else None,
        )
        y = int(getattr(row, "over_label"))
        p_ens = float(ens.predict(expert_probs, ctx))
        losses["ensemble"] += float(logloss(y, p_ens))
        counts["ensemble"] += 1
        for col, p in expert_probs.items():
            if p is None or pd.isna(p):
                continue
            losses[col] += float(logloss(y, float(p)))
            counts[col] += 1
        ens.update(expert_probs, y, ctx)

    def avg(name: str) -> float | None:
        n = counts.get(name, 0)
        if n <= 0:
            return None
        return losses[name] / float(n)

    summary = {name: avg(name) for name in ["ensemble", *experts]}
    print({"rows": int(counts["ensemble"]), "avg_logloss": summary})

    out_path = Path(args.out)
    ens.save(out_path)
    print(f"Wrote ensemble weights -> {out_path}")


if __name__ == "__main__":
    main()
