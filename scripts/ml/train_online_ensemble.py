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


EXPERT_COLS_DEFAULT = ["p_forecast_cal", "p_nn", "p_lr", "p_xgb", "p_lgbm"]


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


def _write_back_outcomes(
    log_df: pd.DataFrame,
    resolved: pd.DataFrame,
    *,
    path: Path,
    only_missing: bool = True,
) -> int:
    if resolved.empty or "__row_id" not in resolved.columns:
        return 0

    if "actual_value" not in log_df.columns:
        log_df["actual_value"] = pd.NA
    if "over_label" not in log_df.columns:
        log_df["over_label"] = pd.NA

    updates = resolved.dropna(subset=["__row_id", "actual_value"]).copy()
    if updates.empty:
        return 0

    updates["__row_id"] = updates["__row_id"].astype(int)
    updates = updates.sort_values("__row_id").drop_duplicates(subset=["__row_id"], keep="last")

    updated_rows = 0
    for row_id_raw, actual_value, over_label in updates[["__row_id", "actual_value", "over_label"]].itertuples(index=False, name=None):
        row_id = int(row_id_raw)
        if row_id < 0 or row_id >= len(log_df):
            continue
        if only_missing and pd.notna(log_df.at[row_id, "actual_value"]):
            continue

        if actual_value is None or pd.isna(actual_value):
            continue

        log_df.at[row_id, "actual_value"] = float(actual_value)
        log_df.at[row_id, "over_label"] = int(over_label) if over_label is not None and not pd.isna(over_label) else pd.NA
        updated_rows += 1

    if updated_rows <= 0:
        return 0

    out_df = log_df.drop(columns=["__row_id"], errors="ignore")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out_df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return updated_rows


def _load_training_frame_from_db(engine, *, days_back: int) -> pd.DataFrame:
    frame = pd.read_sql(
        text(
            """
            select
                projection_id,
                player_id,
                game_id,
                stat_type,
                line_score,
                over_label,
                actual_value,
                n_eff,
                p_forecast_cal,
                p_nn,
                p_lr,
                p_xgb,
                p_lgbm,
                prob_over as p_final,
                coalesce(decision_time, created_at) as decision_time_parsed,
                details->>'is_live' as is_live
            from projection_predictions
            where over_label is not null
              and actual_value is not null
              and coalesce(decision_time, created_at) >= now() - (:days_back * interval '1 day')
            order by coalesce(decision_time, created_at) asc
            """
        ),
        engine,
        params={"days_back": int(max(1, days_back))},
    )
    if frame.empty:
        return frame

    frame["player_id"] = frame.get("player_id").apply(_normalize_id)
    frame["game_id"] = frame.get("game_id").apply(_normalize_id)
    frame["decision_time_parsed"] = _parse_timestamp(frame.get("decision_time_parsed"))
    frame["is_live"] = (
        frame.get("is_live")
        .fillna("false")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "t"})
    )
    frame["n_eff"] = pd.to_numeric(frame.get("n_eff"), errors="coerce")
    frame["line_score"] = pd.to_numeric(frame.get("line_score"), errors="coerce")
    frame["over_label"] = pd.to_numeric(frame.get("over_label"), errors="coerce")
    frame = frame.dropna(subset=["stat_type", "line_score", "over_label", "decision_time_parsed"])
    frame["over_label"] = frame["over_label"].astype(int)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an online contextual Hedge ensemble from logged predictions.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--log-path", default=PRED_LOG_DEFAULT)
    ap.add_argument("--out", default="models/ensemble_weights.json")
    ap.add_argument(
        "--source",
        choices=["db", "csv", "auto"],
        default="db",
        help="Training data source. 'db' uses projection_predictions outcomes, 'csv' uses prediction_log.csv.",
    )
    ap.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="When source includes db, only use resolved rows from the last N days.",
    )
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--shrink", type=float, default=0.01)
    ap.add_argument("--experts", nargs="*", default=None, help="Expert probability columns.")
    ap.add_argument(
        "--min-experts",
        type=int,
        default=2,
        help="Minimum number of non-null expert probs required per row.",
    )
    ap.add_argument(
        "--no-write-outcomes",
        action="store_true",
        help="Disable writing resolved actual_value/over_label back into prediction_log.csv.",
    )
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    joined: pd.DataFrame | None = None

    if args.source in {"db", "auto"}:
        db_frame = _load_training_frame_from_db(engine, days_back=int(args.days_back))
        if not db_frame.empty:
            joined = db_frame
            print(
                f"Loaded {len(joined)} resolved rows from projection_predictions "
                f"(days_back={int(args.days_back)})."
            )
        elif args.source == "db":
            print("No resolved projection outcomes found in projection_predictions.")
            return

    if joined is None and args.source in {"csv", "auto"}:
        log_path = Path(args.log_path)
        if not log_path.exists():
            print(f"Prediction log not found: {log_path}")
            return

        raw_df = pd.read_csv(log_path)
        if raw_df.empty:
            print("Prediction log is empty.")
            return

        required = {"player_id", "game_id", "stat_type", "line_score"}
        req_missing = required - set(raw_df.columns)
        if req_missing:
            raise SystemExit(f"Prediction log missing required columns: {sorted(req_missing)}")

        raw_df = raw_df.reset_index(drop=True)
        raw_df["__row_id"] = raw_df.index.astype(int)
        raw_df["player_id"] = raw_df["player_id"].apply(_normalize_id)
        raw_df["game_id"] = raw_df["game_id"].apply(_normalize_id)
        raw_df["decision_time_parsed"] = _parse_timestamp(raw_df.get("decision_time"))
        if raw_df["decision_time_parsed"].isna().all():
            raw_df["decision_time_parsed"] = _parse_timestamp(raw_df.get("created_at"))

        resolved_all = _load_outcomes(engine, raw_df)
        if args.no_write_outcomes:
            print("Outcome write-back disabled (--no-write-outcomes).")
        else:
            updated_rows = _write_back_outcomes(raw_df, resolved_all, path=log_path, only_missing=True)
            print(f"Outcome write-back: updated {updated_rows} row(s) in {log_path}.")

        df = raw_df.copy()

        experts = args.experts or EXPERT_COLS_DEFAULT
        missing = [col for col in experts if col not in df.columns]
        if missing:
            # New experts may not yet appear in historical logs. Train with available ones.
            print(f"Note: expert columns not yet in log (will be added on next scoring run): {missing}")
            experts = [col for col in experts if col in df.columns]
            if not experts:
                raise SystemExit("No expert columns found in prediction log.")

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

        if resolved_all.empty:
            print("No outcomes available yet for logged rows.")
            return

        joined = resolved_all[resolved_all["__row_id"].isin(df["__row_id"])].copy()
        if joined.empty:
            print("No outcomes available yet for rows eligible for ensemble training.")
            return
        for col in experts:
            joined[col] = pd.to_numeric(joined.get(col), errors="coerce")

    if joined is None or joined.empty:
        print("No training rows available after source selection.")
        return

    experts = args.experts or EXPERT_COLS_DEFAULT
    missing = [col for col in experts if col not in joined.columns]
    if missing:
        print(f"Note: expert columns missing from selected source: {missing}")
        experts = [col for col in experts if col in joined.columns]
        if not experts:
            raise SystemExit("No expert columns available to train ensemble.")

    for col in experts:
        joined[col] = pd.to_numeric(joined.get(col), errors="coerce")

    expert_non_null = joined[experts].notna().sum(axis=1)
    joined = joined[expert_non_null >= int(args.min_experts)].copy()
    if joined.empty:
        print("No resolved rows with enough expert probabilities to train.")
        return

    joined["n_eff"] = pd.to_numeric(joined.get("n_eff"), errors="coerce")
    joined["is_live"] = joined.get("is_live", False).fillna(False).astype(bool)
    joined = joined.dropna(subset=["over_label", "stat_type", "line_score", "decision_time_parsed"])
    if joined.empty:
        print("No resolved rows available after final validation filters.")
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
