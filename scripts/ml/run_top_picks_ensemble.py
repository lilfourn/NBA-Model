from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.db.prediction_logs import append_prediction_rows  # noqa: E402
from app.ml.infer_baseline import infer_over_probs as infer_lr_over_probs  # noqa: E402
from app.ml.nn.infer import (  # noqa: E402
    infer_over_probs as infer_nn_over_probs,
    latest_compatible_checkpoint,
)  # noqa: E402
from app.ml.tabdl.infer import (  # noqa: E402
    infer_over_probs as infer_tabdl_over_probs,
    latest_compatible_checkpoint as latest_compatible_tabdl_checkpoint,
)
from app.ml.xgb.infer import infer_over_probs as infer_xgb_over_probs  # noqa: E402
from app.ml.lgbm.infer import infer_over_probs as infer_lgbm_over_probs  # noqa: E402
from app.ml.artifacts import (  # noqa: E402
    load_joblib_artifact,
    latest_compatible_joblib_path,
)  # noqa: E402
from app.ml.artifact_store import load_latest_artifact_as_file  # noqa: E402
from app.ml.meta_learner import infer_meta_learner  # noqa: E402
from app.modeling.db_logs import load_db_game_logs  # noqa: E402
from app.modeling.forecast_calibration import (  # noqa: E402
    ForecastDistributionCalibrator,
)  # noqa: E402
from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler  # noqa: E402
from app.modeling.conformal import ConformalCalibrator  # noqa: E402
from app.modeling.probability import confidence_from_probability  # noqa: E402
from app.services.scoring import _conformal_set_size, shrink_probability  # noqa: E402
from app.modeling.stat_forecast import (  # noqa: E402
    ForecastParams,
    LeaguePriors,
    StatForecastPredictor,
)  # noqa: E402
from app.modeling.types import Projection  # noqa: E402
from app.ml.selection_policy import SelectionPolicy  # noqa: E402
from app.ml.stat_mappings import (  # noqa: E402
    stat_components,
    stat_diff_components,
    stat_weighted_components,
)
from scripts.ops.log_decisions import (  # noqa: E402
    PRED_LOG_DEFAULT,
    append_prediction_log,
)  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _latest_snapshot_id(engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("select id from snapshots order by fetched_at desc limit 1")
        ).scalar()


def _auto_calibration_path(calibration_arg: str | None) -> str | None:
    if calibration_arg:
        return calibration_arg
    base = Path("data/calibration")
    if not base.exists():
        return None
    candidates = list(base.glob("forecast_calibration_*.json"))
    default = base / "forecast_calibration.json"
    if default.exists():
        candidates.append(default)
    if not candidates:
        return None
    # Prefer most-recently modified calibration.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _snapshot_id_for_game_date(engine, game_date: str) -> str | None:
    """
    Return the most recent snapshot id (by fetched_at) that contains at least one
    *standard* projection for the given slate date.

    Date is interpreted in America/New_York to match nba_games join logic elsewhere.
    """
    with engine.connect() as conn:
        return conn.execute(
            text(
                """
                select pf.snapshot_id
                from projection_features pf
                join projections p
                  on p.snapshot_id = pf.snapshot_id
                 and p.projection_id = pf.projection_id
                join snapshots s on s.id = pf.snapshot_id
                where (pf.start_time at time zone 'America/New_York')::date = :game_date
                  and coalesce(p.odds_type, 0) = 0
                order by s.fetched_at desc
                limit 1
                """
            ),
            {"game_date": game_date},
        ).scalar()


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_over_raw(
    *, mu: float, sigma: float, line: float, continuity: float = 0.5
) -> float:
    if sigma <= 0:
        return 0.5
    z = ((line + continuity) - mu) / sigma
    return 1.0 - _normal_cdf(z)


def _safe_prob(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _row_get(row: object, key: str) -> object:
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(row, key, None)


def _risk_adjusted_confidence(
    *, p_over: float, n_eff: float | None, status: str
) -> float:
    p_pick = max(p_over, 1.0 - p_over)
    rho = 0.85
    if n_eff is None or n_eff < 5:
        rho = 0.65
    if status == "raw":
        rho = min(rho, 0.60)
    elif status == "fallback_global":
        rho = min(rho, 0.75)
    return 0.5 + rho * (p_pick - 0.5)


def _ensure_ensemble_experts(
    ens: ContextualHedgeEnsembler, experts: list[str]
) -> ContextualHedgeEnsembler:
    if ens.experts == experts:
        return ens
    ens.experts = list(experts)
    seed = ens._init_weights()  # noqa: SLF001
    for ctx_key, weights in ens.weights.items():
        for expert in experts:
            if expert not in weights:
                weights[expert] = float(seed.get(expert, 0.0))
        total = sum(max(0.0, float(weights.get(expert, 0.0))) for expert in experts)
        if not math.isfinite(total) or total <= 0:
            ens.weights[ctx_key] = dict(seed)
            continue
        for expert in experts:
            weights[expert] = max(0.0, float(weights.get(expert, 0.0))) / total
    return ens


def _load_projection_frame(
    engine, snapshot_id: str, *, include_non_today: bool
) -> pd.DataFrame:
    query = text(
        """
        select
            pf.snapshot_id,
            pf.projection_id,
            pf.player_id,
            pf.game_id,
            pf.line_score,
            pf.stat_type,
            pf.projection_type,
            pf.trending_count,
            pf.is_live,
            pf.in_game,
            pf.today,
            pf.minutes_to_start,
            pf.fetched_at,
            pf.start_time,
            pl.display_name as player_name,
            coalesce(nullif(pl.image_url, ''), nullif(p.attributes->>'custom_image', '')) as player_image_url,
            pl.combo as combo
        from projection_features pf
        join projections p
            on p.snapshot_id = pf.snapshot_id
            and p.projection_id = pf.projection_id
        join players pl on pl.id = pf.player_id
        where pf.snapshot_id = :snapshot_id
          and coalesce(p.odds_type, 0) = 0
          and lower(coalesce(p.event_type, p.attributes->>'event_type', '')) <> 'combo'
          and (pl.combo is null or pl.combo = false)
          and (:include_non_today = true or coalesce(pf.today, false) = true)
          and (pf.is_live is null or pf.is_live = false)
          and (pf.in_game is null or pf.in_game = false)
          and (pf.minutes_to_start is null or pf.minutes_to_start >= 0)
        """
    )
    return pd.read_sql(
        query,
        engine,
        params={
            "snapshot_id": snapshot_id,
            "include_non_today": bool(include_non_today),
        },
    )


def _is_supported_stat_type(stat_type: str) -> bool:
    if not stat_type:
        return False
    return (
        stat_components(stat_type) is not None
        or stat_diff_components(stat_type) is not None
        or stat_weighted_components(stat_type) is not None
    )


def _to_projection(row: object) -> Projection:
    return Projection(
        projection_id=str(_row_get(row, "projection_id")),
        player_id=str(_row_get(row, "player_id") or ""),
        player_name=str(_row_get(row, "player_name") or ""),
        stat_type=str(_row_get(row, "stat_type") or ""),
        line_score=float(_row_get(row, "line_score") or 0.0),
        start_time=_row_get(row, "start_time"),
        game_id=_row_get(row, "game_id"),
        event_type=None,
        projection_type=_row_get(row, "projection_type"),
        trending_count=_row_get(row, "trending_count"),
        is_today=bool(_row_get(row, "today"))
        if _row_get(row, "today") is not None
        else None,
        is_combo=bool(_row_get(row, "combo") or False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print top picks from an online-ensemble (forecast+nn+lr)."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument(
        "--game-date",
        default=None,
        help="Optional slate date (YYYY-MM-DD, America/New_York). If set and --snapshot-id is not provided, selects the latest snapshot containing that date.",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--calibration", default=None, help="Optional forecast calibration JSON."
    )
    parser.add_argument("--ensemble-weights", default="models/ensemble_weights.json")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument(
        "--include-non-today",
        action="store_true",
        help="Include projections not flagged as today by PrizePicks.",
    )
    parser.add_argument(
        "--rank", default="risk_adj", choices=["risk_adj", "confidence", "edge", "ev"]
    )
    parser.add_argument("--decimal-odds", type=float, default=None)
    parser.add_argument("--break-even-prob", type=float, default=None)
    parser.add_argument("--log-decisions", action="store_true")
    parser.add_argument("--log-path", default=PRED_LOG_DEFAULT)
    parser.add_argument(
        "--no-db-log",
        action="store_true",
        help="Skip inserting logged predictions into Postgres projection_predictions.",
    )
    parser.add_argument(
        "--log-top-only",
        action="store_true",
        help="Log only the displayed top picks (default: log all scored rows).",
    )
    args = parser.parse_args()
    if (
        args.rank in {"edge", "ev"}
        and args.decimal_odds is None
        and args.break_even_prob is None
    ):
        raise SystemExit("--rank edge/ev requires --decimal-odds or --break-even-prob.")

    load_env()
    engine = get_engine(args.database_url)
    snapshot_id = args.snapshot_id
    if not snapshot_id and args.game_date:
        snapshot_id = _snapshot_id_for_game_date(engine, str(args.game_date))
    snapshot_id = snapshot_id or _latest_snapshot_id(engine)
    if not snapshot_id:
        print("No snapshots found.")
        return

    frame = _load_projection_frame(
        engine,
        str(snapshot_id),
        include_non_today=bool(args.include_non_today),
    )
    if frame.empty:
        print("No projections available for this snapshot.")
        return

    unsupported = (
        frame["stat_type"]
        .fillna("")
        .astype(str)
        .map(lambda v: not _is_supported_stat_type(v))
    )
    if unsupported.any():
        dropped = int(unsupported.sum())
        frame = frame[~unsupported].copy()
        print(
            f"Note: dropped {dropped} projections with unsupported stat_type (can't be labeled/backtested)."
        )
        if frame.empty:
            print("No supported projections left after filtering.")
            return

    args.calibration = _auto_calibration_path(args.calibration)
    if args.calibration:
        print(f"Using calibration: {args.calibration}")

    models_dir = Path(args.models_dir)
    nn_path = latest_compatible_checkpoint(models_dir, "nn_gru_attention_*.pt")
    tabdl_path = latest_compatible_tabdl_checkpoint(models_dir, "tabdl_*.pt")
    lr_path = latest_compatible_joblib_path(models_dir, "baseline_logreg_*.joblib")
    xgb_path = latest_compatible_joblib_path(models_dir, "xgb_*.joblib")
    lgbm_path = latest_compatible_joblib_path(models_dir, "lgbm_*.joblib")
    meta_path = latest_compatible_joblib_path(models_dir, "meta_learner_*.joblib")

    calibrator = None
    if args.calibration:
        calibrator = ForecastDistributionCalibrator.load(args.calibration)

    # Forecast expert
    logs = load_db_game_logs(engine)
    params = ForecastParams()
    needed_stat_types = sorted(
        {str(v) for v in frame["stat_type"].fillna("").tolist() if str(v)}
    )
    priors = LeaguePriors(
        logs, stat_types=needed_stat_types, minutes_prior=params.minutes_prior
    )
    forecast = StatForecastPredictor(
        logs,
        min_games=args.min_games,
        params=params,
        league_priors=priors,
        calibrator=calibrator,
    )

    forecast_map: dict[str, dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        proj = _to_projection(row)
        pred = forecast.predict(proj)
        if pred is None:
            continue
        p_fc = _safe_prob(pred.prob_over)
        if p_fc is None:
            continue
        details = pred.details or {}
        forecast_map[proj.projection_id] = {
            "p_forecast_cal": p_fc,
            "mu_hat": float(details.get("raw_mean", pred.mean or 0.0)),
            "sigma_hat": float(details.get("raw_std", pred.std or 0.0)),
            "n_eff": details.get("n_eff"),
            "calibration_status": str(details.get("calibration_status") or "raw"),
            "model_version": pred.model_version,
        }

    # NN expert (optional)
    p_nn: dict[str, float] = {}
    if nn_path:
        try:
            nn_inf = infer_nn_over_probs(
                engine=engine,
                model_path=str(nn_path),
                snapshot_id=str(snapshot_id),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: NN expert failed ({nn_path.name}): {exc.__class__.__name__}: {exc}. "
                "Continuing without NN."
            )
        else:
            for idx, r in enumerate(nn_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(nn_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_nn[str(proj_id)] = prob

    # Deep tabular expert (optional)
    p_tabdl: dict[str, float] = {}
    if tabdl_path:
        try:
            tabdl_inf = infer_tabdl_over_probs(
                engine=engine,
                model_path=str(tabdl_path),
                snapshot_id=str(snapshot_id),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: TabDL expert failed ({tabdl_path.name}): {exc.__class__.__name__}: {exc}. "
                "Continuing without TabDL."
            )
        else:
            for idx, r in enumerate(tabdl_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(tabdl_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_tabdl[str(proj_id)] = prob

    # LR expert (optional)
    p_lr: dict[str, float] = {}
    if lr_path:
        try:
            lr_inf = infer_lr_over_probs(
                engine=engine,
                model_path=str(lr_path),
                snapshot_id=str(snapshot_id),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: LR expert failed ({lr_path.name}): {exc.__class__.__name__}: {exc}. "
                "Continuing without LR. "
                "If this is a scikit-learn version mismatch, retrain with: "
                "`python -m scripts.ml.train_baseline_model`."
            )
        else:
            for idx, r in enumerate(lr_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(lr_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_lr[str(proj_id)] = prob

    # XGBoost expert (optional)
    p_xgb: dict[str, float] = {}
    if xgb_path:
        try:
            xgb_inf = infer_xgb_over_probs(
                engine=engine,
                model_path=str(xgb_path),
                snapshot_id=str(snapshot_id),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: XGB expert failed ({xgb_path.name}): {exc.__class__.__name__}: {exc}. "
                "Continuing without XGB."
            )
        else:
            for idx, r in enumerate(xgb_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(xgb_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_xgb[str(proj_id)] = prob

    # LightGBM expert (optional)
    p_lgbm: dict[str, float] = {}
    if lgbm_path:
        try:
            lgbm_inf = infer_lgbm_over_probs(
                engine=engine,
                model_path=str(lgbm_path),
                snapshot_id=str(snapshot_id),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: LGBM expert failed ({lgbm_path.name}): {exc.__class__.__name__}: {exc}. "
                "Continuing without LGBM."
            )
        else:
            for idx, r in enumerate(lgbm_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(lgbm_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_lgbm[str(proj_id)] = prob

    # Load conformal calibrators keyed by expert; each calibrator must score
    # the expert distribution it was fit on.
    conformal_by_expert: dict[str, ConformalCalibrator] = {}
    for expert_key, model_path in [
        ("p_lr", lr_path),
        ("p_xgb", xgb_path),
        ("p_lgbm", lgbm_path),
    ]:
        if model_path and model_path.exists():
            try:
                payload = load_joblib_artifact(str(model_path))
                conformal_data = payload.get("conformal")
                if conformal_data:
                    conformal_by_expert[expert_key] = ConformalCalibrator(
                        **conformal_data
                    )
            except Exception:  # noqa: BLE001
                pass
    if nn_path and nn_path.exists():
        try:
            import torch as _torch

            payload = _torch.load(str(nn_path), map_location="cpu")
            conformal_data = payload.get("conformal")
            if conformal_data:
                conformal_by_expert["p_nn"] = ConformalCalibrator(**conformal_data)
        except Exception:  # noqa: BLE001
            pass
    if tabdl_path and tabdl_path.exists():
        try:
            import torch as _torch

            payload = _torch.load(str(tabdl_path), map_location="cpu")
            conformal_data = payload.get("conformal")
            if conformal_data:
                conformal_by_expert["p_tabdl"] = ConformalCalibrator(
                    **conformal_data
                )
        except Exception:  # noqa: BLE001
            pass

    experts = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]
    if Path(args.ensemble_weights).exists():
        ens = _ensure_ensemble_experts(
            ContextualHedgeEnsembler.load(args.ensemble_weights), experts
        )
    else:
        ens = ContextualHedgeEnsembler(experts=experts, eta=0.2, shrink_to_uniform=0.01)

    # Load context priors, stat calibrator, and exclusion lists
    from app.ml.context_prior import get_context_prior, load_context_priors
    from app.ml.stat_calibrator import StatTypeCalibrator
    from app.services.scoring import (
        EXCLUDED_STAT_TYPES,
        PRIOR_ONLY_STAT_TYPES,
        MIN_EDGE,
    )

    _context_priors = load_context_priors()
    _stat_calibrator = StatTypeCalibrator.load()
    _selection_policy_path: Path | None = None
    try:
        db_policy = load_latest_artifact_as_file(
            engine, "selection_policy", suffix=".json"
        )
        if db_policy is not None:
            _selection_policy_path = Path(db_policy)
    except Exception:
        _selection_policy_path = None
    if _selection_policy_path is None:
        local_policy = Path(args.models_dir) / "selection_policy.json"
        if local_policy.exists():
            _selection_policy_path = local_policy
    _selection_policy = (
        SelectionPolicy.load(str(_selection_policy_path))
        if _selection_policy_path is not None
        else SelectionPolicy.load()
    )

    # Load inversion correction flags
    from app.ml.inversion_corrections import load_inversion_flags

    _inversion_flags = load_inversion_flags()
    if _inversion_flags:
        print(f"Inversion auto-correction active for: {list(_inversion_flags.keys())}")

    scored = []
    skipped_nonfinite = 0
    for row in frame.itertuples(index=False):
        proj_id = str(getattr(row, "projection_id", ""))
        stat_type = str(getattr(row, "stat_type", "") or "")
        if not stat_type or not proj_id:
            continue

        # Skip degenerate stat types
        if stat_type in EXCLUDED_STAT_TYPES:
            continue

        f = forecast_map.get(proj_id) or {}
        expert_probs = {
            "p_forecast_cal": _safe_prob(f.get("p_forecast_cal")),
            "p_nn": _safe_prob(p_nn.get(proj_id)),
            "p_tabdl": _safe_prob(p_tabdl.get(proj_id)),
            "p_lr": _safe_prob(p_lr.get(proj_id)),
            "p_xgb": _safe_prob(p_xgb.get(proj_id)),
            "p_lgbm": _safe_prob(p_lgbm.get(proj_id)),
        }
        # Auto-correct inverted experts
        for _inv_expert, _should_flip in _inversion_flags.items():
            if _should_flip and expert_probs.get(_inv_expert) is not None:
                expert_probs[_inv_expert] = 1.0 - expert_probs[_inv_expert]

        # Meta-learner blends LR/XGB/LGBM
        p_meta_val = None
        if meta_path:
            try:
                p_meta_val = infer_meta_learner(
                    model_path=str(meta_path),
                    expert_probs=expert_probs,
                )
            except Exception:  # noqa: BLE001
                pass
        is_live = bool(getattr(row, "is_live", False) or False)
        n_eff = f.get("n_eff")
        try:
            n_eff_val = float(n_eff) if n_eff is not None else None
        except (TypeError, ValueError):
            n_eff_val = None
        ctx = Context(stat_type=stat_type, is_live=is_live, n_eff=n_eff_val)
        _is_prior_only = stat_type in PRIOR_ONLY_STAT_TYPES
        _line_score = float(getattr(row, "line_score", 0.0) or 0.0)
        _ctx_prior = get_context_prior(
            _context_priors, stat_type=stat_type, line_score=_line_score
        )

        if _is_prior_only:
            p_raw = _ctx_prior if _ctx_prior is not None else 0.5
            p_final = p_raw
        else:
            p_raw = float(ens.predict(expert_probs, ctx))
            if not math.isfinite(p_raw):
                skipped_nonfinite += 1
                continue
            p_final = shrink_probability(
                p_raw, n_eff=n_eff_val, context_prior=_ctx_prior
            )
            # Apply per-stat isotonic recalibration
            p_final = _stat_calibrator.transform(p_final, stat_type)

        pick = "OVER" if p_final >= 0.5 else "UNDER"
        conf = float(confidence_from_probability(p_final))
        status = str(f.get("calibration_status") or "raw") if f else "raw"
        p_adj = float(
            _risk_adjusted_confidence(p_over=p_final, n_eff=n_eff_val, status=status)
        )

        score = p_adj
        if args.rank == "confidence":
            score = conf
        elif args.rank in {"edge", "ev"}:
            p_be = (
                float(args.break_even_prob)
                if args.break_even_prob is not None
                else (1.0 / float(args.decimal_odds))
            )
            if args.rank == "edge":
                score = p_adj - p_be
            else:
                decimal_odds = (
                    float(args.decimal_odds)
                    if args.decimal_odds is not None
                    else (1.0 / p_be)
                )
                score = (p_adj * decimal_odds) - 1.0
        if not math.isfinite(float(score)):
            skipped_nonfinite += 1
            continue

        conformal_size = _conformal_set_size(conformal_by_expert, expert_probs)
        p_pick = max(p_final, 1.0 - p_final)
        selection_threshold = float(
            _selection_policy.threshold_for(stat_type, conformal_size)
        )
        selection_margin = float(p_pick - selection_threshold)

        scored.append(
            {
                "projection_id": proj_id,
                "player_id": str(getattr(row, "player_id", "") or ""),
                "player_name": str(getattr(row, "player_name", "") or ""),
                "player_image_url": str(getattr(row, "player_image_url", "") or "")
                or None,
                "game_id": getattr(row, "game_id", None),
                "stat_type": stat_type,
                "line_score": float(getattr(row, "line_score", 0.0) or 0.0),
                "is_live": is_live,
                "pick": pick,
                "prob_over": p_final,
                "p_raw": p_raw,
                "confidence": conf,
                "rank_score": float(score),
                "p_forecast_cal": expert_probs["p_forecast_cal"],
                "p_nn": expert_probs["p_nn"],
                "p_tabdl": expert_probs["p_tabdl"],
                "p_lr": expert_probs["p_lr"],
                "p_xgb": expert_probs["p_xgb"],
                "p_lgbm": expert_probs["p_lgbm"],
                "p_meta": _safe_prob(p_meta_val),
                "mu_hat": float(f.get("mu_hat") or 0.0) if f else None,
                "sigma_hat": float(f.get("sigma_hat") or 0.0) if f else None,
                "calibration_status": status,
                "model_version": str(f.get("model_version") or "forecast")
                if f
                else None,
                "n_eff": n_eff_val,
                "conformal_set_size": conformal_size,
                "p_pick": p_pick,
                "selection_threshold": selection_threshold,
                "selection_margin": selection_margin,
                "policy_version": _selection_policy.version,
            }
        )

    if not scored:
        print("No rows scored. Check filters and data coverage.")
        return
    if skipped_nonfinite:
        print(
            f"Note: skipped {skipped_nonfinite} projections due to non-finite probabilities/scores."
        )

    # Compute edge score for each pick
    from app.services.scoring import _compute_edge, _grade_from_edge

    for item in scored:
        ep = {
            "p_forecast_cal": item.get("p_forecast_cal"),
            "p_nn": item.get("p_nn"),
            "p_tabdl": item.get("p_tabdl"),
            "p_lr": item.get("p_lr"),
            "p_xgb": item.get("p_xgb"),
            "p_lgbm": item.get("p_lgbm"),
            "p_meta": item.get("p_meta"),
        }
        item["edge"] = _compute_edge(
            item["prob_over"],
            ep,
            item.get("conformal_set_size"),
            n_eff=item.get("n_eff"),
        )
        item["grade"] = _grade_from_edge(item["edge"])

    # Apply publishability filters
    for item in scored:
        p_pick = float(item.get("p_pick") or max(item["prob_over"], 1.0 - item["prob_over"]))
        selection_threshold = float(item.get("selection_threshold") or 0.60)
        _st = item.get("stat_type", "")
        item["is_publishable"] = bool(
            p_pick >= selection_threshold
            and item.get("edge", 0) >= MIN_EDGE
            and _st not in PRIOR_ONLY_STAT_TYPES
        )

    scored.sort(key=lambda item: item["edge"], reverse=True)
    # Enforce abstain: only publishable picks in top-N
    publishable = [item for item in scored if item.get("is_publishable", False)]
    top = publishable[: args.top]

    print(f"\nTop {len(top)} Ensemble Picks (snapshot {snapshot_id})")
    print("=" * 80)
    for rank, item in enumerate(top, start=1):
        grade = item["grade"]
        edge = item["edge"]
        p = item["prob_over"]
        line = f"{item['player_name']} | {item['stat_type']} | line {item['line_score']:.2f}"
        detail = f"{item['pick']} {p:.1%} | edge {edge:.0f} [{grade}]"
        print(f"{rank:>2}. {line} -> {detail}")

    if args.log_decisions:
        decision_time = datetime.now(timezone.utc).isoformat()
        calibration_version = Path(args.calibration).name if args.calibration else None

        to_log = top if args.log_top_only else scored
        rows = []
        for item in to_log:
            mu_hat = item.get("mu_hat")
            sigma_hat = item.get("sigma_hat")
            line_score = float(item["line_score"])
            p_over_raw = None
            if mu_hat is not None and sigma_hat is not None:
                p_over_raw = float(
                    _p_over_raw(
                        mu=float(mu_hat), sigma=float(sigma_hat), line=line_score
                    )
                )
            rows.append(
                {
                    "snapshot_id": str(snapshot_id),
                    "projection_id": item["projection_id"],
                    "game_id": item.get("game_id"),
                    "player_id": item.get("player_id"),
                    "player_image_url": item.get("player_image_url"),
                    "stat_type": item["stat_type"],
                    "is_live": bool(item.get("is_live") or False),
                    "decision_time": decision_time,
                    "line_score": line_score,
                    "pick": item.get("pick"),
                    "confidence": item.get("confidence"),
                    "prob_over": item.get("prob_over"),
                    "mu_hat": mu_hat,
                    "sigma_hat": sigma_hat,
                    "p_over_raw": p_over_raw,
                    "p_over_cal": item.get("p_forecast_cal"),
                    "p_forecast_cal": item.get("p_forecast_cal"),
                    "p_nn": item.get("p_nn"),
                    "p_tabdl": item.get("p_tabdl"),
                    "p_lr": item.get("p_lr"),
                    "p_xgb": item.get("p_xgb"),
                    "p_lgbm": item.get("p_lgbm"),
                    "p_meta": item.get("p_meta"),
                    "p_raw": item.get("p_raw"),
                    "p_final": item.get("prob_over"),
                    "p_pick": item.get("p_pick"),
                    "selection_threshold": item.get("selection_threshold"),
                    "selection_margin": item.get("selection_margin"),
                    "policy_version": item.get("policy_version"),
                    "model_version": item.get("model_version"),
                    "calibration_version": calibration_version,
                    "calibration_status": item.get("calibration_status"),
                    "n_eff": item.get("n_eff"),
                    "rank_score": item.get("rank_score"),
                    "conformal_set_size": item.get("conformal_set_size"),
                    "edge": item.get("edge"),
                    "grade": item.get("grade"),
                    "is_publishable": bool(item.get("is_publishable")),
                }
            )
        append_prediction_log(pd.DataFrame(rows), path=args.log_path)
        print(f"\nLogged {len(rows)} predictions -> {args.log_path}")
        if not args.no_db_log:
            try:
                inserted = append_prediction_rows(engine, rows)
                print(f"Logged {inserted} predictions -> projection_predictions")
            except Exception as exc:
                import traceback

                traceback.print_exc()
                print(f"WARNING: DB prediction logging failed: {exc}")


if __name__ == "__main__":
    main()
