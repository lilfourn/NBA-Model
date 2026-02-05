from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib as _joblib

import pandas as pd
import torch as _torch
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.infer_baseline import infer_over_probs as infer_lr_over_probs
from app.ml.nn.infer import infer_over_probs as infer_nn_over_probs
from app.ml.xgb.infer import infer_over_probs as infer_xgb_over_probs
from app.modeling.conformal import ConformalCalibrator
from app.modeling.db_logs import load_db_game_logs
from app.modeling.forecast_calibration import ForecastDistributionCalibrator
from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler
from app.modeling.probability import confidence_from_probability
from app.modeling.stat_forecast import ForecastParams, LeaguePriors, StatForecastPredictor
from app.modeling.types import Projection
from app.ml.stat_mappings import (
    stat_components,
    stat_diff_components,
    stat_weighted_components,
)


@dataclass
class ScoredPick:
    projection_id: str
    player_name: str
    player_image_url: str | None
    player_id: str
    game_id: str | None
    stat_type: str
    line_score: float
    pick: str
    prob_over: float
    confidence: float
    rank_score: float
    p_forecast_cal: float | None
    p_nn: float | None
    p_lr: float | None
    p_xgb: float | None
    mu_hat: float | None
    sigma_hat: float | None
    calibration_status: str
    n_eff: float | None
    conformal_set_size: int | None
    edge: float
    grade: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoringResult:
    snapshot_id: str
    scored_at: str
    total_scored: int
    picks: list[ScoredPick]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "scored_at": self.scored_at,
            "total_scored": self.total_scored,
            "picks": [p.to_dict() for p in self.picks],
        }


def _compute_edge(
    p_over: float,
    expert_probs: dict[str, float | None],
    conformal_set_size: int | None,
    n_eff: float | None = None,
) -> float:
    """Composite prediction score 0-100 combining all signals.

    Designed to discriminate between picks — a score of 90+ should be rare.

    Components (weights sum to 100):
      - Avg expert strength (35%): mean |p - 0.5| across available experts
      - Expert alignment   (30%): how tightly experts cluster (1 - spread)
      - Conformal bonus    (15%): full bonus if set_size==1
      - Data quality       (20%): based on n_eff (effective sample size)
    """
    available = [v for v in expert_probs.values() if v is not None]

    # 1. Average expert directional strength: mean of |p - 0.5| * 2 across experts
    if available:
        strengths = [abs(v - 0.5) * 2.0 for v in available]
        avg_strength = sum(strengths) / len(strengths)
    else:
        avg_strength = abs(p_over - 0.5) * 2.0

    # 2. Expert alignment: 1 - spread.  Spread = range of expert probs / 1.0
    #    All experts at same value → alignment=1.  Max disagreement → alignment=0.
    if len(available) >= 2:
        spread = max(available) - min(available)
        alignment = 1.0 - min(1.0, spread)
    else:
        alignment = 0.5

    # 3. Conformal bonus
    conformal = 1.0 if conformal_set_size == 1 else 0.0

    # 4. Data quality: log-scaled n_eff.  n_eff>=30 → full credit, 0 → no credit.
    if n_eff is not None and n_eff > 0:
        data_q = min(1.0, n_eff / 30.0)
    else:
        data_q = 0.3  # default when n_eff unknown

    edge = (avg_strength * 35.0) + (alignment * 30.0) + (conformal * 15.0) + (data_q * 20.0)
    return round(min(100.0, max(0.0, edge)), 1)


def _grade_from_edge(edge: float) -> str:
    if edge >= 90:
        return "A+"
    if edge >= 80:
        return "A"
    if edge >= 70:
        return "B"
    if edge >= 55:
        return "C"
    if edge >= 40:
        return "D"
    return "F"


def _conformal_set_size(calibrators: list[ConformalCalibrator], p_over: float) -> int | None:
    """Majority-vote conformal set size across available calibrators.

    Returns 1 (confident unique prediction) or 2 (ambiguous).
    Returns None if no calibrators are available.
    """
    if not calibrators:
        return None
    sizes = [cal.predict(p_over).set_size for cal in calibrators]
    # Majority vote: if most say set_size=1, return 1
    return 1 if sum(1 for s in sizes if s == 1) > len(sizes) / 2 else 2


def _latest_snapshot_id(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(text("select id from snapshots order by fetched_at desc limit 1")).scalar()


def _snapshot_id_for_game_date(engine: Engine, game_date: str) -> str | None:
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


def _latest_model_path(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob(pattern))
    return candidates[-1] if candidates else None


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
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _risk_adjusted_confidence(*, p_over: float, n_eff: float | None, status: str) -> float:
    p_pick = max(p_over, 1.0 - p_over)
    rho = 0.85
    if n_eff is None or n_eff < 5:
        rho = 0.65
    if status == "raw":
        rho = min(rho, 0.60)
    elif status == "fallback_global":
        rho = min(rho, 0.75)
    return 0.5 + rho * (p_pick - 0.5)


def _is_supported_stat_type(stat_type: str) -> bool:
    if not stat_type:
        return False
    return (
        stat_components(stat_type) is not None
        or stat_diff_components(stat_type) is not None
        or stat_weighted_components(stat_type) is not None
    )


def _load_projection_frame(engine: Engine, snapshot_id: str, *, include_non_today: bool) -> pd.DataFrame:
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
        params={"snapshot_id": snapshot_id, "include_non_today": bool(include_non_today)},
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
        is_today=bool(_row_get(row, "today")) if _row_get(row, "today") is not None else None,
        is_combo=bool(_row_get(row, "combo") or False),
    )


def score_ensemble(
    engine: Engine,
    *,
    snapshot_id: str | None = None,
    game_date: str | None = None,
    models_dir: str = "models",
    calibration_path: str | None = None,
    ensemble_weights_path: str = "models/ensemble_weights.json",
    min_games: int = 5,
    top: int = 50,
    rank: str = "risk_adj",
    include_non_today: bool = False,
) -> ScoringResult:
    resolved_snapshot = snapshot_id
    if not resolved_snapshot and game_date:
        resolved_snapshot = _snapshot_id_for_game_date(engine, str(game_date))
    resolved_snapshot = resolved_snapshot or _latest_snapshot_id(engine)
    if not resolved_snapshot:
        return ScoringResult(
            snapshot_id="",
            scored_at=datetime.now(timezone.utc).isoformat(),
            total_scored=0,
            picks=[],
        )

    frame = _load_projection_frame(
        engine,
        str(resolved_snapshot),
        include_non_today=include_non_today,
    )
    if frame.empty:
        return ScoringResult(
            snapshot_id=str(resolved_snapshot),
            scored_at=datetime.now(timezone.utc).isoformat(),
            total_scored=0,
            picks=[],
        )

    unsupported = frame["stat_type"].fillna("").astype(str).map(lambda v: not _is_supported_stat_type(v))
    if unsupported.any():
        frame = frame[~unsupported].copy()
        if frame.empty:
            return ScoringResult(
                snapshot_id=str(resolved_snapshot),
                scored_at=datetime.now(timezone.utc).isoformat(),
                total_scored=0,
                picks=[],
            )

    calibration_path = _auto_calibration_path(calibration_path)
    models_path = Path(models_dir)
    nn_path = _latest_model_path(models_path, "nn_gru_attention_*.pt")
    lr_path = _latest_model_path(models_path, "baseline_logreg_*.joblib")
    xgb_path = _latest_model_path(models_path, "xgb_*.joblib")

    # Load conformal calibrators from saved models
    conformal_cals: list[ConformalCalibrator] = []
    for _mp in [lr_path, xgb_path]:
        if _mp and _mp.exists():
            try:
                _pl = _joblib.load(str(_mp))
                _cd = _pl.get("conformal")
                if _cd:
                    conformal_cals.append(ConformalCalibrator(**_cd))
            except Exception:  # noqa: BLE001
                pass
    if nn_path and nn_path.exists():
        try:
            _pl = _torch.load(str(nn_path), map_location="cpu")
            _cd = _pl.get("conformal")
            if _cd:
                conformal_cals.append(ConformalCalibrator(**_cd))
        except Exception:  # noqa: BLE001
            pass

    calibrator = None
    if calibration_path:
        calibrator = ForecastDistributionCalibrator.load(calibration_path)

    # Forecast expert
    logs = load_db_game_logs(engine)
    params = ForecastParams()
    needed_stat_types = sorted({str(v) for v in frame["stat_type"].fillna("").tolist() if str(v)})
    priors = LeaguePriors(logs, stat_types=needed_stat_types, minutes_prior=params.minutes_prior)
    forecast = StatForecastPredictor(
        logs,
        min_games=min_games,
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
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            pass
        else:
            for idx, r in enumerate(nn_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(nn_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_nn[str(proj_id)] = prob

    # LR expert (optional)
    p_lr: dict[str, float] = {}
    if lr_path:
        try:
            lr_inf = infer_lr_over_probs(
                engine=engine,
                model_path=str(lr_path),
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            pass
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
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            pass
        else:
            for idx, r in enumerate(xgb_inf.frame.itertuples(index=False)):
                proj_id = getattr(r, "projection_id", None)
                if proj_id is None:
                    continue
                prob = float(xgb_inf.probs[idx])
                if not math.isfinite(prob):
                    continue
                p_xgb[str(proj_id)] = prob

    experts = ["p_forecast_cal", "p_nn", "p_lr", "p_xgb"]
    if Path(ensemble_weights_path).exists():
        ens = ContextualHedgeEnsembler.load(ensemble_weights_path)
    else:
        ens = ContextualHedgeEnsembler(experts=experts, eta=0.2, shrink_to_uniform=0.01)

    scored: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        proj_id = str(getattr(row, "projection_id", ""))
        stat_type = str(getattr(row, "stat_type", "") or "")
        if not stat_type or not proj_id:
            continue

        f = forecast_map.get(proj_id) or {}
        expert_probs = {
            "p_forecast_cal": _safe_prob(f.get("p_forecast_cal")),
            "p_nn": _safe_prob(p_nn.get(proj_id)),
            "p_lr": _safe_prob(p_lr.get(proj_id)),
            "p_xgb": _safe_prob(p_xgb.get(proj_id)),
        }
        is_live = bool(getattr(row, "is_live", False) or False)
        n_eff = f.get("n_eff")
        try:
            n_eff_val = float(n_eff) if n_eff is not None else None
        except (TypeError, ValueError):
            n_eff_val = None
        ctx = Context(stat_type=stat_type, is_live=is_live, n_eff=n_eff_val)
        p_final = float(ens.predict(expert_probs, ctx))
        if not math.isfinite(p_final):
            continue
        pick = "OVER" if p_final >= 0.5 else "UNDER"
        conf = float(confidence_from_probability(p_final))
        status = str(f.get("calibration_status") or "raw") if f else "raw"
        p_adj = float(_risk_adjusted_confidence(p_over=p_final, n_eff=n_eff_val, status=status))

        score = p_adj
        if rank == "confidence":
            score = conf
        if not math.isfinite(float(score)):
            continue

        scored.append(
            {
                "projection_id": proj_id,
                "player_name": str(getattr(row, "player_name", "") or ""),
                "player_image_url": _optional_str(getattr(row, "player_image_url", None)),
                "player_id": str(getattr(row, "player_id", "") or ""),
                "game_id": getattr(row, "game_id", None),
                "stat_type": stat_type,
                "line_score": float(getattr(row, "line_score", 0.0) or 0.0),
                "pick": pick,
                "prob_over": p_final,
                "confidence": conf,
                "rank_score": float(score),
                "p_forecast_cal": expert_probs["p_forecast_cal"],
                "p_nn": expert_probs["p_nn"],
                "p_lr": expert_probs["p_lr"],
                "p_xgb": expert_probs["p_xgb"],
                "mu_hat": float(f.get("mu_hat") or 0.0) if f else None,
                "sigma_hat": float(f.get("sigma_hat") or 0.0) if f else None,
                "calibration_status": status,
                "n_eff": n_eff_val,
                "conformal_set_size": _conformal_set_size(conformal_cals, p_final),
            }
        )
        item = scored[-1]
        item["edge"] = _compute_edge(p_final, expert_probs, item["conformal_set_size"], n_eff=n_eff_val)
        item["grade"] = _grade_from_edge(item["edge"])

    scored.sort(key=lambda item: item["edge"], reverse=True)
    top_picks = scored[:top]

    picks = [
        ScoredPick(
            projection_id=item["projection_id"],
            player_name=item["player_name"],
            player_image_url=item["player_image_url"],
            player_id=item["player_id"],
            game_id=str(item["game_id"]) if item["game_id"] else None,
            stat_type=item["stat_type"],
            line_score=item["line_score"],
            pick=item["pick"],
            prob_over=item["prob_over"],
            confidence=item["confidence"],
            rank_score=item["rank_score"],
            p_forecast_cal=item["p_forecast_cal"],
            p_nn=item["p_nn"],
            p_lr=item["p_lr"],
            p_xgb=item["p_xgb"],
            mu_hat=item["mu_hat"],
            sigma_hat=item["sigma_hat"],
            calibration_status=item["calibration_status"],
            n_eff=item["n_eff"],
            conformal_set_size=item.get("conformal_set_size"),
            edge=item["edge"],
            grade=item["grade"],
        )
        for item in top_picks
    ]

    return ScoringResult(
        snapshot_id=str(resolved_snapshot),
        scored_at=datetime.now(timezone.utc).isoformat(),
        total_scored=len(scored),
        picks=picks,
    )


def list_snapshots(engine: Engine, *, limit: int = 20) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                select id, fetched_at, data_count, included_count
                from snapshots
                order by fetched_at desc
                limit :limit
                """
            ),
            {"limit": limit},
        ).all()

        return [
            {
                "id": str(row.id),
                "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None,
                "data_count": row.data_count,
                "included_count": row.included_count,
            }
            for row in rows
        ]
