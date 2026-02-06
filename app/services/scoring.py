from __future__ import annotations

import logging
import math
import threading
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch as _torch
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.ml.artifact_store import load_latest_artifact_as_file
from app.ml.artifacts import load_joblib_artifact, latest_compatible_joblib_path
from app.ml.infer_baseline import infer_over_probs as infer_lr_over_probs
from app.ml.lgbm.infer import infer_over_probs as infer_lgbm_over_probs
from app.ml.meta_learner import infer_meta_learner
from app.ml.nn.infer import (
    infer_over_probs as infer_nn_over_probs,
    latest_compatible_checkpoint,
)
from app.ml.tabdl.infer import (
    infer_over_probs as infer_tabdl_over_probs,
    latest_compatible_checkpoint as latest_compatible_tabdl_checkpoint,
)
from app.ml.xgb.infer import infer_over_probs as infer_xgb_over_probs
from app.modeling.conformal import ConformalCalibrator
from app.modeling.db_logs import load_db_game_logs
from app.modeling.forecast_calibration import ForecastDistributionCalibrator
from app.modeling.gating_model import GatingModel, build_context_features
from app.modeling.hybrid_ensemble import HybridEnsembleCombiner
from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler
from app.modeling.probability import confidence_from_probability
from app.modeling.stat_forecast import (
    ForecastParams,
    LeaguePriors,
    StatForecastPredictor,
)
from app.modeling.thompson_ensemble import ThompsonSamplingEnsembler
from app.modeling.types import Projection
from app.ml.stat_mappings import (
    stat_components,
    stat_diff_components,
    stat_weighted_components,
)

# --- Scoring cache ---
_CACHE_TTL_SECONDS = 300  # 5 min
_scoring_cache: dict[str, tuple[float, dict]] = {}
_scoring_cache_lock = threading.Lock()


def _cache_key(snapshot_id: str, top: int, rank: str, include_non_today: bool) -> str:
    return f"{snapshot_id}:{top}:{rank}:{include_non_today}"


def invalidate_scoring_cache() -> None:
    """Clear all cached scoring results."""
    with _scoring_cache_lock:
        _scoring_cache.clear()


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
    p_tabdl: float | None
    p_lr: float | None
    p_xgb: float | None
    p_lgbm: float | None
    p_meta: float | None
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


# --- Probability shrinkage ---
# Real sports betting edges are small.  Even the sharpest models rarely
# exceed 60-65% true probability.  Raw model outputs of 99%+ are almost
# certainly overfit.  We shrink toward a context-aware prior using a
# logit-space blend.
#
# logit(p_final) = (1-k)*logit(p_raw) + k*logit(prior)
# With n_eff >= 30 games of history, k = SHRINK_MIN (modest pull).
# With little data, k = SHRINK_MAX (heavy pull toward prior).
#
# The prior is loaded from models/context_priors.json (refreshed daily).
# It adapts per stat_type and line_score bucket.  Fallback: 0.42.
SHRINK_ANCHOR = 0.42  # Global fallback only -- prefer context prior
SHRINK_MIN = 0.05  # best-data picks: 5% pull (models are already calibrated)
SHRINK_MAX = 0.25  # low-data picks: 25% pull toward base rate

# Abstain policy: only publish picks with p_pick >= threshold
# p_pick = max(p_final, 1 - p_final)
PICK_THRESHOLD = 0.57

# Minimum edge score for a pick to be publishable
MIN_EDGE = 2.0

# Stat types with degenerate base rates where no model can add value.
# These are completely excluded from scoring.
EXCLUDED_STAT_TYPES: set[str] = {"Dunks", "Blocked Shots"}

# Stat types with too few samples or skewed base rates.
# We score them with the context prior only and never publish.
PRIOR_ONLY_STAT_TYPES: set[str] = {
    "Blks+Stls",
    "Offensive Rebounds",
    "Personal Fouls",
    "Steals",
}


def _logit(p: float) -> float:
    """Logit function with clamping to avoid infinities."""
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Sigmoid function with overflow protection."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def shrink_probability(
    p: float,
    n_eff: float | None = None,
    context_prior: float | None = None,
) -> float:
    """Shrink a probability toward prior in logit space based on data quality.

    Uses logit-space blending (more stable than linear probability averaging):
      logit(p_final) = (1-k)*logit(p_raw) + k*logit(prior)
    """
    prior = context_prior if context_prior is not None else SHRINK_ANCHOR
    if n_eff is not None and n_eff > 0:
        k = SHRINK_MAX - (SHRINK_MAX - SHRINK_MIN) * min(1.0, n_eff / 30.0)
    else:
        k = SHRINK_MAX
    logit_p = _logit(p)
    logit_prior = _logit(prior)
    logit_final = (1.0 - k) * logit_p + k * logit_prior
    return _sigmoid(logit_final)


def _compute_edge(
    p_shrunk: float,
    expert_probs: dict[str, float | None],
    conformal_set_size: int | None,
    n_eff: float | None = None,
) -> float:
    """Composite prediction score 0-100 using SHRUNK probability.

    Strict: A+ ≈ top 2%, A ≈ top 10%.  Requires multiple conditions
    to be true simultaneously to score high.
    """
    available = [v for v in expert_probs.values() if v is not None]
    n_experts = len(available)
    pick_over = p_shrunk >= 0.5

    # 1. Directional edge (max 20pts)
    #    Based on shrunk probability distance from 50%.
    #    Power scaling so it's hard to max out.
    raw_pct = abs(p_shrunk - 0.5)  # 0..~0.25
    dir_score = min(20.0, (raw_pct / 0.20) ** 0.6 * 20.0)

    # 2. Qualified expert consensus (max 30pts)
    #    Experts must BOTH agree on direction AND output moderate probs.
    #    An expert at 0.01 agreeing on UNDER gets zero credit — that's
    #    overfit noise, not real signal.
    if n_experts >= 2:
        # "Qualified" = agrees on direction AND prob is in [0.25, 0.75]
        qualified = sum(
            1 for v in available if (v >= 0.5) == pick_over and 0.25 <= v <= 0.75
        )
        # Unqualified dissenters are a red flag
        dissenters = sum(1 for v in available if (v >= 0.5) != pick_over)
        # Need at least 2 qualified experts; penalize heavily for dissenters
        q_ratio = max(0.0, qualified - dissenters * 2.0) / n_experts
        # Require ≥ 3 experts for full marks
        coverage = min(1.0, n_experts / 3.0)
        consensus_score = q_ratio * coverage * 30.0
    else:
        consensus_score = 0.0

    # 3. Expert tightness (max 15pts)
    #    How tightly clustered are the qualified experts?
    #    Uses only experts in the pick direction with moderate outputs.
    if n_experts >= 2:
        qualified_probs = [
            v for v in available if (v >= 0.5) == pick_over and 0.25 <= v <= 0.75
        ]
        if len(qualified_probs) >= 2:
            spread = max(qualified_probs) - min(qualified_probs)
            tightness = max(0.0, 1.0 - spread / 0.20)  # 0.20 spread → 0
            tight_score = tightness * 15.0
        else:
            tight_score = 0.0
    else:
        tight_score = 0.0

    # 4. Data quality (max 15pts) — log scaling on n_eff
    if n_eff is not None and n_eff > 0:
        import math as _math

        data_q = min(1.0, _math.log1p(n_eff) / _math.log1p(30.0))
    else:
        data_q = 0.0
    data_score = data_q * 15.0

    # 5. Conformal (max 10pts)
    conf_score = 10.0 if conformal_set_size == 1 else 0.0

    # 6. Anti-overconfidence penalty (max -10pts)
    #    Penalize when ANY expert outputs extreme prob (>90% or <10%).
    #    These are almost certainly overfit.
    if available:
        extreme = sum(1 for v in available if v > 0.90 or v < 0.10)
        penalty = min(10.0, extreme * 5.0)
    else:
        penalty = 0.0

    edge = dir_score + consensus_score + tight_score + data_score + conf_score - penalty
    return round(min(100.0, max(0.0, edge)), 1)


def _grade_from_edge(edge: float) -> str:
    if edge >= 75:
        return "A+"
    if edge >= 60:
        return "A"
    if edge >= 45:
        return "B"
    if edge >= 30:
        return "C"
    if edge >= 18:
        return "D"
    return "F"


def _conformal_set_size(
    calibrators: list[ConformalCalibrator], p_over: float
) -> int | None:
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
        return conn.execute(
            text("select id from snapshots order by fetched_at desc limit 1")
        ).scalar()


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


def _nn_weight_cap_for_latest_auc(engine: Engine) -> dict[str, float]:
    try:
        with engine.connect() as conn:
            nn_auc = conn.execute(
                text(
                    "select (metrics->>'roc_auc')::float from model_runs "
                    "where model_name = 'nn_gru_attention' and metrics->>'roc_auc' is not null "
                    "order by created_at desc limit 1"
                )
            ).scalar()
            if nn_auc is not None:
                value = float(nn_auc)
                if value >= 0.60:
                    return {}
                if value >= 0.55:
                    return {"p_nn": 0.20}
    except Exception:  # noqa: BLE001
        pass
    return {"p_nn": 0.10}


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


def _ensure_thompson_experts(
    ts: ThompsonSamplingEnsembler, experts: list[str]
) -> ThompsonSamplingEnsembler:
    if ts.experts == experts:
        return ts
    ts.experts = list(experts)
    ctx_keys = set(ts.alpha.keys()) | set(ts.beta.keys())
    for ctx_key in ctx_keys:
        alpha = ts.alpha.setdefault(ctx_key, {})
        beta = ts.beta.setdefault(ctx_key, {})
        for expert in experts:
            alpha.setdefault(expert, float(ts.prior_alpha))
            beta.setdefault(expert, float(ts.prior_beta))
    return ts


def _is_supported_stat_type(stat_type: str) -> bool:
    if not stat_type:
        return False
    return (
        stat_components(stat_type) is not None
        or stat_diff_components(stat_type) is not None
        or stat_weighted_components(stat_type) is not None
    )


def _load_projection_frame(
    engine: Engine, snapshot_id: str, *, include_non_today: bool
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
          and (:include_non_today = true or coalesce(pf.today, true) = true)
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
    force: bool = False,
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

    ck = _cache_key(str(resolved_snapshot), top, rank, include_non_today)
    if not force:
        with _scoring_cache_lock:
            cached = _scoring_cache.get(ck)
        if cached is not None:
            ts, cached_dict = cached
            if (_time.monotonic() - ts) < _CACHE_TTL_SECONDS:
                return ScoringResult(
                    snapshot_id=cached_dict["snapshot_id"],
                    scored_at=cached_dict["scored_at"],
                    total_scored=cached_dict["total_scored"],
                    picks=[ScoredPick(**p) for p in cached_dict["picks"]],
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

    unsupported = (
        frame["stat_type"]
        .fillna("")
        .astype(str)
        .map(lambda v: not _is_supported_stat_type(v))
    )
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

    _use_db = settings.model_source.lower() == "db"
    if _use_db:
        nn_path = load_latest_artifact_as_file(engine, "nn_gru_attention", suffix=".pt")
        tabdl_path = load_latest_artifact_as_file(engine, "tabdl_mlp", suffix=".pt")
        lr_path = load_latest_artifact_as_file(
            engine, "baseline_logreg", suffix=".joblib"
        )
        xgb_path = load_latest_artifact_as_file(engine, "xgb", suffix=".joblib")
        lgbm_path = load_latest_artifact_as_file(engine, "lgbm", suffix=".joblib")
        meta_path = load_latest_artifact_as_file(
            engine, "meta_learner", suffix=".joblib"
        )
    else:
        models_path = Path(models_dir)
        nn_path = latest_compatible_checkpoint(models_path, "nn_gru_attention_*.pt")
        tabdl_path = latest_compatible_tabdl_checkpoint(models_path, "tabdl_*.pt")
        lr_path = latest_compatible_joblib_path(models_path, "baseline_logreg_*.joblib")
        xgb_path = latest_compatible_joblib_path(models_path, "xgb_*.joblib")
        lgbm_path = latest_compatible_joblib_path(models_path, "lgbm_*.joblib")
        meta_path = latest_compatible_joblib_path(models_path, "meta_learner_*.joblib")

    logger.info(
        "Model paths resolved (db=%s): nn=%s tabdl=%s lr=%s xgb=%s lgbm=%s meta=%s",
        _use_db,
        nn_path,
        tabdl_path,
        lr_path,
        xgb_path,
        lgbm_path,
        meta_path,
    )

    # Load conformal calibrators from saved models
    conformal_cals: list[ConformalCalibrator] = []
    for _mp in [lr_path, xgb_path, lgbm_path]:
        if _mp and _mp.exists():
            try:
                _pl = load_joblib_artifact(str(_mp), strict_sklearn_version=False)
                _cd = _pl.get("conformal")
                if _cd:
                    conformal_cals.append(ConformalCalibrator(**_cd))
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load conformal from %s", _mp, exc_info=True)
    if nn_path and nn_path.exists():
        try:
            _pl = _torch.load(str(nn_path), map_location="cpu")
            _cd = _pl.get("conformal")
            if _cd:
                conformal_cals.append(ConformalCalibrator(**_cd))
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load conformal from NN %s", nn_path, exc_info=True
            )
    if tabdl_path and tabdl_path.exists():
        try:
            _pl = _torch.load(str(tabdl_path), map_location="cpu")
            _cd = _pl.get("conformal")
            if _cd:
                conformal_cals.append(ConformalCalibrator(**_cd))
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load conformal from TabDL %s", tabdl_path, exc_info=True
            )

    calibrator = None
    if calibration_path:
        calibrator = ForecastDistributionCalibrator.load(calibration_path)

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
            logger.warning(
                "NN inference failed for snapshot %s", resolved_snapshot, exc_info=True
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
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "TabDL inference failed for snapshot %s",
                resolved_snapshot,
                exc_info=True,
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
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "LR inference failed for snapshot %s", resolved_snapshot, exc_info=True
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
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "XGB inference failed for snapshot %s", resolved_snapshot, exc_info=True
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
                snapshot_id=str(resolved_snapshot),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "LGBM inference failed for snapshot %s",
                resolved_snapshot,
                exc_info=True,
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

    experts = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]
    nn_cap = _nn_weight_cap_for_latest_auc(engine)

    _ens_path: Path | str | None = None
    _ts_path: Path | str | None = None
    _gating_path: Path | str | None = None
    if _use_db:
        _ens_path = load_latest_artifact_as_file(
            engine, "ensemble_weights", suffix=".json"
        )
        _ts_path = load_latest_artifact_as_file(
            engine, "thompson_weights", suffix=".json"
        )
        _gating_path = load_latest_artifact_as_file(
            engine, "gating_model", suffix=".joblib"
        )
    else:
        _ens_path = (
            Path(ensemble_weights_path)
            if Path(ensemble_weights_path).exists()
            else None
        )
        _ts_candidate = Path(models_dir) / "thompson_weights.json"
        _ts_path = _ts_candidate if _ts_candidate.exists() else None
        _gating_candidate = Path(models_dir) / "gating_model.joblib"
        _gating_path = _gating_candidate if _gating_candidate.exists() else None

    if _ens_path:
        ens = _ensure_ensemble_experts(
            ContextualHedgeEnsembler.load(str(_ens_path)), experts
        )
        if not ens.max_weight:
            ens.max_weight = nn_cap
    else:
        ens = ContextualHedgeEnsembler(
            experts=experts, eta=0.2, shrink_to_uniform=0.01, max_weight=nn_cap
        )

    # Load Thompson Sampling ensemble (optional)
    thompson: ThompsonSamplingEnsembler | None = None
    if _ts_path:
        try:
            thompson = _ensure_thompson_experts(
                ThompsonSamplingEnsembler.load(str(_ts_path)), experts
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load Thompson ensemble", exc_info=True)

    # Load Gating Model (optional)
    gating: GatingModel | None = None
    if _gating_path:
        try:
            gating = GatingModel.load(str(_gating_path))
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load Gating model", exc_info=True)

    # Build hybrid combiner if Thompson or Gating available
    hybrid: HybridEnsembleCombiner | None = None
    if thompson is not None or gating is not None:
        hybrid = HybridEnsembleCombiner.from_components(
            thompson=thompson,
            gating=gating,
            experts=experts,
        )

    # Load context-aware priors for shrinkage (daily-refreshed from resolved data)
    from app.ml.context_prior import get_context_prior, load_context_priors
    from app.ml.stat_calibrator import StatTypeCalibrator

    _context_priors = load_context_priors()
    _stat_calibrator = StatTypeCalibrator.load()

    scored: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        proj_id = str(getattr(row, "projection_id", ""))
        stat_type = str(getattr(row, "stat_type", "") or "")
        if not stat_type or not proj_id:
            continue

        # Phase 2: skip degenerate stat types entirely
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
        is_live = bool(getattr(row, "is_live", False) or False)
        n_eff = f.get("n_eff")
        try:
            n_eff_val = float(n_eff) if n_eff is not None else None
        except (TypeError, ValueError):
            n_eff_val = None
        # Meta-learner: blends base experts with context
        p_meta_val: float | None = None
        if meta_path:
            try:
                p_meta_val = infer_meta_learner(
                    model_path=str(meta_path),
                    expert_probs=expert_probs,
                    n_eff=n_eff_val,
                )
            except Exception:  # noqa: BLE001
                if not hasattr(score_ensemble, "_meta_warned"):
                    logger.warning("Meta-learner inference failed", exc_info=True)
                    score_ensemble._meta_warned = True  # type: ignore[attr-defined]
        ctx = Context(stat_type=stat_type, is_live=is_live, n_eff=n_eff_val)
        # Primary: hybrid combiner (Thompson + Gating + Meta)
        # Fallback chain: hybrid → meta-learner → Hedge
        p_raw = None
        if hybrid is not None:
            try:
                # Build context features for gating model
                avail = {k: v for k, v in expert_probs.items() if v is not None}
                ctx_feats = None
                if avail and gating is not None and gating.is_fitted:
                    ctx_feats = build_context_features(
                        {k: np.array([v]) for k, v in avail.items()},
                        n_eff=np.array([n_eff_val or 0.0]),
                    )[0]
                ctx_tuple = (
                    stat_type,
                    "live" if is_live else "pregame",
                    "high" if (n_eff_val or 0) >= 15 else "low",
                )
                p_hybrid = hybrid.predict(
                    expert_probs,
                    ctx_tuple,
                    context_features=ctx_feats,
                    p_meta=p_meta_val
                    if p_meta_val is not None and math.isfinite(p_meta_val)
                    else None,
                    n_eff=n_eff_val,
                )
                if math.isfinite(p_hybrid):
                    p_raw = p_hybrid
            except Exception:  # noqa: BLE001
                if not hasattr(score_ensemble, "_hybrid_warned"):
                    logger.warning("Hybrid ensemble prediction failed", exc_info=True)
                    score_ensemble._hybrid_warned = True  # type: ignore[attr-defined]
        if p_raw is None and p_meta_val is not None and math.isfinite(p_meta_val):
            p_raw = p_meta_val
        if p_raw is None:
            p_raw = float(ens.predict(expert_probs, ctx))
        if not math.isfinite(p_raw):
            continue
        # Shrink toward context-aware prior in logit space.
        _line_score = float(getattr(row, "line_score", 0.0) or 0.0)
        _ctx_prior = get_context_prior(
            _context_priors, stat_type=stat_type, line_score=_line_score
        )

        # Prior-only stat types: use context prior directly, never publish
        _is_prior_only = stat_type in PRIOR_ONLY_STAT_TYPES
        if _is_prior_only:
            p_final = _ctx_prior if _ctx_prior is not None else 0.5
            p_raw = p_final
        else:
            p_final = shrink_probability(
                p_raw, n_eff=n_eff_val, context_prior=_ctx_prior
            )
            # Apply per-stat-type isotonic recalibration
            p_final = _stat_calibrator.transform(p_final, stat_type)

        pick = "OVER" if p_final >= 0.5 else "UNDER"
        conf = float(confidence_from_probability(p_final))
        status = str(f.get("calibration_status") or "raw") if f else "raw"
        p_adj = float(
            _risk_adjusted_confidence(p_over=p_final, n_eff=n_eff_val, status=status)
        )

        score = p_adj
        if rank == "confidence":
            score = conf
        if not math.isfinite(float(score)):
            continue

        conformal_size = _conformal_set_size(conformal_cals, p_final)
        p_pick = max(p_final, 1.0 - p_final)
        # Publishable: meets confidence threshold, has narrow conformal set,
        # has minimum edge, and is not a prior-only stat type
        is_publishable = bool(
            p_pick >= PICK_THRESHOLD and conformal_size != 2 and not _is_prior_only
        )

        scored.append(
            {
                "projection_id": proj_id,
                "player_name": str(getattr(row, "player_name", "") or ""),
                "player_image_url": _optional_str(
                    getattr(row, "player_image_url", None)
                ),
                "player_id": str(getattr(row, "player_id", "") or ""),
                "game_id": getattr(row, "game_id", None),
                "stat_type": stat_type,
                "line_score": float(getattr(row, "line_score", 0.0) or 0.0),
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
                "n_eff": n_eff_val,
                "conformal_set_size": conformal_size,
            }
        )
        item = scored[-1]
        item["edge"] = _compute_edge(
            p_final, expert_probs, item["conformal_set_size"], n_eff=n_eff_val
        )
        item["grade"] = _grade_from_edge(item["edge"])
        # Apply min_edge filter
        if item["edge"] < MIN_EDGE:
            is_publishable = False
        item["is_publishable"] = is_publishable

    scored.sort(key=lambda item: item["edge"], reverse=True)
    # Enforce abstain policy: only return publishable picks in the top-N
    publishable = [item for item in scored if item.get("is_publishable", False)]
    top_picks = publishable[:top]

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
            p_tabdl=item["p_tabdl"],
            p_lr=item["p_lr"],
            p_xgb=item["p_xgb"],
            p_lgbm=item["p_lgbm"],
            p_meta=item["p_meta"],
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

    result = ScoringResult(
        snapshot_id=str(resolved_snapshot),
        scored_at=datetime.now(timezone.utc).isoformat(),
        total_scored=len(scored),
        picks=picks,
    )
    with _scoring_cache_lock:
        _scoring_cache[ck] = (_time.monotonic(), result.to_dict())
    return result


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
