from __future__ import annotations

import logging
import math
import threading
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)

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
# It adapts per stat_type and line_score bucket.  Fallback: 0.50 (neutral).
SHRINK_ANCHOR = 0.50  # Neutral fallback — context priors handle empirical rates
SHRINK_MIN = 0.05  # best-data picks: 5% pull (models are already calibrated)
SHRINK_MAX = 0.15  # low-data picks: 15% pull toward prior (was 0.25)

# Abstain policy: only publish picks with p_pick >= threshold
# p_pick = max(p_final, 1 - p_final)
PICK_THRESHOLD = 0.57

# Minimum edge score for a pick to be publishable
MIN_EDGE = 2.0

# Minimum effective sample size for publishability.
# n_eff < 10 has 30-47% accuracy — guaranteed losers.
MIN_NEFF = 10.0

# Maximum |mu_hat - line| for publishability.
# Forecast edges > 4 points have 47% accuracy — overconfident.
MAX_FORECAST_EDGE = 4.0

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
    context_prior: float | None = None,
    mu_hat: float | None = None,
    line_score: float | None = None,
    sigma_hat: float | None = None,
) -> float:
    """Composite prediction score 0-100 (v2: data-driven).

    Redesigned based on analysis of 9,952 resolved predictions:
    - Expert split decisions (34-50% agreement) had 61.3% accuracy
    - Full consensus (84%+) had only 51.8%
    - |mu-line| 2-3 was the sweet spot (55.4%)
    - n_eff 10-20 was best (53%)
    - Confidence 60-65% was best (55.2%)
    - Higher confidence was WORSE (75%+ → 45.9%)
    """
    available = [v for v in expert_probs.values() if v is not None]
    n_experts = len(available)
    pick_over = p_shrunk >= 0.5

    # 1. Forecast edge sweet spot (max 25pts)
    #    |mu-line| of 2-3 is peak accuracy.  Below 1: weak signal.
    #    Above 4: overconfident and unreliable.
    fc_edge_score = 0.0
    if mu_hat is not None and line_score is not None:
        abs_edge = abs(float(mu_hat) - float(line_score))
        if abs_edge <= 1.0:
            fc_edge_score = abs_edge * 8.0  # 0→8
        elif abs_edge <= 3.0:
            fc_edge_score = 8.0 + (abs_edge - 1.0) / 2.0 * 17.0  # 8→25
        elif abs_edge <= 4.0:
            fc_edge_score = 25.0 - (abs_edge - 3.0) * 15.0  # 25→10
        else:
            fc_edge_score = max(0.0, 10.0 - (abs_edge - 4.0) * 5.0)  # 10→0

    # 2. Data quality sweet spot (max 20pts)
    #    n_eff 10-20 is peak.  Below 10: unreliable.  Above 25: slight drop.
    data_score = 0.0
    if n_eff is not None:
        if n_eff < 5:
            data_score = 0.0
        elif n_eff < 10:
            data_score = (n_eff - 5) / 5.0 * 8.0  # 0→8
        elif n_eff <= 20:
            data_score = 20.0  # full marks
        elif n_eff <= 30:
            data_score = 20.0 - (n_eff - 20) / 10.0 * 5.0  # 20→15
        else:
            data_score = 15.0

    # 3. Expert disagreement signal (max 25pts)
    #    Empirically, split decisions (34-50% agreement) have 61.3% accuracy
    #    while full consensus (84%+) has only 51.8%.  This rewards picks
    #    where the minority disagrees — suggesting the model found a genuine
    #    edge that not all models see.
    disagree_score = 0.0
    if n_experts >= 2:
        agree_count = sum(1 for v in available if (v >= 0.5) == pick_over)
        agree_pct = agree_count / n_experts
        if agree_pct <= 0.5:
            disagree_score = 25.0  # split or minority pick → max
        elif agree_pct <= 0.67:
            disagree_score = 25.0 - (agree_pct - 0.5) / 0.17 * 10.0  # 25→15
        elif agree_pct <= 0.84:
            disagree_score = 15.0 - (agree_pct - 0.67) / 0.17 * 5.0  # 15→10
        else:
            disagree_score = 10.0 - (agree_pct - 0.84) / 0.16 * 5.0  # 10→5

    # 4. Confidence sweet spot (max 15pts)
    #    60-65% is peak (55.2% accuracy).  Higher is worse, not better.
    conf_val = max(p_shrunk, 1.0 - p_shrunk)
    if conf_val <= 0.55:
        conf_score = (conf_val - 0.50) / 0.05 * 5.0  # 0→5
    elif conf_val <= 0.65:
        conf_score = 5.0 + (conf_val - 0.55) / 0.10 * 10.0  # 5→15
    elif conf_val <= 0.75:
        conf_score = 15.0 - (conf_val - 0.65) / 0.10 * 10.0  # 15→5
    else:
        conf_score = max(0.0, 5.0 - (conf_val - 0.75) / 0.25 * 5.0)  # 5→0

    # 5. Uncertainty sweet spot (max 10pts)
    #    sigma 4-6 is peak (54.6%).  Very low or high uncertainty = worse.
    sigma_score = 0.0
    if sigma_hat is not None:
        s = float(sigma_hat)
        if s <= 2.0:
            sigma_score = s / 2.0 * 4.0  # 0→4
        elif s <= 6.0:
            sigma_score = 4.0 + (s - 2.0) / 4.0 * 6.0  # 4→10
        elif s <= 8.0:
            sigma_score = 10.0 - (s - 6.0) / 2.0 * 4.0  # 10→6
        else:
            sigma_score = max(0.0, 6.0 - (s - 8.0))  # 6→0

    # 6. Conformal (bonus 5pts)
    conf_bonus = 5.0 if conformal_set_size == 1 else 0.0

    edge = (
        fc_edge_score
        + data_score
        + disagree_score
        + conf_score
        + sigma_score
        + conf_bonus
    )
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
    calibrators_by_expert: dict[str, ConformalCalibrator],
    expert_probs: dict[str, float | None],
) -> int | None:
    """Majority-vote conformal set size across calibrated expert probabilities."""
    sizes: list[int] = []
    for expert, calibrator in calibrators_by_expert.items():
        p = expert_probs.get(expert)
        if p is None:
            continue
        sizes.append(int(calibrator.predict(float(p)).set_size))
    if not sizes:
        return None
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


# Clip expert probabilities to prevent logit-space outlier domination.
# In logit space, extreme values (e.g. 0.08 → logit=-2.44) have outsized
# influence vs moderate values (0.54 → logit=+0.16).  A single degenerate
# expert at 8% can outweigh three reasonable experts at 54%.
# Clipping to [0.15, 0.85] bounds any expert's logit influence to ±1.73.
_EXPERT_PROB_FLOOR = 0.15
_EXPERT_PROB_CEIL = 0.85


def _clip_expert_probs(
    expert_probs: dict[str, float | None],
) -> dict[str, float | None]:
    """Clip expert probabilities to a sane range before ensemble combination."""
    return {
        k: max(_EXPERT_PROB_FLOOR, min(_EXPERT_PROB_CEIL, v)) if v is not None else None
        for k, v in expert_probs.items()
    }


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
    """Determine NN weight cap based on best recent AUC.

    Uses MAX AUC from the last 5 NN training runs to avoid penalizing the NN
    when recursive training uploads multiple rounds (where the last round
    may not be the best).
    """
    try:
        with engine.connect() as conn:
            nn_auc = conn.execute(
                text(
                    "select max((metrics->>'roc_auc')::float) from ("
                    "  select metrics from model_runs "
                    "  where model_name = 'nn_gru_attention' "
                    "    and metrics->>'roc_auc' is not null "
                    "  order by created_at desc limit 5"
                    ") sub"
                )
            ).scalar()
            if nn_auc is not None:
                value = float(nn_auc)
                logger.info("NN best recent AUC: %.4f", value)
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

    # Load conformal calibrators keyed by expert so each calibrator is applied
    # to the distribution it was calibrated on.
    conformal_by_expert: dict[str, ConformalCalibrator] = {}
    _conformal_joblib_paths = [
        ("p_lr", lr_path),
        ("p_xgb", xgb_path),
        ("p_lgbm", lgbm_path),
    ]
    for expert_key, model_path in _conformal_joblib_paths:
        if model_path and model_path.exists():
            try:
                payload = load_joblib_artifact(str(model_path))
                conformal_data = payload.get("conformal")
                if conformal_data:
                    conformal_by_expert[expert_key] = ConformalCalibrator(
                        **conformal_data
                    )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to load conformal from %s", model_path, exc_info=True
                )
    if nn_path and nn_path.exists():
        try:
            payload = _torch.load(str(nn_path), map_location="cpu")
            conformal_data = payload.get("conformal")
            if conformal_data:
                conformal_by_expert["p_nn"] = ConformalCalibrator(**conformal_data)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load conformal from NN %s", nn_path, exc_info=True
            )
    if tabdl_path and tabdl_path.exists():
        try:
            payload = _torch.load(str(tabdl_path), map_location="cpu")
            conformal_data = payload.get("conformal")
            if conformal_data:
                conformal_by_expert["p_tabdl"] = ConformalCalibrator(**conformal_data)
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
        # Load optimized mixing weights if available
        if _use_db:
            _mix_path = load_latest_artifact_as_file(
                engine, "hybrid_mixing", suffix=".json"
            )
        else:
            _mix_candidate = Path(models_dir) / "hybrid_mixing.json"
            _mix_path = _mix_candidate if _mix_candidate.exists() else None
        if _mix_path:
            try:
                hybrid.load_mixing(str(_mix_path))
                logger.info(
                    "Hybrid mixing: alpha=%.2f beta=%.2f gamma=%.2f",
                    hybrid.alpha,
                    hybrid.beta,
                    hybrid.gamma,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load hybrid mixing weights", exc_info=True)

    # Load context-aware priors for shrinkage (daily-refreshed from resolved data)
    from app.ml.context_prior import get_context_prior, load_context_priors
    from app.ml.stat_calibrator import StatTypeCalibrator

    if _use_db:
        _cp_path = load_latest_artifact_as_file(
            engine, "context_priors", suffix=".json"
        )
        _context_priors = (
            load_context_priors(str(_cp_path)) if _cp_path else load_context_priors()
        )
        _sc_path = load_latest_artifact_as_file(
            engine, "stat_calibrator", suffix=".joblib"
        )
        _stat_calibrator = (
            StatTypeCalibrator.load(str(_sc_path))
            if _sc_path
            else StatTypeCalibrator.load()
        )
    else:
        _context_priors = load_context_priors()
        _stat_calibrator = StatTypeCalibrator.load()

    # --- Stat-type expert routing ---
    # For stat types where one expert clearly outperforms the ensemble,
    # use that expert's probability directly instead of ensembling.
    _expert_routing: dict[str, str] = {}
    try:
        _routing_path = Path("data/stat_expert_routing.json")
        if _use_db:
            _routing_artifact = load_latest_artifact_as_file(
                engine, "stat_expert_routing", suffix=".json"
            )
            if _routing_artifact:
                _routing_path = Path(str(_routing_artifact))
        if _routing_path.exists():
            import json as _json
            _routing_raw = _json.loads(_routing_path.read_text(encoding="utf-8"))
            _expert_routing = {
                k: v for k, v in _routing_raw.items()
                if not k.startswith("_") and isinstance(v, str)
            }
            if _expert_routing:
                logger.info(
                    "Stat-type expert routing active for %d stat types: %s",
                    len(_expert_routing),
                    ", ".join(f"{k}->{v}" for k, v in _expert_routing.items()),
                )
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load expert routing", exc_info=True)

    # --- Inversion corrections ---
    # Some experts are anti-predictive (accuracy < 50%, improves when flipped).
    # Load flags from model_health report and flip those experts' probabilities.
    from app.ml.inversion_corrections import load_inversion_flags

    _inversion_flags: dict[str, bool] = {}
    try:
        _inversion_flags = load_inversion_flags(engine if _use_db else None)
        if _inversion_flags:
            logger.info(
                "Inversion corrections active for: %s",
                ", ".join(k for k, v in _inversion_flags.items() if v),
            )
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load inversion flags", exc_info=True)

    # --- Artifact availability diagnostic ---
    _n_stat_cals = len(_stat_calibrator.calibrators) if _stat_calibrator else 0
    _n_ctx_stats = len(_context_priors.get("stat_type_priors", {}))
    _has_hybrid_mix = hybrid is not None
    _hybrid_info = (
        f"alpha={hybrid.alpha:.2f} beta={hybrid.beta:.2f} gamma={hybrid.gamma:.2f}"
        if hybrid
        else "N/A"
    )
    logger.info(
        "Scoring artifact status: stat_calibrator=%d stat-types, "
        "context_priors=%d stat-types, hybrid=%s (%s), thompson=%s, gating=%s",
        _n_stat_cals,
        _n_ctx_stats,
        _has_hybrid_mix,
        _hybrid_info,
        thompson is not None,
        gating is not None and gating.is_fitted,
    )
    if _n_stat_cals == 0:
        logger.warning(
            "StatTypeCalibrator has NO per-stat calibrators — "
            "isotonic recalibration is pass-through. "
            "Ensure stat_calibrator artifact is uploaded to DB."
        )
    if _n_ctx_stats == 0:
        logger.warning(
            "Context priors are EMPTY — shrinkage will use neutral 0.50 fallback. "
            "Ensure context_priors artifact is uploaded to DB."
        )

    def _apply_inversions(ep: dict[str, float | None]) -> dict[str, float | None]:
        """Flip expert probs where inversion flags are set."""
        if not _inversion_flags:
            return ep
        for k in ep:
            if ep[k] is not None and _inversion_flags.get(k):
                ep[k] = 1.0 - ep[k]
        return ep

    # --- Batch-level expert debiasing ---
    # If all experts systematically lean one direction, shift them toward 0.5
    # while preserving relative ordering (which IS the real signal).
    _DEBIAS_THRESHOLD = 0.05  # trigger when batch mean deviates >5pp from 0.5
    _all_expert_vals: list[float] = []
    for _row in frame.itertuples(index=False):
        _pid = str(getattr(_row, "projection_id", ""))
        _st = str(getattr(_row, "stat_type", "") or "")
        if not _st or not _pid or _st in EXCLUDED_STAT_TYPES:
            continue
        _f = forecast_map.get(_pid) or {}
        _ep = {
            "p_forecast_cal": _safe_prob(_f.get("p_forecast_cal")),
            "p_nn": _safe_prob(p_nn.get(_pid)),
            "p_tabdl": _safe_prob(p_tabdl.get(_pid)),
            "p_lr": _safe_prob(p_lr.get(_pid)),
            "p_xgb": _safe_prob(p_xgb.get(_pid)),
            "p_lgbm": _safe_prob(p_lgbm.get(_pid)),
        }
        _ep = _apply_inversions(_ep)
        for _v in _ep.values():
            if _v is not None:
                _all_expert_vals.append(_v)

    _expert_shift = 0.0
    if _all_expert_vals:
        _batch_mean = sum(_all_expert_vals) / len(_all_expert_vals)
        if abs(_batch_mean - 0.5) > _DEBIAS_THRESHOLD:
            _expert_shift = 0.5 - _batch_mean
            logger.info(
                "Expert debiasing: batch mean P(over)=%.3f, applying shift=%+.3f "
                "(%d expert values across %d props)",
                _batch_mean,
                _expert_shift,
                len(_all_expert_vals),
                len(frame),
            )

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
        # Apply inversion corrections for anti-predictive experts
        expert_probs = _apply_inversions(expert_probs)
        # Apply batch-level debiasing shift
        if _expert_shift != 0.0:
            for _ek in expert_probs:
                if expert_probs[_ek] is not None:
                    expert_probs[_ek] = max(0.01, min(0.99, expert_probs[_ek] + _expert_shift))
        # Clip to prevent logit-space outlier domination (e.g. TabDL at 8%)
        expert_probs = _clip_expert_probs(expert_probs)

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
        # Stat-type expert routing: if a preferred expert exists for this stat
        # type and has a valid probability, use it directly as p_raw.
        p_raw = None
        _routed_expert = _expert_routing.get(stat_type)
        if _routed_expert and expert_probs.get(_routed_expert) is not None:
            p_raw = expert_probs[_routed_expert]

        # Primary: hybrid combiner (Thompson + Gating + Meta)
        # Fallback chain: routing → hybrid → meta-learner → Hedge
        if p_raw is None and hybrid is not None:
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

        conformal_size = _conformal_set_size(conformal_by_expert, expert_probs)
        p_pick = max(p_final, 1.0 - p_final)
        # Expert diversity: require at least 1 expert on each side of 0.5.
        # If all experts agree on one direction, the model is just echoing
        # the base rate, not providing a differentiated signal.
        _avail_eps = [v for v in expert_probs.values() if v is not None]
        _n_over_exp = sum(1 for v in _avail_eps if v >= 0.5)
        _n_under_exp = sum(1 for v in _avail_eps if v < 0.5)
        _has_diversity = min(_n_over_exp, _n_under_exp) >= 1
        # Data quality filters
        _has_min_neff = n_eff_val is not None and n_eff_val >= MIN_NEFF
        _mu_hat_val = float(f.get("mu_hat") or 0.0) if f else None
        _sigma_hat_val = float(f.get("sigma_hat") or 0.0) if f else None
        _forecast_edge_ok = True
        if _mu_hat_val is not None and _line_score > 0:
            _forecast_edge_ok = abs(_mu_hat_val - _line_score) <= MAX_FORECAST_EDGE
        # Publishable: meets all quality gates
        is_publishable = bool(
            p_pick >= PICK_THRESHOLD
            and conformal_size != 2
            and not _is_prior_only
            and _has_diversity
            and _has_min_neff
            and _forecast_edge_ok
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
            p_final, expert_probs, item["conformal_set_size"],
            n_eff=n_eff_val, context_prior=_ctx_prior,
            mu_hat=_mu_hat_val, line_score=_line_score,
            sigma_hat=_sigma_hat_val,
        )
        item["grade"] = _grade_from_edge(item["edge"])
        # Apply min_edge filter
        if item["edge"] < MIN_EDGE:
            is_publishable = False
        item["is_publishable"] = is_publishable

    # --- Spread-based abstention ---
    # If all p_final values are in a narrow band, the model lacks discrimination
    # and is just predicting the base rate for every prop. Abstain entirely.
    _MIN_SPREAD = 0.10
    if scored:
        _all_p_final = [item["prob_over"] for item in scored]
        _p_spread = max(_all_p_final) - min(_all_p_final)
        if _p_spread < _MIN_SPREAD:
            logger.warning(
                "Model discrimination too low: p_final spread=%.3f (min=%.3f). "
                "Marking all picks non-publishable.",
                _p_spread,
                _MIN_SPREAD,
            )
            for item in scored:
                item["is_publishable"] = False

    scored.sort(key=lambda item: item["edge"], reverse=True)
    # Enforce abstain policy: only return publishable picks in the top-N
    publishable = [item for item in scored if item.get("is_publishable", False)]
    top_picks = publishable[:top]

    # --- Direction balance guardrail ---
    # When >75% of picks lean one direction, apply a soft correction:
    # re-score with a nudged context prior to reduce the imbalance.
    _IMBALANCE_THRESHOLD = 0.75  # trigger correction above this ratio
    _IMBALANCE_PRIOR_NUDGE = 0.50  # push context prior toward neutral
    if top_picks:
        n_over = sum(1 for item in top_picks if item["pick"] == "OVER")
        n_under = len(top_picks) - n_over
        pct_under = n_under / len(top_picks) * 100
        pct_over = n_over / len(top_picks) * 100
        logger.info(
            "Pick direction balance: %d OVER (%.0f%%) / %d UNDER (%.0f%%) out of %d published picks",
            n_over,
            pct_over,
            n_under,
            pct_under,
            len(top_picks),
        )
        dominant_pct = max(pct_over, pct_under) / 100.0
        if dominant_pct > _IMBALANCE_THRESHOLD:
            dominant_dir = "OVER" if pct_over > pct_under else "UNDER"
            logger.warning(
                "Direction imbalance guardrail triggered: %.0f%% of picks are %s. "
                "Demoting edge scores for %s picks that follow the base rate.",
                dominant_pct * 100,
                dominant_dir,
                dominant_dir,
            )
            # Demote edge for picks in the over-represented direction whose
            # probability is close to the context prior (i.e. not adding value).
            for item in top_picks:
                if item["pick"] == dominant_dir:
                    st = item["stat_type"]
                    ls = item["line_score"]
                    cp = get_context_prior(
                        _context_priors, stat_type=st, line_score=ls
                    )
                    # Penalize more aggressively during imbalance
                    item["edge"] = _compute_edge(
                        item["prob_over"],
                        {
                            k: item.get(k)
                            for k in [
                                "p_forecast_cal", "p_nn", "p_tabdl",
                                "p_lr", "p_xgb", "p_lgbm",
                            ]
                        },
                        item["conformal_set_size"],
                        n_eff=item["n_eff"],
                        context_prior=_IMBALANCE_PRIOR_NUDGE,
                        mu_hat=item.get("mu_hat"),
                        line_score=item.get("line_score"),
                        sigma_hat=item.get("sigma_hat"),
                    )
                    item["grade"] = _grade_from_edge(item["edge"])
                    if item["edge"] < MIN_EDGE:
                        item["is_publishable"] = False
            # Re-sort and re-filter after demotion
            publishable = [
                item for item in scored if item.get("is_publishable", False)
            ]
            top_picks = sorted(
                publishable, key=lambda x: x["edge"], reverse=True
            )[:top]

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
