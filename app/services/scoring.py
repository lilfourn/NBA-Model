from __future__ import annotations

from collections import Counter
import json
import logging
import math
import os
import threading
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch as _torch
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.ml.artifact_store import load_latest_artifact_as_file
from app.ml.artifacts import load_joblib_artifact, latest_compatible_joblib_path
from app.ml.infer_baseline import infer_over_probs as infer_lr_over_probs
from app.ml.lgbm.infer import infer_over_probs as infer_lgbm_over_probs
from app.ml.nn.infer import (
    infer_over_probs as infer_nn_over_probs,
    latest_compatible_checkpoint,
)
from app.ml.tabdl.infer import (
    infer_over_probs as infer_tabdl_over_probs,
    latest_compatible_checkpoint as latest_compatible_tabdl_checkpoint,
)
from app.ml.selection_policy import SelectionPolicy
from app.ml.xgb.infer import infer_over_probs as infer_xgb_over_probs
from app.modeling.conformal import ConformalCalibrator
from app.modeling.db_logs import load_db_game_logs
from app.modeling.forecast_calibration import ForecastDistributionCalibrator
from app.modeling.probability import confidence_from_probability
from app.modeling.stat_forecast import (
    ForecastParams,
    LeaguePriors,
    StatForecastPredictor,
)
from app.modeling.types import Projection
from app.ml.stat_mappings import (
    stat_components,
    stat_diff_components,
    stat_weighted_components,
)

logger = logging.getLogger(__name__)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

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
    p_pick: float = 0.5
    selection_threshold: float = 0.60
    selection_margin: float = 0.0
    policy_version: str = "legacy"
    is_publishable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoringResult:
    snapshot_id: str
    scored_at: str
    total_scored: int
    picks: list[ScoredPick]
    publishable_count: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "scored_at": self.scored_at,
            "total_scored": self.total_scored,
            "publishable_count": self.publishable_count,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
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
SHRINK_MAX = 0.25  # low-data picks: 25% pull toward prior

# Abstain policy: only publish picks with p_pick >= threshold
# p_pick = max(p_final, 1 - p_final)
PICK_THRESHOLD = 0.60
SELECTION_POLICY_PATH = os.getenv("SELECTION_POLICY_PATH", "models/selection_policy.json")
MAX_TOP_STAT_SHARE = 0.70

# Minimum edge score for a pick to be publishable
MIN_EDGE = 5.0

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
    "Offensive Rebounds",
    "Two Pointers Made",
    "Turnovers",
    "Blks+Stls",
}

ENSEMBLE_EXPERTS = ("p_lr", "p_xgb", "p_lgbm", "p_nn", "p_forecast_cal", "p_tabdl")


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


def _direction_imbalance_penalty(
    *,
    edge: float,
    prob_over: float,
    dominant_dir: str,
    dominant_pct: float,
    threshold: float,
    context_prior: float | None,
    max_penalty: float = 12.0,
) -> float:
    """Apply deterministic edge demotion for dominant-direction picks near context prior."""
    if dominant_pct <= threshold:
        return edge

    pick_dir = "OVER" if prob_over >= 0.5 else "UNDER"
    if pick_dir != dominant_dir:
        return edge

    # Scale penalty from 0..max_penalty as dominance moves from threshold..100%.
    dominance_scale = (dominant_pct - threshold) / (1.0 - threshold)
    dominance_scale = max(0.0, min(1.0, dominance_scale))

    prior = 0.5 if context_prior is None else float(context_prior)
    distance_to_prior = abs(float(prob_over) - prior)
    # Picks close to prior carry less differentiated signal and are demoted more.
    prior_alignment = 1.0 - min(1.0, distance_to_prior / 0.20)

    penalty = round(max_penalty * dominance_scale * prior_alignment, 1)
    return round(max(0.0, float(edge) - penalty), 1)


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


def _select_empty_publishable_fallback(
    scored: list[dict[str, Any]],
    *,
    top: int,
) -> list[dict[str, Any]]:
    """Return best-effort picks when strict publishable set is empty."""

    def _passes_soft_quality(
        item: dict[str, Any],
        *,
        min_pick: float,
        min_edge: float,
        require_non_ambiguous_conformal: bool,
    ) -> bool:
        if bool(item.get("is_prior_only")):
            return False
        try:
            p_pick = float(item.get("p_pick"))
            edge = float(item.get("edge"))
        except (TypeError, ValueError):
            return False
        if p_pick < min_pick or edge < min_edge:
            return False
        if require_non_ambiguous_conformal and item.get("conformal_set_size") == 2:
            return False
        if not bool(item.get("forecast_edge_ok", True)):
            return False
        n_eff = item.get("n_eff")
        if n_eff is not None:
            try:
                if float(n_eff) < 5.0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    strict_soft = [
        item
        for item in scored
        if _passes_soft_quality(
            item,
            min_pick=0.58,
            min_edge=max(2.5, MIN_EDGE * 0.5),
            require_non_ambiguous_conformal=True,
        )
    ]
    if strict_soft:
        return strict_soft[:top]

    relaxed_soft = [
        item
        for item in scored
        if _passes_soft_quality(
            item,
            min_pick=0.55,
            min_edge=1.0,
            require_non_ambiguous_conformal=False,
        )
    ]
    if relaxed_soft:
        return relaxed_soft[:top]

    non_prior_only = [item for item in scored if not bool(item.get("is_prior_only"))]
    if non_prior_only:
        return non_prior_only[:top]

    # Last-resort guarantee for non-empty UX if any rows were scored.
    return scored[:top]


def _item_projection_id(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("projection_id")
    else:
        value = getattr(item, "projection_id", None)
    if value is None:
        return str(id(item))
    value_text = str(value).strip()
    return value_text or str(id(item))


def _item_stat_type(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("stat_type")
    else:
        value = getattr(item, "stat_type", None)
    value_text = str(value or "").strip()
    return value_text or "unknown"


def _select_diverse_top(
    items: list[Any],
    *,
    top: int,
    max_top_stat_share: float = MAX_TOP_STAT_SHARE,
) -> list[Any]:
    """Apply a per-stat concentration cap to avoid one-stat dominance."""
    if top <= 0 or not items:
        return []
    capped_top = min(int(top), len(items))
    max_share = max(0.05, min(1.0, float(max_top_stat_share)))
    available_by_stat: Counter[str] = Counter(_item_stat_type(item) for item in items)
    if len(available_by_stat) <= 1:
        return items[:capped_top]

    target_total = 0
    stat_cap = 0
    for candidate_total in range(capped_top, 0, -1):
        candidate_cap = max(1, math.floor(candidate_total * max_share))
        capacity = sum(
            min(int(count), candidate_cap) for count in available_by_stat.values()
        )
        if capacity >= candidate_total:
            target_total = candidate_total
            stat_cap = candidate_cap
            break
    if target_total <= 0:
        return []

    selected: list[Any] = []
    selected_ids: set[str] = set()
    stat_counts: Counter[str] = Counter()

    for item in items:
        proj_id = _item_projection_id(item)
        if proj_id in selected_ids:
            continue
        stat_type = _item_stat_type(item)
        if stat_counts[stat_type] >= stat_cap:
            continue
        selected.append(item)
        selected_ids.add(proj_id)
        stat_counts[stat_type] += 1
        if len(selected) >= target_total:
            return selected
    return selected


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
# Clipping to [0.25, 0.75] bounds any expert's logit influence to ±1.10.
_EXPERT_PROB_FLOOR = 0.25
_EXPERT_PROB_CEIL = 0.75


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
        is_today=(
            bool(_row_get(row, "today")) if _row_get(row, "today") is not None else None
        ),
        is_combo=bool(_row_get(row, "combo") or False),
    )


def _latest_logged_snapshot_id(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text(
                """
                select pp.snapshot_id
                from projection_predictions pp
                where pp.snapshot_id is not null
                order by coalesce(pp.decision_time, pp.created_at) desc, pp.created_at desc
                limit 1
                """
            )
        ).scalar()


def _logged_snapshot_id_for_game_date(engine: Engine, game_date: str) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text(
                """
                select pp.snapshot_id
                from projection_predictions pp
                join projection_features pf
                  on pf.snapshot_id = pp.snapshot_id
                 and pf.projection_id = pp.projection_id
                where pp.snapshot_id is not null
                  and (pf.start_time at time zone 'America/New_York')::date = :game_date
                order by coalesce(pp.decision_time, pp.created_at) desc, pp.created_at desc
                limit 1
                """
            ),
            {"game_date": game_date},
        ).scalar()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _details_dict(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text_value = raw.strip()
        if not text_value:
            return {}
        try:
            parsed = json.loads(text_value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def score_logged_predictions(
    engine: Engine,
    *,
    snapshot_id: str | None = None,
    game_date: str | None = None,
    top: int = 50,
    rank: str = "risk_adj",
    include_non_today: bool = False,
) -> ScoringResult:
    resolved_snapshot = snapshot_id
    if not resolved_snapshot and game_date:
        resolved_snapshot = _logged_snapshot_id_for_game_date(engine, str(game_date))
    resolved_snapshot = resolved_snapshot or _latest_logged_snapshot_id(engine)
    if not resolved_snapshot:
        return ScoringResult(
            snapshot_id="",
            scored_at=datetime.now(timezone.utc).isoformat(),
            total_scored=0,
            picks=[],
        )

    query = text(
        """
        with latest as (
            select distinct on (pp.projection_id)
                pp.projection_id,
                pp.player_id,
                pp.game_id,
                pp.stat_type,
                pp.line_score,
                pp.pick,
                pp.prob_over,
                pp.confidence,
                pp.rank_score,
                pp.p_forecast_cal,
                pp.p_nn,
                pp.p_tabdl,
                pp.p_lr,
                pp.p_xgb,
                pp.p_lgbm,
                pp.n_eff,
                pp.mean as mu_hat,
                pp.std as sigma_hat,
                pp.details,
                coalesce(pp.decision_time, pp.created_at) as scored_at
            from projection_predictions pp
            where pp.snapshot_id = :snapshot_id
            order by
                pp.projection_id,
                coalesce(pp.decision_time, pp.created_at) desc,
                pp.created_at desc,
                pp.id desc
        )
        select
            l.*,
            coalesce(pl.display_name, '') as player_name,
            coalesce(
                nullif(pl.image_url, ''),
                nullif(p.attributes->>'custom_image', '')
            ) as player_image_url
        from latest l
        left join players pl
            on pl.id = l.player_id
        left join projections p
            on p.snapshot_id = :snapshot_id
           and p.projection_id = l.projection_id
        left join projection_features pf
            on pf.snapshot_id = :snapshot_id
           and pf.projection_id = l.projection_id
        where (:include_non_today = true or coalesce(pf.today, true) = true)
        """
    )
    frame = pd.read_sql(
        query,
        engine,
        params={
            "snapshot_id": str(resolved_snapshot),
            "include_non_today": bool(include_non_today),
        },
    )
    if frame.empty:
        return ScoringResult(
            snapshot_id=str(resolved_snapshot),
            scored_at=datetime.now(timezone.utc).isoformat(),
            total_scored=0,
            picks=[],
        )

    if rank == "confidence":
        frame = frame.sort_values(
            by=["confidence", "rank_score"],
            ascending=[False, False],
            na_position="last",
        )
    else:
        frame = frame.sort_values(
            by=["rank_score", "confidence"],
            ascending=[False, False],
            na_position="last",
        )

    scored_at_ts = pd.to_datetime(frame.get("scored_at"), errors="coerce", utc=True)
    if getattr(scored_at_ts, "empty", True):
        scored_at = datetime.now(timezone.utc).isoformat()
    else:
        max_ts = scored_at_ts.max()
        scored_at = (
            max_ts.to_pydatetime().isoformat()
            if pd.notna(max_ts)
            else datetime.now(timezone.utc).isoformat()
        )

    scored_items: list[tuple[ScoredPick, bool]] = []
    for row in frame.itertuples(index=False):
        details = _details_dict(getattr(row, "details", None))

        prob_over = _safe_float(getattr(row, "prob_over", None))
        if prob_over is None:
            prob_over = _safe_float(details.get("p_final"))
        if prob_over is None:
            prob_over = 0.5
        prob_over = max(0.0, min(1.0, prob_over))

        confidence = _safe_float(getattr(row, "confidence", None))
        if confidence is None:
            confidence = float(confidence_from_probability(prob_over))
        confidence = max(0.0, min(1.0, confidence))

        rank_score = _safe_float(getattr(row, "rank_score", None))
        if rank_score is None:
            rank_score = confidence

        edge = _safe_float(details.get("edge"))
        if edge is None:
            if rank_score <= 1.5:
                edge = rank_score * 100.0
            else:
                edge = rank_score
        edge = round(max(0.0, min(100.0, float(edge))), 1)

        grade_raw = details.get("grade")
        grade = (
            str(grade_raw).strip().upper()
            if isinstance(grade_raw, str) and grade_raw.strip()
            else _grade_from_edge(edge)
        )

        conformal_set_size = None
        conformal_raw = details.get("conformal_set_size")
        if conformal_raw is not None:
            try:
                conformal_set_size = int(conformal_raw)
            except (TypeError, ValueError):
                conformal_set_size = None

        calibration_status = (
            str(details.get("calibration_status") or "logged").strip() or "logged"
        )
        p_pick = _safe_float(getattr(row, "p_pick", None))
        if p_pick is None:
            p_pick = _safe_float(details.get("p_pick"))
        if p_pick is None:
            p_pick = max(prob_over, 1.0 - prob_over)
        p_pick = max(0.0, min(1.0, p_pick))

        selection_threshold = _safe_float(getattr(row, "selection_threshold", None))
        if selection_threshold is None:
            selection_threshold = _safe_float(details.get("selection_threshold"))
        if selection_threshold is None:
            selection_threshold = PICK_THRESHOLD

        selection_margin = _safe_float(getattr(row, "selection_margin", None))
        if selection_margin is None:
            selection_margin = _safe_float(details.get("selection_margin"))
        if selection_margin is None:
            selection_margin = p_pick - selection_threshold

        policy_version = str(
            details.get("policy_version")
            or getattr(row, "policy_version", None)
            or "legacy"
        ).strip() or "legacy"

        pick = str(getattr(row, "pick", "") or "").strip().upper()
        if pick not in {"OVER", "UNDER"}:
            pick = "OVER" if prob_over >= 0.5 else "UNDER"

        player_name = (
            str(getattr(row, "player_name", "") or "").strip()
            or str(details.get("player_name") or "").strip()
            or "Unknown"
        )

        publishable_raw = details.get("is_publishable")
        if isinstance(publishable_raw, bool):
            is_publishable = publishable_raw
        elif isinstance(publishable_raw, (int, float)):
            is_publishable = bool(publishable_raw)
        elif isinstance(publishable_raw, str):
            is_publishable = publishable_raw.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
        else:
            is_publishable = bool(selection_margin >= 0.0 and edge >= MIN_EDGE)

        scored_pick = ScoredPick(
            projection_id=str(getattr(row, "projection_id", "") or ""),
            player_name=player_name,
            player_image_url=_optional_str(getattr(row, "player_image_url", None)),
            player_id=str(getattr(row, "player_id", "") or ""),
            game_id=_optional_str(getattr(row, "game_id", None)),
            stat_type=str(getattr(row, "stat_type", "") or ""),
            line_score=float(_safe_float(getattr(row, "line_score", None)) or 0.0),
            pick=pick,
            prob_over=prob_over,
            confidence=confidence,
            rank_score=float(rank_score),
            p_forecast_cal=_safe_float(getattr(row, "p_forecast_cal", None)),
            p_nn=_safe_float(getattr(row, "p_nn", None)),
            p_tabdl=_safe_float(getattr(row, "p_tabdl", None)),
            p_lr=_safe_float(getattr(row, "p_lr", None)),
            p_xgb=_safe_float(getattr(row, "p_xgb", None)),
            p_lgbm=_safe_float(getattr(row, "p_lgbm", None)),
            p_meta=_safe_float(details.get("p_meta")),
            mu_hat=_safe_float(getattr(row, "mu_hat", None)),
            sigma_hat=_safe_float(getattr(row, "sigma_hat", None)),
            calibration_status=calibration_status,
            n_eff=_safe_float(getattr(row, "n_eff", None)),
            conformal_set_size=conformal_set_size,
            edge=edge,
            grade=grade,
            p_pick=p_pick,
            selection_threshold=float(selection_threshold),
            selection_margin=float(selection_margin),
            policy_version=policy_version,
            is_publishable=is_publishable,
        )
        scored_items.append((scored_pick, is_publishable))

    publishable = [pick for pick, ok in scored_items if ok]
    top_picks = _select_diverse_top(publishable, top=int(top))
    fallback_used = False
    fallback_reason: str | None = None
    if not top_picks:
        top_picks = _select_diverse_top(
            [pick for pick, _ in scored_items],
            top=int(top),
        )
        if top_picks:
            fallback_used = True
            fallback_reason = "logged_no_publishable"

    return ScoringResult(
        snapshot_id=str(resolved_snapshot),
        scored_at=scored_at,
        total_scored=int(len(frame)),
        picks=top_picks,
        publishable_count=int(len(publishable)),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
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
    light_mode = _env_truthy("SCORING_LIGHT_MODE", False)
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
                    publishable_count=int(cached_dict.get("publishable_count") or 0),
                    fallback_used=bool(cached_dict.get("fallback_used", False)),
                    fallback_reason=cached_dict.get("fallback_reason"),
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

    _configured_model_source = (settings.model_source or "fs").strip().lower()
    _use_db = _configured_model_source == "db"
    models_path = Path(models_dir)
    if _use_db:
        nn_path = load_latest_artifact_as_file(engine, "nn_gru_attention", suffix=".pt")
        tabdl_path = load_latest_artifact_as_file(engine, "tabdl_mlp", suffix=".pt")
        lr_path = load_latest_artifact_as_file(
            engine, "baseline_logreg", suffix=".joblib"
        )
        xgb_path = load_latest_artifact_as_file(engine, "xgb", suffix=".joblib")
        lgbm_path = load_latest_artifact_as_file(engine, "lgbm", suffix=".joblib")
    else:
        if not models_path.exists():
            logger.warning(
                "MODELS_DIR=%s does not exist. Local learned experts are unavailable.",
                models_path,
            )
        nn_path = latest_compatible_checkpoint(models_path, "nn_gru_attention_*.pt")
        tabdl_path = latest_compatible_tabdl_checkpoint(models_path, "tabdl_*.pt")
        lr_path = latest_compatible_joblib_path(models_path, "baseline_logreg_*.joblib")
        xgb_path = latest_compatible_joblib_path(models_path, "xgb_*.joblib")
        lgbm_path = latest_compatible_joblib_path(models_path, "lgbm_*.joblib")
        if not any((nn_path, tabdl_path, lr_path, xgb_path, lgbm_path)):
            logger.warning(
                "No local learned expert artifacts found in %s. "
                "Falling back to DB model_artifacts.",
                models_path,
            )
            try:
                nn_path = load_latest_artifact_as_file(
                    engine, "nn_gru_attention", suffix=".pt"
                )
                tabdl_path = load_latest_artifact_as_file(
                    engine, "tabdl_mlp", suffix=".pt"
                )
                lr_path = load_latest_artifact_as_file(
                    engine, "baseline_logreg", suffix=".joblib"
                )
                xgb_path = load_latest_artifact_as_file(
                    engine, "xgb", suffix=".joblib"
                )
                lgbm_path = load_latest_artifact_as_file(
                    engine, "lgbm", suffix=".joblib"
                )
                _use_db = True
            except Exception:
                logger.warning(
                    "DB artifact fallback failed; continuing without learned experts.",
                    exc_info=True,
                )

    # Stacking meta-learner (optional)
    _stacking_model = None
    if light_mode:
        logger.info("Stacking meta-model disabled in SCORING_LIGHT_MODE.")
    else:
        try:
            from app.ml.stacking import predict_stacking

            if _use_db:
                _stacking_path = load_latest_artifact_as_file(
                    engine, "stacking_meta", suffix=".joblib"
                )
            else:
                _stacking_path = latest_compatible_joblib_path(
                    models_path, "stacking_meta_*.joblib"
                )
            if _stacking_path and _stacking_path.exists():
                _stacking_payload = load_joblib_artifact(str(_stacking_path))
                _stacking_model = _stacking_payload.get("model")
        except Exception:
            logger.warning(
                "Stacking meta-model not available, using logit average fallback"
            )

    logger.info(
        "Model paths resolved (configured_source=%s effective_db=%s): "
        "nn=%s tabdl=%s lr=%s xgb=%s lgbm=%s stacking=%s",
        _configured_model_source,
        _use_db,
        nn_path,
        tabdl_path,
        lr_path,
        xgb_path,
        lgbm_path,
        _stacking_model is not None,
    )

    if light_mode:
        logger.warning(
            "SCORING_LIGHT_MODE enabled: disabling memory-heavy experts "
            "(forecast, nn, tabdl, stacking)."
        )
        nn_path = None
        tabdl_path = None
        _stacking_model = None

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

    forecast_map: dict[str, dict[str, object]] = {}
    if light_mode:
        logger.info("Forecast expert skipped in SCORING_LIGHT_MODE.")
    else:
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

    logger.info(
        "Expert coverage for snapshot %s: forecast=%d nn=%d tabdl=%d lr=%d xgb=%d lgbm=%d",
        resolved_snapshot,
        len(forecast_map),
        len(p_nn),
        len(p_tabdl),
        len(p_lr),
        len(p_xgb),
        len(p_lgbm),
    )
    if not any((p_nn, p_tabdl, p_lr, p_xgb, p_lgbm)):
        logger.warning(
            "No learned experts produced predictions for snapshot %s. "
            "This usually indicates missing or incompatible model artifacts and "
            "can result in zero publishable picks.",
            resolved_snapshot,
        )

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
        _sp_path = load_latest_artifact_as_file(
            engine, "selection_policy", suffix=".json"
        )
        _selection_policy = (
            SelectionPolicy.load(str(_sp_path))
            if _sp_path
            else SelectionPolicy.load(SELECTION_POLICY_PATH)
        )
    else:
        _context_priors = load_context_priors()
        _stat_calibrator = StatTypeCalibrator.load()
        _selection_policy = SelectionPolicy.load(SELECTION_POLICY_PATH)

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
                k: v
                for k, v in _routing_raw.items()
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

    # --- Artifact availability diagnostic ---
    _n_stat_cals = len(_stat_calibrator.calibrators) if _stat_calibrator else 0
    _n_ctx_stats = len(_context_priors.get("stat_type_priors", {}))
    logger.info(
        "Scoring artifact status: stat_calibrator=%d stat-types, "
        "context_priors=%d stat-types, selection_policy=%s",
        _n_stat_cals,
        _n_ctx_stats,
        _selection_policy.version,
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
        """Disabled: systemic inversion signals indicate calibration issues, not individual expert problems."""
        return ep

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
        expert_probs = _apply_inversions(expert_probs)
        # Clip to prevent logit-space outlier domination (e.g. TabDL at 8%)
        expert_probs = _clip_expert_probs(expert_probs)

        n_eff = f.get("n_eff")
        try:
            n_eff_val = float(n_eff) if n_eff is not None else None
        except (TypeError, ValueError):
            n_eff_val = None
        # Ensemble: stacking meta-model or logit average fallback
        if _stacking_model is not None:
            from app.ml.stacking import predict_stacking

            p_raw = predict_stacking(
                _stacking_model, {e: expert_probs.get(e) for e in ENSEMBLE_EXPERTS}
            )
        else:
            avail = [
                expert_probs[e]
                for e in ENSEMBLE_EXPERTS
                if expert_probs.get(e) is not None
            ]
            p_raw = (
                _sigmoid(sum(_logit(p) for p in avail) / len(avail)) if avail else 0.5
            )
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
            p_pre_cal = _ctx_prior if _ctx_prior is not None else 0.5
            p_final = p_pre_cal
            p_raw = p_final
            calibrator_source = "prior_only"
            calibrator_mode = "bypass"
        else:
            p_pre_cal = shrink_probability(
                p_raw, n_eff=n_eff_val, context_prior=_ctx_prior
            )
            # Apply per-stat-type isotonic recalibration
            p_final, calibrator_source, calibrator_mode = (
                _stat_calibrator.transform_with_info(p_pre_cal, stat_type)
            )

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
        selection_threshold = float(
            _selection_policy.threshold_for(stat_type, conformal_size)
        )
        selection_margin = float(p_pick - selection_threshold)
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
            selection_margin >= 0.0
            and not _is_prior_only
            and _has_diversity
            and _has_min_neff
            and _forecast_edge_ok
        )
        reject_reasons: list[str] = []
        if selection_margin < 0.0:
            reject_reasons.append("low_pick_confidence")
        if _is_prior_only:
            reject_reasons.append("prior_only_stat")
        if not _has_diversity:
            reject_reasons.append("no_expert_diversity")
        if not _has_min_neff:
            reject_reasons.append("low_n_eff")
        if not _forecast_edge_ok:
            reject_reasons.append("forecast_edge_too_large")

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
                "p_pre_cal": p_pre_cal,
                "confidence": conf,
                "rank_score": float(score),
                "p_forecast_cal": expert_probs["p_forecast_cal"],
                "p_nn": expert_probs["p_nn"],
                "p_tabdl": expert_probs["p_tabdl"],
                "p_lr": expert_probs["p_lr"],
                "p_xgb": expert_probs["p_xgb"],
                "p_lgbm": expert_probs["p_lgbm"],
                "p_meta": None,
                "mu_hat": float(f.get("mu_hat") or 0.0) if f else None,
                "sigma_hat": float(f.get("sigma_hat") or 0.0) if f else None,
                "calibration_status": status,
                "n_eff": n_eff_val,
                "conformal_set_size": conformal_size,
                "p_pick": p_pick,
                "selection_threshold": selection_threshold,
                "selection_margin": selection_margin,
                "policy_version": _selection_policy.version,
                "calibrator_source": calibrator_source,
                "calibrator_mode": calibrator_mode,
                "is_prior_only": _is_prior_only,
                "has_diversity": _has_diversity,
                "has_min_neff": _has_min_neff,
                "forecast_edge_ok": _forecast_edge_ok,
                "reject_reasons": reject_reasons,
            }
        )
        item = scored[-1]
        item["edge"] = _compute_edge(
            p_final,
            expert_probs,
            item["conformal_set_size"],
            n_eff=n_eff_val,
            mu_hat=_mu_hat_val,
            line_score=_line_score,
            sigma_hat=_sigma_hat_val,
        )
        item["grade"] = _grade_from_edge(item["edge"])
        # Apply min_edge filter
        if item["edge"] < MIN_EDGE:
            is_publishable = False
            item["reject_reasons"].append("edge_below_min")
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
                reasons = item.setdefault("reject_reasons", [])
                if "low_model_discrimination" not in reasons:
                    reasons.append("low_model_discrimination")

    scored.sort(key=lambda item: item["edge"], reverse=True)
    # Enforce abstain policy: only return publishable picks in the top-N
    publishable = [item for item in scored if item.get("is_publishable", False)]
    top_picks = _select_diverse_top(publishable, top=top)

    # --- Direction balance guardrail ---
    # When >75% of picks lean one direction, apply a soft correction:
    # demote dominant-direction picks that sit near the context prior.
    _IMBALANCE_THRESHOLD = 0.75  # trigger correction above this ratio
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
                    # Penalize more aggressively during imbalance
                    item["edge"] = _compute_edge(
                        item["prob_over"],
                        {
                            k: item.get(k)
                            for k in [
                                "p_forecast_cal",
                                "p_nn",
                                "p_tabdl",
                                "p_lr",
                                "p_xgb",
                                "p_lgbm",
                            ]
                        },
                        item["conformal_set_size"],
                        n_eff=item["n_eff"],
                        mu_hat=item.get("mu_hat"),
                        line_score=item.get("line_score"),
                        sigma_hat=item.get("sigma_hat"),
                    )
                    item["edge"] = _direction_imbalance_penalty(
                        edge=item["edge"],
                        prob_over=item["prob_over"],
                        dominant_dir=dominant_dir,
                        dominant_pct=dominant_pct,
                        threshold=_IMBALANCE_THRESHOLD,
                        context_prior=get_context_prior(
                            _context_priors,
                            stat_type=st,
                            line_score=ls,
                        ),
                    )
                    item["grade"] = _grade_from_edge(item["edge"])
                    if item["edge"] < MIN_EDGE:
                        item["is_publishable"] = False
                        reasons = item.setdefault("reject_reasons", [])
                        if "edge_below_min" not in reasons:
                            reasons.append("edge_below_min")
            # Re-sort and re-filter after demotion
            publishable = [item for item in scored if item.get("is_publishable", False)]
            top_picks = _select_diverse_top(
                sorted(publishable, key=lambda x: x["edge"], reverse=True),
                top=top,
            )

    fallback_used = False
    fallback_reason: str | None = None
    if not top_picks and scored:
        reject_counts: Counter[str] = Counter()
        for item in scored:
            for reason in item.get("reject_reasons", []):
                reject_counts[str(reason)] += 1
        logger.warning(
            "No publishable picks for snapshot %s. Rejection breakdown=%s",
            resolved_snapshot,
            dict(reject_counts.most_common(8)),
        )
        top_picks = _select_empty_publishable_fallback(scored, top=top)
        top_picks = _select_diverse_top(top_picks, top=top)
        if top_picks:
            fallback_used = True
            fallback_reason = "soft_fallback_no_publishable"
            logger.warning(
                "Soft fallback active: returning %d ranked non-publishable picks "
                "for snapshot %s",
                len(top_picks),
                resolved_snapshot,
            )

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
            p_pick=float(item.get("p_pick") or 0.5),
            selection_threshold=float(item.get("selection_threshold") or 0.60),
            selection_margin=float(item.get("selection_margin") or 0.0),
            policy_version=str(item.get("policy_version") or "legacy"),
            is_publishable=bool(item.get("is_publishable", False)),
        )
        for item in top_picks
    ]

    result = ScoringResult(
        snapshot_id=str(resolved_snapshot),
        scored_at=datetime.now(timezone.utc).isoformat(),
        total_scored=len(scored),
        picks=picks,
        publishable_count=len(publishable),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
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
