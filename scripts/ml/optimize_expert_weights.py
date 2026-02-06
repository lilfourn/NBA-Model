from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.modeling.online_ensemble import Context  # noqa: E402
from app.modeling.online_ensemble import ContextualHedgeEnsembler  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

DEFAULT_EXPERTS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]
EPS = 1e-6


def _clip_prob(p: float) -> float:
    if not math.isfinite(float(p)):
        return 0.5
    return float(min(1.0 - EPS, max(EPS, p)))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _ctx_key(stat_type: object, is_live: object, n_eff: object) -> tuple[str, str, str]:
    n_eff_val: float | None = None
    try:
        if n_eff is not None and not pd.isna(n_eff):
            n_eff_val = float(n_eff)
    except Exception:  # noqa: BLE001
        n_eff_val = None
    ctx = Context(
        stat_type=str(stat_type or ""),
        is_live=bool(is_live) if is_live is not None and not pd.isna(is_live) else False,
        n_eff=n_eff_val,
    )
    return ctx.key()


def _ctx_key_str(ctx_key: tuple[str, str, str]) -> str:
    return json.dumps(list(ctx_key), ensure_ascii=False, separators=(",", ":"))


def _weights_from_vector(experts: list[str], vec: np.ndarray) -> dict[str, float]:
    return {expert: float(vec[idx]) for idx, expert in enumerate(experts)}


def _vector_from_weights(experts: list[str], weights: dict[str, float], default: float = 0.0) -> np.ndarray:
    out = np.full(len(experts), float(default), dtype=np.float64)
    for i, expert in enumerate(experts):
        value = weights.get(expert)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out[i] = max(0.0, v)
    total = float(out.sum())
    if total <= 0:
        out[:] = 1.0 / float(max(1, len(experts)))
    else:
        out /= total
    return out


def _project_simplex(vec: np.ndarray) -> np.ndarray:
    if vec.size == 0:
        return vec
    u = np.sort(vec)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, vec.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(vec, 1.0 / float(vec.size))
    rho = int(np.where(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    projected = np.maximum(vec - theta, 0.0)
    denom = float(projected.sum())
    if denom <= 0:
        return np.full_like(projected, 1.0 / float(projected.size))
    return projected / denom


def _weighted_logit_pool(row: pd.Series, experts: list[str], weights: dict[str, float]) -> float:
    probs: list[float] = []
    w_vals: list[float] = []
    for expert in experts:
        value = row.get(expert)
        if value is None or pd.isna(value):
            continue
        p = _clip_prob(float(value))
        w = float(weights.get(expert, 0.0))
        if not math.isfinite(w) or w <= 0:
            continue
        probs.append(p)
        w_vals.append(w)

    if not probs:
        return 0.5
    w_sum = float(sum(w_vals))
    if w_sum <= 0:
        return float(np.mean(probs))

    z = 0.0
    for p, w in zip(probs, w_vals):
        z += (w / w_sum) * math.log(p / (1.0 - p))
    return _sigmoid(z)


def _predict_frame(
    frame: pd.DataFrame,
    experts: list[str],
    *,
    global_weights: dict[str, float],
    context_weights: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    preds = np.full(len(frame), 0.5, dtype=np.float64)
    if frame.empty:
        return preds
    context_weights = context_weights or {}
    for idx, row in enumerate(frame.itertuples(index=False)):
        key = _ctx_key(
            getattr(row, "stat_type", None),
            getattr(row, "is_live", False),
            getattr(row, "n_eff", None),
        )
        key_s = _ctx_key_str(key)
        weights = context_weights.get(key_s) or global_weights
        preds[idx] = _weighted_logit_pool(pd.Series(row._asdict()), experts, weights)
    return np.clip(preds, EPS, 1.0 - EPS)


def _evaluate_predictions(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | None]:
    if labels.size == 0 or preds.size == 0:
        return {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}
    picks = (preds >= 0.5).astype(int)
    auc: float | None = None
    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, preds))
    return {
        "accuracy": float(accuracy_score(labels, picks)),
        "logloss": float(log_loss(labels, np.clip(preds, EPS, 1.0 - EPS))),
        "roc_auc": auc,
        "n": int(labels.size),
    }


def _evaluate_single_experts(frame: pd.DataFrame, experts: list[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for expert in experts:
        if expert not in frame.columns:
            out[expert] = {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}
            continue
        sub = frame.dropna(subset=[expert, "over_label"]).copy()
        if sub.empty:
            out[expert] = {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}
            continue
        probs = np.clip(pd.to_numeric(sub[expert], errors="coerce").to_numpy(dtype=float), EPS, 1.0 - EPS)
        labels = sub["over_label"].astype(int).to_numpy()
        finite_mask = np.isfinite(probs)
        if not np.any(finite_mask):
            out[expert] = {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}
            continue
        out[expert] = _evaluate_predictions(labels[finite_mask], probs[finite_mask])
    return out


def _filter_frame_for_experts(frame: pd.DataFrame, experts: list[str], *, min_experts: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if not experts:
        return frame.head(0).copy()
    out = frame.copy()
    required = min(len(experts), max(1, int(min_experts)))
    out["available_experts"] = out[experts].notna().sum(axis=1)
    out = out[out["available_experts"] >= required].copy()
    return out


def _objective(
    frame: pd.DataFrame,
    experts: list[str],
    weights: dict[str, float],
    *,
    prior: dict[str, float] | None = None,
    reg_lambda: float = 0.0,
) -> float:
    labels = frame["over_label"].astype(int).to_numpy()
    preds = np.array([_weighted_logit_pool(row, experts, weights) for _, row in frame.iterrows()], dtype=np.float64)
    ll = float(log_loss(labels, np.clip(preds, EPS, 1.0 - EPS)))
    if prior and reg_lambda > 0:
        penalty = 0.0
        for expert in experts:
            d = float(weights.get(expert, 0.0) - prior.get(expert, 0.0))
            penalty += d * d
        ll += float(reg_lambda) * penalty
    return ll


def _optimize_weight_vector(
    frame: pd.DataFrame,
    experts: list[str],
    *,
    random_trials: int,
    seed: int,
    prior: dict[str, float] | None = None,
    reg_lambda: float = 0.0,
) -> dict[str, float]:
    n = len(experts)
    if n == 0:
        return {}
    rng = np.random.default_rng(int(seed))

    if prior:
        best_vec = _vector_from_weights(experts, prior, default=1.0 / float(n))
    else:
        best_vec = np.full(n, 1.0 / float(n), dtype=np.float64)
    best_w = _weights_from_vector(experts, best_vec)
    best_obj = _objective(frame, experts, best_w, prior=prior, reg_lambda=reg_lambda)

    # Structured candidates: one-hot + prior + uniform.
    for i in range(n):
        vec = np.zeros(n, dtype=np.float64)
        vec[i] = 1.0
        w = _weights_from_vector(experts, vec)
        score = _objective(frame, experts, w, prior=prior, reg_lambda=reg_lambda)
        if score < best_obj:
            best_obj = score
            best_vec = vec
            best_w = w

    # Random Dirichlet search.
    for _ in range(max(250, int(random_trials))):
        vec = rng.dirichlet(np.ones(n, dtype=np.float64))
        w = _weights_from_vector(experts, vec)
        score = _objective(frame, experts, w, prior=prior, reg_lambda=reg_lambda)
        if score < best_obj:
            best_obj = score
            best_vec = vec
            best_w = w

    # Local refinement.
    step = 0.08
    for _ in range(400):
        proposal = _project_simplex(best_vec + rng.normal(0.0, step, size=n))
        w = _weights_from_vector(experts, proposal)
        score = _objective(frame, experts, w, prior=prior, reg_lambda=reg_lambda)
        if score + 1e-10 < best_obj:
            best_obj = score
            best_vec = proposal
            best_w = w
        else:
            step *= 0.995

    return best_w


def _uniform_weights(experts: list[str]) -> dict[str, float]:
    if not experts:
        return {}
    w = 1.0 / float(len(experts))
    return {expert: w for expert in experts}


def _load_frame(engine, *, days_back: int, experts: list[str], min_experts: int) -> pd.DataFrame:
    expert_select = ", ".join(
        [
            "p_forecast_cal",
            "p_nn",
            "coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl",
            "p_lr",
            "p_xgb",
            "p_lgbm",
        ]
    )
    query = text(
        f"""
        select
            id::text as prediction_id,
            stat_type,
            n_eff,
            coalesce((details->>'is_live')::boolean, false) as is_live,
            over_label,
            outcome,
            prob_over as p_final,
            coalesce(decision_time, created_at) as decision_time,
            created_at,
            {expert_select}
        from projection_predictions
        where over_label is not null
          and actual_value is not null
          and outcome in ('over', 'under')
          and coalesce(decision_time, created_at) >= now() - (:days_back * interval '1 day')
        order by coalesce(decision_time, created_at) asc, created_at asc, id asc
        """
    )
    frame = pd.read_sql(query, engine, params={"days_back": int(max(1, days_back))})
    if frame.empty:
        return frame
    frame["over_label"] = pd.to_numeric(frame["over_label"], errors="coerce")
    frame = frame.dropna(subset=["over_label"])
    frame["over_label"] = frame["over_label"].astype(int)
    frame["n_eff"] = pd.to_numeric(frame["n_eff"], errors="coerce")
    for col in experts:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], errors="coerce", utc=True)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["decision_time", "stat_type"])
    frame["available_experts"] = frame[experts].notna().sum(axis=1)
    frame = frame[frame["available_experts"] >= int(max(1, min_experts))].copy()
    return frame


def _time_split(frame: pd.DataFrame, holdout_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, frame
    h = min(0.5, max(0.05, float(holdout_frac)))
    split_idx = int(round(len(frame) * (1.0 - h)))
    split_idx = min(max(split_idx, 1), len(frame) - 1)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def _contextual_weights(
    train_frame: pd.DataFrame,
    experts: list[str],
    *,
    global_weights: dict[str, float],
    min_context_rows: int,
    shrink_strength: float,
    random_trials: int,
    seed: int,
    reg_lambda: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if train_frame.empty:
        return out
    frame = train_frame.copy()
    frame["context_key"] = [
        _ctx_key_str(_ctx_key(s, l, n))
        for s, l, n in frame[["stat_type", "is_live", "n_eff"]].itertuples(index=False, name=None)
    ]
    for ctx_key, grp in frame.groupby("context_key", sort=False):
        n_rows = int(len(grp))
        if n_rows < int(max(1, min_context_rows)):
            out[ctx_key] = dict(global_weights)
            continue
        local = _optimize_weight_vector(
            grp,
            experts,
            random_trials=max(200, int(random_trials // 8)),
            seed=int(seed + n_rows),
            prior=global_weights,
            reg_lambda=float(reg_lambda),
        )
        blend = float(n_rows) / float(n_rows + max(1.0, float(shrink_strength)))
        blended = {
            expert: (blend * float(local.get(expert, 0.0)) + (1.0 - blend) * float(global_weights.get(expert, 0.0)))
            for expert in experts
        }
        vec = _project_simplex(_vector_from_weights(experts, blended))
        out[ctx_key] = _weights_from_vector(experts, vec)
    return out


def _evaluate_current_ensemble_replay(
    valid_frame: pd.DataFrame,
    experts: list[str],
    *,
    ensemble_path: Path,
) -> dict[str, float | None]:
    if valid_frame.empty or not ensemble_path.exists():
        return {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}
    try:
        ens = ContextualHedgeEnsembler.load(ensemble_path)
    except Exception:  # noqa: BLE001
        return {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}

    preds: list[float] = []
    labels: list[int] = []
    for row in valid_frame.itertuples(index=False):
        expert_probs = {expert: getattr(row, expert, None) for expert in experts}
        ctx = Context(
            stat_type=str(getattr(row, "stat_type", "") or ""),
            is_live=bool(getattr(row, "is_live", False) or False),
            n_eff=float(getattr(row, "n_eff")) if getattr(row, "n_eff") is not None and not pd.isna(getattr(row, "n_eff")) else None,
        )
        try:
            p = float(ens.predict(expert_probs, ctx))
        except Exception:  # noqa: BLE001
            continue
        if not math.isfinite(p):
            continue
        y = int(getattr(row, "over_label"))
        preds.append(_clip_prob(p))
        labels.append(y)

    if not labels:
        return {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}
    return _evaluate_predictions(np.array(labels, dtype=int), np.array(preds, dtype=np.float64))


def _run_optimization_bundle(
    *,
    frame: pd.DataFrame,
    experts: list[str],
    holdout_frac: float,
    random_trials: int,
    seed: int,
    contextual: bool,
    min_context_rows: int,
    shrink_strength: float,
    reg_lambda: float,
    current_ensemble_path: Path,
) -> dict[str, Any]:
    train_frame, valid_frame = _time_split(frame, holdout_frac=holdout_frac)
    if train_frame.empty or valid_frame.empty:
        raise RuntimeError("Not enough rows for time holdout split.")

    uniform = _uniform_weights(experts)
    global_opt = _optimize_weight_vector(
        train_frame,
        experts,
        random_trials=int(random_trials),
        seed=int(seed),
        prior=None,
        reg_lambda=0.0,
    )

    context_weights: dict[str, dict[str, float]] = {}
    if bool(contextual):
        context_weights = _contextual_weights(
            train_frame,
            experts,
            global_weights=global_opt,
            min_context_rows=int(min_context_rows),
            shrink_strength=float(shrink_strength),
            random_trials=int(random_trials),
            seed=int(seed),
            reg_lambda=float(reg_lambda),
        )

    y_train = train_frame["over_label"].astype(int).to_numpy()
    y_valid = valid_frame["over_label"].astype(int).to_numpy()

    train_uniform_preds = _predict_frame(train_frame, experts, global_weights=uniform)
    valid_uniform_preds = _predict_frame(valid_frame, experts, global_weights=uniform)
    train_global_preds = _predict_frame(train_frame, experts, global_weights=global_opt)
    valid_global_preds = _predict_frame(valid_frame, experts, global_weights=global_opt)
    valid_context_preds = _predict_frame(
        valid_frame,
        experts,
        global_weights=global_opt,
        context_weights=context_weights if context_weights else None,
    )

    valid_pfinal = valid_frame.dropna(subset=["p_final"]).copy()
    if not valid_pfinal.empty:
        p = np.clip(pd.to_numeric(valid_pfinal["p_final"], errors="coerce").to_numpy(dtype=float), EPS, 1.0 - EPS)
        y = valid_pfinal["over_label"].astype(int).to_numpy()
        metrics_pfinal = _evaluate_predictions(y, p)
    else:
        metrics_pfinal = {"accuracy": None, "logloss": None, "roc_auc": None, "n": 0}

    metrics_current_replay = _evaluate_current_ensemble_replay(
        valid_frame,
        experts,
        ensemble_path=current_ensemble_path,
    )
    valid_single = _evaluate_single_experts(valid_frame, experts)

    train_metrics = {
        "uniform": _evaluate_predictions(y_train, train_uniform_preds),
        "optimized_global": _evaluate_predictions(y_train, train_global_preds),
    }
    valid_metrics = {
        "uniform": _evaluate_predictions(y_valid, valid_uniform_preds),
        "optimized_global": _evaluate_predictions(y_valid, valid_global_preds),
        "optimized_contextual": _evaluate_predictions(y_valid, valid_context_preds),
        "current_ensemble_replay": metrics_current_replay,
        "current_p_final_logged": metrics_pfinal,
        "single_experts": valid_single,
    }

    return {
        "frame": frame,
        "train_frame": train_frame,
        "valid_frame": valid_frame,
        "global_opt": global_opt,
        "context_weights": context_weights,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }


def _select_weak_experts_to_prune(
    *,
    experts: list[str],
    global_weights: dict[str, float],
    valid_metrics: dict[str, Any],
    min_keep: int,
    min_experts: int,
    weight_threshold: float,
    min_valid_rows: int,
) -> tuple[list[str], list[str], dict[str, dict[str, float | int | None]]]:
    if not experts:
        return [], [], {}
    min_keep_safe = min(len(experts), max(2, int(min_keep), int(min_experts)))
    if len(experts) <= min_keep_safe:
        return list(experts), [], {}

    uniform_logloss = None
    uniform_metrics = valid_metrics.get("uniform")
    if isinstance(uniform_metrics, dict):
        ll = uniform_metrics.get("logloss")
        if ll is not None:
            uniform_logloss = float(ll)

    single_experts = valid_metrics.get("single_experts")
    if not isinstance(single_experts, dict):
        return list(experts), [], {}

    candidates: list[tuple[float, float, str, int]] = []
    reasons: dict[str, dict[str, float | int | None]] = {}
    for expert in experts:
        weight = float(global_weights.get(expert, 0.0))
        m = single_experts.get(expert) if isinstance(single_experts, dict) else None
        if not isinstance(m, dict):
            continue
        n_rows = int(m.get("n") or 0)
        ll_val = m.get("logloss")
        ll = float(ll_val) if ll_val is not None else None
        weak_weight = weight < float(weight_threshold)
        weak_perf = False
        if ll is None or n_rows < int(min_valid_rows):
            weak_perf = True
        elif uniform_logloss is not None and ll >= uniform_logloss:
            weak_perf = True
        if weak_weight and weak_perf:
            candidates.append((weight, -(ll if ll is not None else 9e9), expert, n_rows))
            reasons[expert] = {
                "weight": weight,
                "single_logloss": ll,
                "single_rows": n_rows,
                "uniform_logloss": uniform_logloss,
            }

    if not candidates:
        return list(experts), [], {}

    candidates.sort(key=lambda item: (item[0], item[1]))
    keep = list(experts)
    dropped: list[str] = []
    for _, _, expert, _ in candidates:
        if len(keep) <= min_keep_safe:
            break
        if expert in keep:
            keep.remove(expert)
            dropped.append(expert)

    if not dropped:
        return list(experts), [], {}
    return keep, dropped, {k: reasons[k] for k in dropped if k in reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced expert-weight optimizer (global + contextual).")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--days-back", type=int, default=120)
    parser.add_argument("--experts", nargs="*", default=DEFAULT_EXPERTS)
    parser.add_argument("--min-experts", type=int, default=2)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--random-trials", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contextual", action="store_true", default=True)
    parser.add_argument("--min-context-rows", type=int, default=40)
    parser.add_argument("--shrink-strength", type=float, default=80.0)
    parser.add_argument("--reg-lambda", type=float, default=0.08)
    parser.add_argument(
        "--auto-prune-weak-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically drop low-weight experts that also underperform on holdout.",
    )
    parser.add_argument(
        "--prune-weight-threshold",
        type=float,
        default=0.02,
        help="Candidate prune threshold for optimized global weight.",
    )
    parser.add_argument(
        "--prune-min-keep",
        type=int,
        default=4,
        help="Never keep fewer than this many experts after pruning.",
    )
    parser.add_argument(
        "--prune-min-valid-rows",
        type=int,
        default=40,
        help="Treat experts with fewer holdout rows than this as unstable for pruning logic.",
    )
    parser.add_argument("--report-out", default="models/expert_weights_optimized.json")
    parser.add_argument("--ensemble-out", default="models/ensemble_weights_optimized.json")
    parser.add_argument("--current-ensemble", default="models/ensemble_weights.json")
    parser.add_argument("--apply", action="store_true", help="Overwrite --current-ensemble with optimized weights.")
    parser.add_argument("--upload-db", action="store_true", help="Upload optimized ensemble weights to DB artifacts.")
    args = parser.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    experts = [str(e) for e in args.experts if str(e).strip()]
    if not experts:
        raise SystemExit("No experts provided.")

    frame = _load_frame(
        engine,
        days_back=int(args.days_back),
        experts=experts,
        min_experts=int(args.min_experts),
    )
    frame = _filter_frame_for_experts(frame, experts, min_experts=int(args.min_experts))
    if frame.empty:
        raise SystemExit("No resolved rows available for optimization.")
    base = _run_optimization_bundle(
        frame=frame,
        experts=experts,
        holdout_frac=float(args.holdout_frac),
        random_trials=int(args.random_trials),
        seed=int(args.seed),
        contextual=bool(args.contextual),
        min_context_rows=int(args.min_context_rows),
        shrink_strength=float(args.shrink_strength),
        reg_lambda=float(args.reg_lambda),
        current_ensemble_path=Path(args.current_ensemble),
    )

    pruned_experts: list[str] = []
    prune_reasons: dict[str, dict[str, float | int | None]] = {}
    if bool(args.auto_prune_weak_experts):
        keep_experts, dropped, reasons = _select_weak_experts_to_prune(
            experts=experts,
            global_weights=base["global_opt"],
            valid_metrics=base["valid_metrics"],
            min_keep=int(args.prune_min_keep),
            min_experts=int(args.min_experts),
            weight_threshold=float(args.prune_weight_threshold),
            min_valid_rows=int(args.prune_min_valid_rows),
        )
        if dropped and keep_experts:
            pruned_frame = _filter_frame_for_experts(frame, keep_experts, min_experts=int(args.min_experts))
            if len(pruned_frame) >= 20:
                experts = keep_experts
                frame = pruned_frame
                base = _run_optimization_bundle(
                    frame=frame,
                    experts=experts,
                    holdout_frac=float(args.holdout_frac),
                    random_trials=int(args.random_trials),
                    seed=int(args.seed),
                    contextual=bool(args.contextual),
                    min_context_rows=int(args.min_context_rows),
                    shrink_strength=float(args.shrink_strength),
                    reg_lambda=float(args.reg_lambda),
                    current_ensemble_path=Path(args.current_ensemble),
                )
                pruned_experts = list(dropped)
                prune_reasons = dict(reasons)

    train_frame = base["train_frame"]
    valid_frame = base["valid_frame"]
    global_opt = base["global_opt"]
    context_weights = base["context_weights"]
    train_metrics = base["train_metrics"]
    valid_metrics = base["valid_metrics"]

    # Build optimized ensemble state compatible with ContextualHedgeEnsembler.
    all_ctx = {
        _ctx_key_str(_ctx_key(s, l, n))
        for s, l, n in frame[["stat_type", "is_live", "n_eff"]].itertuples(index=False, name=None)
    }
    weights_state: dict[str, dict[str, float]] = {}
    for ctx_key in sorted(all_ctx):
        weights_state[ctx_key] = context_weights.get(ctx_key, global_opt)

    ensemble_state = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experts": list(experts),
        "eta": 0.2,
        "shrink_to_uniform": 0.01,
        "weights": weights_state,
    }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_valid": int(len(valid_frame)),
        "experts": list(experts),
        "pruned_experts": pruned_experts,
        "prune_reasons": prune_reasons,
        "weights_optimized_global": {k: float(v) for k, v in global_opt.items()},
        "weights_optimized_context_count": int(len(context_weights)),
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
        },
        "ensemble_state_path": str(Path(args.ensemble_out)),
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ensemble_path = Path(args.ensemble_out)
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_path.write_text(json.dumps(ensemble_state, indent=2), encoding="utf-8")

    if args.apply:
        current_path = Path(args.current_ensemble)
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.write_text(json.dumps(ensemble_state, indent=2), encoding="utf-8")

    if args.upload_db:
        try:
            from app.ml.artifact_store import upload_file

            target = Path(args.current_ensemble) if args.apply else ensemble_path
            upload_file(engine, model_name="ensemble_weights", file_path=target)
            print(f"[optimize_expert_weights] uploaded to DB artifact store: {target}")
        except Exception as exc:  # noqa: BLE001
            print(f"[optimize_expert_weights] DB upload failed (non-fatal): {exc}")

    print(f"[optimize_expert_weights] wrote report: {report_path}")
    print(f"[optimize_expert_weights] wrote ensemble state: {ensemble_path}")
    print(
        {
            "rows_total": report["rows_total"],
            "rows_train": report["rows_train"],
            "rows_valid": report["rows_valid"],
            "experts": report["experts"],
            "pruned_experts": report["pruned_experts"],
            "weights_optimized_global": report["weights_optimized_global"],
            "context_weights": report["weights_optimized_context_count"],
            "valid_metrics": valid_metrics,
            "applied": bool(args.apply),
        }
    )


if __name__ == "__main__":
    main()
