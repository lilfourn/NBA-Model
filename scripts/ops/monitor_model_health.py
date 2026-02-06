"""Model health monitoring: rolling accuracy, log loss, weight collapse, alerts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.stat_mappings import stat_value_from_row  # noqa: E402
from app.utils.names import normalize_name  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]
ROLLING_WINDOW = 50
ACCURACY_ALERT_THRESHOLD = 0.48
LOGLOSS_ALERT_THRESHOLD = 0.75
WEIGHT_COLLAPSE_THRESHOLD = 0.80
MIN_ALERT_EXPERT_WEIGHT = 0.03


def _logloss(y: int, p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _load_resolved_predictions(engine, *, days_back: int = 90) -> pd.DataFrame:
    """Load resolved predictions directly from DB."""
    from sqlalchemy import text as sa_text
    df = pd.read_sql(
        sa_text(
            """
            select
                id::text as prediction_id,
                prob_over as p_final,
                p_forecast_cal, p_nn, coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl, p_lr, p_xgb, p_lgbm,
                over_label,
                is_correct,
                outcome,
                coalesce(decision_time, created_at) as decision_time,
                created_at
            from projection_predictions
            where over_label is not null
              and actual_value is not null
              and outcome in ('over', 'under')
              and coalesce(decision_time, created_at) >= now() - (:days_back * interval '1 day')
            order by coalesce(decision_time, created_at) asc, created_at asc, id asc
            """
        ),
        engine,
        params={"days_back": int(max(1, days_back))},
    )
    for col in EXPERT_COLS + ["p_final"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_correct" in df.columns:
        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
    return df


def _load_ensemble_mean_weights(weights_path: str) -> dict[str, float]:
    path = Path(weights_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    weights = data.get("weights") or data.get("global_weights") or {}
    if isinstance(weights, dict) and weights and all(isinstance(v, dict) for v in weights.values()):
        nested_weights = weights
        all_w: dict[str, list[float]] = {}
        for bucket_weights in nested_weights.values():
            for expert, value in bucket_weights.items():
                all_w.setdefault(str(expert), []).append(float(value))
        weights = {expert: float(np.mean(values)) for expert, values in all_w.items()}
    if not weights:
        buckets = data.get("context_weights", {})
        if buckets:
            all_w: dict[str, list[float]] = {}
            for bucket_weights in buckets.values():
                if isinstance(bucket_weights, dict):
                    for k, v in bucket_weights.items():
                        all_w.setdefault(k, []).append(float(v))
            weights = {k: float(np.mean(vs)) for k, vs in all_w.items()}

    out: dict[str, float] = {}
    for expert, value in (weights or {}).items():
        try:
            out[str(expert)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _check_rolling_metrics(
    joined: pd.DataFrame,
    *,
    ensemble_weights: dict[str, float] | None = None,
    min_alert_weight: float = MIN_ALERT_EXPERT_WEIGHT,
) -> dict[str, list[dict]]:
    """Check rolling accuracy and logloss for each expert + ensemble."""
    alerts: list[dict] = []
    suppressed_alerts: list[dict] = []
    metrics: dict[str, dict] = {}

    for expert in EXPERT_COLS + ["p_final"]:
        if expert not in joined.columns:
            continue
        weight_map = ensemble_weights or {}
        has_weight_map = bool(weight_map)
        if expert in EXPERT_COLS:
            default_weight = 0.0 if has_weight_map else 1.0
            expert_weight = float(weight_map.get(expert, default_weight))
            alert_eligible = expert_weight >= float(min_alert_weight)
            if has_weight_map and expert not in weight_map:
                suppression_reason = "expert not present in active ensemble"
            else:
                suppression_reason = (
                    f"ensemble_weight={expert_weight:.6f} "
                    f"< min_alert_weight={float(min_alert_weight):.3f}"
                )
        else:
            expert_weight = 1.0
            alert_eligible = True
            suppression_reason = ""
        valid = joined.dropna(subset=[expert, "over_label"])
        if len(valid) < ROLLING_WINDOW:
            continue

        recent = valid.tail(ROLLING_WINDOW)
        probs = recent[expert].to_numpy(dtype=float)
        labels = recent["over_label"].to_numpy(dtype=int)

        if expert == "p_final" and "is_correct" in recent.columns:
            is_correct = pd.to_numeric(recent["is_correct"], errors="coerce").dropna()
            if len(is_correct) == len(recent):
                accuracy = float(is_correct.mean())
            else:
                picks = (probs >= 0.5).astype(int)
                accuracy = float((picks == labels).mean())
        else:
            picks = (probs >= 0.5).astype(int)
            accuracy = float((picks == labels).mean())
        avg_ll = float(np.mean([_logloss(int(y), float(p)) for y, p in zip(labels, probs)]))

        metrics[expert] = {
            "rolling_accuracy": round(accuracy, 4),
            "rolling_logloss": round(avg_ll, 4),
            "n": int(len(recent)),
            "ensemble_weight": round(expert_weight, 6),
            "alert_eligible": bool(alert_eligible),
        }

        if accuracy < ACCURACY_ALERT_THRESHOLD:
            payload = {
                "type": "low_accuracy",
                "expert": expert,
                "value": round(accuracy, 4),
                "threshold": ACCURACY_ALERT_THRESHOLD,
                "message": f"{expert} rolling {ROLLING_WINDOW}-bet accuracy {accuracy:.1%} < {ACCURACY_ALERT_THRESHOLD:.0%}",
            }
            if alert_eligible:
                alerts.append(payload)
            else:
                payload["suppression_reason"] = suppression_reason
                suppressed_alerts.append(payload)
        if avg_ll > LOGLOSS_ALERT_THRESHOLD:
            payload = {
                "type": "high_logloss",
                "expert": expert,
                "value": round(avg_ll, 4),
                "threshold": LOGLOSS_ALERT_THRESHOLD,
                "message": f"{expert} rolling logloss {avg_ll:.3f} > {LOGLOSS_ALERT_THRESHOLD}",
            }
            if alert_eligible:
                alerts.append(payload)
            else:
                payload["suppression_reason"] = suppression_reason
                suppressed_alerts.append(payload)

    return {"metrics": metrics, "alerts": alerts, "suppressed_alerts": suppressed_alerts}


def _check_ensemble_weights(weights_path: str) -> list[dict]:
    """Check if any expert dominates the ensemble weights."""
    alerts = []
    weights = _load_ensemble_mean_weights(weights_path)

    if not weights:
        return alerts

    total = sum(weights.values())
    if total <= 0:
        return alerts

    for expert, w in weights.items():
        ratio = w / total
        if ratio > WEIGHT_COLLAPSE_THRESHOLD:
            alerts.append({
                "type": "weight_collapse",
                "expert": expert,
                "value": round(ratio, 4),
                "threshold": WEIGHT_COLLAPSE_THRESHOLD,
                "message": f"{expert} has {ratio:.0%} of ensemble weight (collapse risk)",
            })

    return alerts


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor model health from prediction log.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--days-back", type=int, default=90)
    ap.add_argument("--ensemble-weights", default="models/ensemble_weights.json")
    ap.add_argument("--output", default="data/reports/model_health.json")
    ap.add_argument("--alert-email", action="store_true", help="Send email if alerts fire.")
    args = ap.parse_args()

    load_env()

    engine = get_engine(args.database_url)
    joined = _load_resolved_predictions(engine, days_back=args.days_back)

    all_alerts: list[dict] = []
    ensemble_weights = _load_ensemble_mean_weights(args.ensemble_weights)

    if not joined.empty:
        result = _check_rolling_metrics(
            joined,
            ensemble_weights=ensemble_weights,
            min_alert_weight=MIN_ALERT_EXPERT_WEIGHT,
        )
        all_alerts.extend(result["alerts"])
        suppressed_alerts = result.get("suppressed_alerts", [])
        expert_metrics = result["metrics"]
        print(f"Rolling {ROLLING_WINDOW}-bet metrics:")
        for expert, m in expert_metrics.items():
            status = "OK"
            if not bool(m.get("alert_eligible", True)):
                status = "INFO"
            elif m["rolling_accuracy"] < ACCURACY_ALERT_THRESHOLD:
                status = "ALERT"
            elif m["rolling_logloss"] > LOGLOSS_ALERT_THRESHOLD:
                status = "WARN"
            weight_text = ""
            if expert in EXPERT_COLS:
                weight_text = f" w={float(m.get('ensemble_weight', 0.0)):.3f}"
            print(
                f"  {expert:<20} acc={m['rolling_accuracy']:.1%} ll={m['rolling_logloss']:.3f}"
                f"{weight_text} [{status}]"
            )
    else:
        expert_metrics = {}
        suppressed_alerts = []
        print("No outcomes available yet for rolling metrics.")

    weight_alerts = _check_ensemble_weights(args.ensemble_weights)
    all_alerts.extend(weight_alerts)

    report = {
        "prediction_log_rows": len(joined),
        "outcome_rows": len(joined) if not joined.empty else 0,
        "rolling_window": ROLLING_WINDOW,
        "expert_metrics": expert_metrics,
        "alerts": all_alerts,
        "suppressed_alerts": suppressed_alerts,
        "alert_count": len(all_alerts),
        "suppressed_alert_count": len(suppressed_alerts),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nHealth report -> {output}")

    if all_alerts:
        print(f"\n{'!'*50}")
        print(f"  {len(all_alerts)} ALERT(S) FIRED:")
        for a in all_alerts:
            print(f"  - {a['message']}")
        print(f"{'!'*50}")

        if args.alert_email:
            email_script = ROOT / "scripts" / "ops" / "send_email.py"
            if email_script.exists():
                import subprocess
                body = "\n".join(a["message"] for a in all_alerts)
                body_file = output.parent / "health_alert_body.txt"
                body_file.write_text(body, encoding="utf-8")
                subprocess.run(
                    [sys.executable, str(email_script),
                     "--subject", "Model Health Alert",
                     "--body-file", str(body_file)],
                    check=False,
                )
    else:
        print("\nAll models healthy.")
    if suppressed_alerts:
        print(f"Suppressed alerts (audit only): {len(suppressed_alerts)}")


if __name__ == "__main__":
    main()
