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

EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_lr", "p_xgb", "p_lgbm"]
ROLLING_WINDOW = 50
ACCURACY_ALERT_THRESHOLD = 0.48
LOGLOSS_ALERT_THRESHOLD = 0.75
WEIGHT_COLLAPSE_THRESHOLD = 0.80


def _logloss(y: int, p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _load_prediction_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in EXPERT_COLS + ["p_final"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_outcomes(engine, df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight outcome join — reuses logic from train_online_ensemble."""
    from scripts.ml.train_online_ensemble import _load_outcomes
    return _load_outcomes(engine, df)


def _check_rolling_metrics(
    joined: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Check rolling accuracy and logloss for each expert + ensemble."""
    alerts: list[dict] = []
    metrics: dict[str, dict] = {}

    for expert in EXPERT_COLS + ["p_final"]:
        if expert not in joined.columns:
            continue
        valid = joined.dropna(subset=[expert, "over_label"])
        if len(valid) < ROLLING_WINDOW:
            continue

        recent = valid.tail(ROLLING_WINDOW)
        probs = recent[expert].to_numpy(dtype=float)
        labels = recent["over_label"].to_numpy(dtype=int)

        picks = (probs >= 0.5).astype(int)
        accuracy = float((picks == labels).mean())
        avg_ll = float(np.mean([_logloss(int(y), float(p)) for y, p in zip(labels, probs)]))

        metrics[expert] = {
            "rolling_accuracy": round(accuracy, 4),
            "rolling_logloss": round(avg_ll, 4),
            "n": int(len(recent)),
        }

        if accuracy < ACCURACY_ALERT_THRESHOLD:
            alerts.append({
                "type": "low_accuracy",
                "expert": expert,
                "value": round(accuracy, 4),
                "threshold": ACCURACY_ALERT_THRESHOLD,
                "message": f"{expert} rolling {ROLLING_WINDOW}-bet accuracy {accuracy:.1%} < {ACCURACY_ALERT_THRESHOLD:.0%}",
            })
        if avg_ll > LOGLOSS_ALERT_THRESHOLD:
            alerts.append({
                "type": "high_logloss",
                "expert": expert,
                "value": round(avg_ll, 4),
                "threshold": LOGLOSS_ALERT_THRESHOLD,
                "message": f"{expert} rolling logloss {avg_ll:.3f} > {LOGLOSS_ALERT_THRESHOLD}",
            })

    return {"metrics": metrics, "alerts": alerts}


def _check_ensemble_weights(weights_path: str) -> list[dict]:
    """Check if any expert dominates the ensemble weights."""
    alerts = []
    path = Path(weights_path)
    if not path.exists():
        return alerts
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return alerts

    weights = data.get("weights") or data.get("global_weights") or {}
    if not weights:
        # Try extracting from context buckets
        buckets = data.get("context_weights", {})
        if buckets:
            all_w: dict[str, list[float]] = {}
            for bucket_weights in buckets.values():
                if isinstance(bucket_weights, dict):
                    for k, v in bucket_weights.items():
                        all_w.setdefault(k, []).append(float(v))
            weights = {k: float(np.mean(vs)) for k, vs in all_w.items()}

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
    ap.add_argument("--log-path", default="data/monitoring/prediction_log.csv")
    ap.add_argument("--ensemble-weights", default="models/ensemble_weights.json")
    ap.add_argument("--output", default="data/reports/model_health.json")
    ap.add_argument("--alert-email", action="store_true", help="Send email if alerts fire.")
    args = ap.parse_args()

    load_env()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"No prediction log at {log_path}")
        return

    df = _load_prediction_log(str(log_path))
    if df.empty:
        print("Prediction log is empty.")
        return

    engine = get_engine(args.database_url)
    joined = _load_outcomes(engine, df)

    all_alerts: list[dict] = []

    if not joined.empty:
        result = _check_rolling_metrics(joined)
        all_alerts.extend(result["alerts"])
        expert_metrics = result["metrics"]
        print(f"Rolling {ROLLING_WINDOW}-bet metrics:")
        for expert, m in expert_metrics.items():
            status = "OK"
            if m["rolling_accuracy"] < ACCURACY_ALERT_THRESHOLD:
                status = "ALERT"
            elif m["rolling_logloss"] > LOGLOSS_ALERT_THRESHOLD:
                status = "WARN"
            print(f"  {expert:<20} acc={m['rolling_accuracy']:.1%} ll={m['rolling_logloss']:.3f} [{status}]")
    else:
        expert_metrics = {}
        print("No outcomes available yet for rolling metrics.")

    weight_alerts = _check_ensemble_weights(args.ensemble_weights)
    all_alerts.extend(weight_alerts)

    report = {
        "prediction_log_rows": len(df),
        "outcome_rows": len(joined) if not joined.empty else 0,
        "rolling_window": ROLLING_WINDOW,
        "expert_metrics": expert_metrics,
        "alerts": all_alerts,
        "alert_count": len(all_alerts),
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


if __name__ == "__main__":
    main()
