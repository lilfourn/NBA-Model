"""Generate matplotlib report images for model performance visualization.

Produces:
1. Weight evolution chart (hedge + thompson over time)
2. Per-expert rolling accuracy chart
3. Calibration reliability diagram
4. Drift history summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def _plot_weight_evolution(history_path: str, output_dir: Path) -> None:
    """Plot ensemble weight evolution over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime

    p = Path(history_path)
    if not p.exists():
        print(f"  No weight history at {p}")
        return

    entries = []
    for line in p.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(entries) < 2:
        print(f"  Not enough weight history entries ({len(entries)})")
        return

    timestamps = [e.get("timestamp", "") for e in entries]
    dates = []
    for ts in timestamps:
        try:
            dates.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except (ValueError, AttributeError):
            dates.append(None)

    # Hedge weights
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Hedge average weights
    hedge_experts: set[str] = set()
    for e in entries:
        if "hedge_avg" in e:
            hedge_experts.update(e["hedge_avg"].keys())

    if hedge_experts:
        for expert in sorted(hedge_experts):
            vals = [e.get("hedge_avg", {}).get(expert, np.nan) for e in entries]
            valid_dates = [d for d, v in zip(dates, vals) if d is not None and np.isfinite(v)]
            valid_vals = [v for d, v in zip(dates, vals) if d is not None and np.isfinite(v)]
            if valid_dates:
                axes[0].plot(valid_dates, valid_vals, marker=".", label=expert, alpha=0.8)
        axes[0].set_ylabel("Weight")
        axes[0].set_title("Hedge Ensemble Weights Over Time")
        axes[0].legend(fontsize=8, ncol=3)
        axes[0].grid(True, alpha=0.3)

    # Panel 2: Thompson weights
    ts_experts: set[str] = set()
    for e in entries:
        if "thompson_avg" in e:
            ts_experts.update(e["thompson_avg"].keys())

    if ts_experts:
        for expert in sorted(ts_experts):
            vals = [e.get("thompson_avg", {}).get(expert, np.nan) for e in entries]
            valid_dates = [d for d, v in zip(dates, vals) if d is not None and np.isfinite(v)]
            valid_vals = [v for d, v in zip(dates, vals) if d is not None and np.isfinite(v)]
            if valid_dates:
                axes[1].plot(valid_dates, valid_vals, marker=".", label=expert, alpha=0.8)
        axes[1].set_ylabel("Mean(α/(α+β))")
        axes[1].set_title("Thompson Sampling Expert Quality Over Time")
        axes[1].legend(fontsize=8, ncol=3)
        axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out = output_dir / "weight_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def _plot_drift_history(drift_path: str, output_dir: Path) -> None:
    """Plot drift detection metrics over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(drift_path)
    if not p.exists():
        print(f"  No drift report at {p}")
        return

    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  Invalid drift report JSON")
        return

    checks = report.get("checks", [])
    if not checks:
        print("  No drift checks in report")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [c["check_type"] for c in checks]
    values = [c["metric_value"] for c in checks]
    thresholds = [c["threshold"] for c in checks]
    colors = ["red" if c["is_drifted"] else "green" for c in checks]

    x = np.arange(len(names))
    ax.bar(x, values, color=colors, alpha=0.7, label="Metric Value")
    ax.scatter(x, thresholds, color="black", marker="_", s=200, linewidths=2, zorder=5, label="Threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Metric Value")
    ax.set_title("Drift Detection Results")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = output_dir / "drift_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def _plot_calibration_diagram(output_dir: Path) -> None:
    """Plot calibration reliability diagram from feature importance data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Try to load bias diagnostic data
    bias_path = Path("data/reports/bias_diagnostic.json")
    if not bias_path.exists():
        print("  No bias diagnostic data for calibration diagram")
        return

    try:
        data = json.loads(bias_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  Invalid bias diagnostic JSON")
        return

    bins = data.get("calibration_bins", [])
    if not bins:
        print("  No calibration bins in diagnostic data")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    mean_pred = [b.get("mean_predicted", 0) for b in bins]
    mean_actual = [b.get("mean_actual", 0) for b in bins]
    counts = [b.get("count", 0) for b in bins]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(mean_pred, mean_actual, s=[max(10, c / 2) for c in counts],
               alpha=0.7, c="steelblue", edgecolors="navy")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "calibration_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate model performance reports.")
    ap.add_argument("--output-dir", default="data/reports", help="Output directory for charts")
    ap.add_argument("--weight-history", default="data/reports/weight_history.jsonl")
    ap.add_argument("--drift-report", default="data/reports/drift_report.json")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reports...")
    _plot_weight_evolution(args.weight_history, output_dir)
    _plot_drift_history(args.drift_report, output_dir)
    _plot_calibration_diagram(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
