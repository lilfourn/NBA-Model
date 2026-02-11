"""Diagnose OVER/UNDER prediction bias across experts and ensemble.

Replays the scoring pipeline on the latest snapshot and reports:
- % OVER vs UNDER picks
- Per-expert mean P(over)
- Per-stat-type breakdown
- Shrinkage impact analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.services.scoring import score_ensemble  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose prediction bias.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--snapshot-id", default=None)
    ap.add_argument("--game-date", default=None)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--output", default="data/reports/bias_diagnostic.json")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    result = score_ensemble(
        engine,
        snapshot_id=args.snapshot_id,
        game_date=args.game_date,
        top=args.top,
        include_non_today=True,
        force=True,
    )

    if not result.picks:
        print("No picks returned. Check snapshot/game_date.")
        return

    picks = result.picks
    n = len(picks)
    print(f"\n{'='*60}")
    print(f"BIAS DIAGNOSTIC — {n} picks, snapshot={result.snapshot_id[:8]}...")
    print(f"{'='*60}")

    # 1. Overall OVER/UNDER distribution
    counts = Counter(p.pick for p in picks)
    n_over = counts.get("OVER", 0)
    n_under = counts.get("UNDER", 0)
    print("\n## Overall Distribution")
    print(f"  OVER:  {n_over:4d} ({100*n_over/n:.1f}%)")
    print(f"  UNDER: {n_under:4d} ({100*n_under/n:.1f}%)")

    # 2. prob_over distribution
    probs = [p.prob_over for p in picks]
    print("\n## prob_over (post-shrinkage)")
    print(f"  mean={np.mean(probs):.4f}  median={np.median(probs):.4f}")
    print(f"  min={np.min(probs):.4f}  max={np.max(probs):.4f}")
    print(f"  % < 0.5: {100*sum(1 for p in probs if p < 0.5)/n:.1f}%")

    # 3. Per-expert mean P(over)
    expert_keys = ["p_forecast_cal", "p_nn", "p_lr", "p_xgb", "p_lgbm", "p_meta"]
    print("\n## Per-Expert Mean P(over)")
    expert_stats = {}
    for key in expert_keys:
        vals = [getattr(p, key) for p in picks if getattr(p, key) is not None]
        if vals:
            mean_p = np.mean(vals)
            pct_under = 100 * sum(1 for v in vals if v < 0.5) / len(vals)
            print(f"  {key:20s}  n={len(vals):4d}  mean={mean_p:.4f}  %<0.5={pct_under:.1f}%")
            expert_stats[key] = {"n": len(vals), "mean": round(float(mean_p), 4), "pct_under": round(pct_under, 1)}
        else:
            print(f"  {key:20s}  n=   0  (not available)")
            expert_stats[key] = {"n": 0, "mean": None, "pct_under": None}

    # 4. Per-stat-type breakdown
    by_stat: dict[str, dict] = {}
    stat_counts: Counter = Counter()
    for p in picks:
        stat_counts[p.stat_type] += 1
        if p.stat_type not in by_stat:
            by_stat[p.stat_type] = {"over": 0, "under": 0, "probs": []}
        by_stat[p.stat_type]["over" if p.pick == "OVER" else "under"] += 1
        by_stat[p.stat_type]["probs"].append(p.prob_over)

    print("\n## Per-Stat-Type Breakdown")
    for stat in sorted(by_stat.keys(), key=lambda s: stat_counts[s], reverse=True):
        d = by_stat[stat]
        total = d["over"] + d["under"]
        mean_p = np.mean(d["probs"])
        print(f"  {stat:25s}  n={total:3d}  OVER={d['over']:3d} UNDER={d['under']:3d}  mean_p={mean_p:.4f}")
        by_stat[stat] = {
            "n": total,
            "over": d["over"],
            "under": d["under"],
            "mean_prob_over": round(float(mean_p), 4),
        }

    # 5. Shrinkage impact: how many picks would flip without shrinkage?
    # We can't replay without shrinkage here, but we can check how many are near 0.5
    near_boundary = sum(1 for p in probs if 0.45 <= p <= 0.55)
    print("\n## Shrinkage Impact")
    print(f"  Picks in [0.45, 0.55] range: {near_boundary} ({100*near_boundary/n:.1f}%)")
    print("  These are most vulnerable to shrinkage-induced direction flips.")

    # 6. Edge score distribution
    edges = [p.edge for p in picks]
    print("\n## Edge Score Distribution")
    print(f"  mean={np.mean(edges):.1f}  median={np.median(edges):.1f}")
    grades = Counter(p.grade for p in picks)
    for g in ["A+", "A", "B", "C", "D", "F"]:
        print(f"  {g}: {grades.get(g, 0)}")

    # 7. Ensemble weights audit
    weights_path = Path("models/ensemble_weights.json")
    weights_info = {}
    if weights_path.exists():
        wdata = json.loads(weights_path.read_text())
        experts_in_weights = wdata.get("experts", [])
        n_contexts = len(wdata.get("weights", {}))
        print("\n## Ensemble Weights Audit")
        print(f"  Experts in weights file: {experts_in_weights}")
        print(f"  Context buckets: {n_contexts}")
        expected = ["p_forecast_cal", "p_nn", "p_lr", "p_xgb", "p_lgbm"]
        missing = [e for e in expected if e not in experts_in_weights]
        if missing:
            print(f"  ⚠️  MISSING EXPERTS: {missing}")
        weights_info = {
            "experts": experts_in_weights,
            "n_contexts": n_contexts,
            "missing_experts": missing,
        }

    # Save report
    report = {
        "snapshot_id": result.snapshot_id,
        "n_picks": n,
        "overall": {"over": n_over, "under": n_under, "pct_over": round(100 * n_over / n, 1)},
        "prob_over_stats": {
            "mean": round(float(np.mean(probs)), 4),
            "median": round(float(np.median(probs)), 4),
        },
        "expert_stats": expert_stats,
        "by_stat_type": by_stat,
        "near_boundary_count": near_boundary,
        "ensemble_weights": weights_info,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {out}")


if __name__ == "__main__":
    main()
