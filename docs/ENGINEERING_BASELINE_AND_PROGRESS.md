# NBA Stats Project: Engineering Baseline and 60% Hit-Rate Tracker

Last updated: 2026-02-11 (America/Chicago)

## Goal

Target: achieve and sustain **>= 60% hit rate accuracy** on resolved picks.

For this document, "hit rate" means:
- Pick is `OVER` if `prob_over >= 0.5`, else `UNDER`.
- A hit occurs when the pick direction matches resolved `over_label` (or stored `is_correct` when present).

---

## Current Baseline Snapshot

Two different metric views currently exist and disagree:

1. Live DB monitor (fresh run on 2026-02-06)
- Source: `scripts.ops.monitor_model_health --days-back 90`
- Rows scored: 5223
- Ensemble (`p_final`) rolling-50 accuracy: **0.48**
- Ensemble (`p_final`) rolling-50 logloss: **0.8002**

2. Local CSV monitoring file
- Source: `data/monitoring/prediction_log.csv`
- Rows with resolved label in file: 416
- Overall hit rate (file-level): **0.5769**
- Rolling-50 hit rate (file-level): **0.64**

Interpretation:
- The system currently has **metric-source fragmentation**.
- The live DB monitor is the operational source used by cron/alerts and is currently below the 60% target.
- The local CSV appears to reflect a narrower or older subset.

---

## End-to-End Architecture

## 1) Data Ingestion

PrizePicks:
- Client: `app/clients/prizepicks.py`
- Loader: `app/db/prizepicks_loader.py`
- Pipeline entrypoint: `scripts/prizepicks/collect_prizepicks.py`
- Filters enforced during load:
  - only standard odds (`odds_type=0`)
  - excludes goblin/demon
  - excludes combos
- Tracks line movement using prior snapshot:
  - `line_score_prev`
  - `line_score_delta`
  - `line_movement` (`new/up/down/same`)

NBA stats:
- Client: `app/clients/nba_stats.py`
- Loader: `app/db/nba_loader.py`
- Pipeline entrypoint: `scripts/nba/fetch_nba_stats.py`
- Failure handling:
  - bulk fetch fallback to day-by-day chunking
  - optional degraded-empty mode
  - can reuse existing DB rows if upstream unavailable

Fallback player sources:
- StatMuse and Basketball Reference via `scripts/nba/fetch_nba_player_gamelogs.py`
- Intended as fallback/validation, not primary when NBA Stats is healthy.

---

## 2) Data Model and Storage

Core tables:
- `snapshots`, `projections`, `projection_features`
- `nba_players`, `nba_games`, `nba_player_game_stats`
- `projection_predictions` (logged picks and resolved outcomes)
- `model_runs` (training history metrics)
- `model_artifacts` (DB-backed artifact storage)

Prediction lifecycle:
1. Score snapshot projections.
2. Log predictions to `projection_predictions`.
3. Later resolve outcomes using boxscores (`resolve_projection_outcomes`).
4. Train/update ensemble from resolved rows.

---

## 3) Feature Layer

Training data assembly:
- `app/ml/dataset.py::load_training_data`
- Joins PrizePicks projections to NBA games and player boxscores.
- Includes combo decomposition for historical stat reconstruction.
- Uses name/team override maps:
  - `data/name_overrides.json`
  - `data/team_abbrev_overrides.json`

History features:
- `app/ml/feature_engineering.py`
- Includes:
  - historical mean/std
  - rest/B2B
  - rolling windows (3/5/10)
  - trend slope
  - coefficient of variation
  - hot/cold streak counts
  - line-relative features
- Leakage protection:
  - excludes same-day game stats by date-normalized cutoff
  - excludes live/in-game rows for training

Opponent features:
- `app/ml/opponent_features.py`
- Rolling defensive allowances and opponent rank context.

---

## 4) Expert Models

Forecast expert:
- `app/modeling/stat_forecast.py`
- Weighted recency model with short/long decay windows.
- Stabilized toward league priors via effective sample size.
- Produces `mu_hat`, `sigma_hat`, `n_eff`, and calibrated probability when calibration exists.

ML experts:
- Logistic regression: `app/ml/train.py` + `app/ml/infer_baseline.py`
- XGBoost: `app/ml/xgb/*`
- LightGBM: `app/ml/lgbm/*`
- NN (GRU-attention): `app/ml/nn/*`
- TabDL (MLP embeddings): `app/ml/tabdl/*`

Meta-learner:
- `app/ml/meta_learner.py`
- Trained on OOF predictions (`data/oof_predictions.csv`) plus `n_eff_log`.

Conformal:
- `app/modeling/conformal.py`
- Stored in model artifacts and used as additional confidence signal.

---

## 5) Ensemble and Ranking Math

Primary scoring path:
- `app/services/scoring.py::score_ensemble`

Current ensemble behavior:
- Expert probabilities are collected from `p_forecast_cal`, `p_nn`, `p_tabdl`, `p_lr`, `p_xgb`, `p_lgbm`.
- Optional stacking meta-model is used when available; otherwise logit-mean fallback is used.
- Expert probabilities are clipped to `[0.25, 0.75]` before combination to prevent outlier domination.
- Stat types in `EXCLUDED_STAT_TYPES` are skipped; `PRIOR_ONLY_STAT_TYPES` are scored from context prior only and never published.

Probability shrinkage and calibration:
- Shrinkage is logit-space blending toward context prior with neutral fallback anchor `0.50`.
- Shrink strength increases when `n_eff` is low (`SHRINK_MAX`) and decreases with stronger history (`SHRINK_MIN`).
- Per-stat isotonic recalibration is applied via `StatTypeCalibrator`.

Publishability gates:
- Picks must pass confidence threshold, conformal ambiguity filter, expert diversity, minimum `n_eff`, forecast-edge guardrail, and minimum edge score.
- If publishable set is empty, a soft fallback returns ranked best-effort picks and marks fallback metadata in response.

Edge/grade:
- Composite edge score combines forecast edge, data quality, expert disagreement signal, confidence band, uncertainty, and conformal bonus.
- Grades map edge into: `A+`, `A`, `B`, `C`, `D`, `F`.
- Direction-imbalance guardrail demotes dominant-direction picks near context prior when published picks are overly one-sided.

---

## 6) Serving and Frontend Integration

Backend:
- FastAPI app: `app/main.py`
- Picks endpoint: `app/api/picks.py`
- Stats endpoints: `app/api/stats.py`
- Job endpoints: `app/api/jobs.py`

Frontend:
- API client: `frontend/lib/api.ts`
- Main pages:
  - picks dashboard: `frontend/app/page.tsx`
  - stats dashboard: `frontend/app/stats/page.tsx`

Operational note:
- Scoring has in-process cache TTL of 5 minutes.

---

## 7) Training and Ops Orchestration

Host cron:
- Collect every 3h: `scripts/cron_collect.sh`
- Train daily: `scripts/cron_train.sh`

Modal schedules:
- Collect every 3h CT
- Train daily 14:00 CT
- Weekly calibration Monday 10:00 CT
- File: `modal_app.py`

Training pipeline includes:
- baseline, NN, TabDL, XGB, LGBM
- OOF generation + meta-learner
- online ensemble training
- health monitor
- drift check
- conditional retuning on drift signal

---

## Known Gaps Blocking Reliable 60% Tracking

1. Metric source inconsistency
- `data/reports/model_health.json` can be stale relative to live DB state.
- CSV and DB views do not currently represent the same sample.

2. Drift report baseline window is often empty
- `scripts/ops/check_drift.py` uses `created_at` windows.
- Backfills or bulk inserts can collapse baseline rows to zero.
- Result: drift detection may report "no drift" with no valid comparison.

3. Live performance below target on operational monitor
- Latest live rolling-50 is 0.48 for `p_final` from DB monitor run on 2026-02-06.

---

## 60% Progress Tracker

## Definition of Success (proposed)

Use DB as source of truth (`projection_predictions` resolved rows):
- Primary: rolling-50 hit rate >= 0.60
- Stability: rolling-200 hit rate >= 0.58
- Quality guardrail: rolling-50 logloss <= 0.70

If you want stricter criteria, raise rolling windows (for example 100/300).

## Update Commands

Live health snapshot:

```bash
python -m scripts.ops.monitor_model_health --days-back 90 --ensemble-weights models/ensemble_weights.json --output data/reports/model_health.json
```

Drift snapshot:

```bash
python -m scripts.ops.check_drift --recent-days 7 --baseline-days 30 --output data/reports/drift_report.json
```

CSV quick check:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/monitoring/prediction_log.csv")
df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
df["p_final"] = pd.to_numeric(df["p_final"], errors="coerce")
resolved = df.dropna(subset=["over_label", "p_final"])
hit = ((resolved["p_final"] >= 0.5) == (resolved["over_label"] == 1)).astype(float)
print("n=", len(hit), "overall=", round(float(hit.mean()), 4), "rolling50=", round(float(hit.tail(50).mean()), 4))
PY
```

## Experiment Log Template

For each training cycle, append:
- Date/time (CT)
- Data window used
- Models retrained
- Ensemble weights summary
- Rolling-50 hit rate
- Rolling-200 hit rate
- Rolling-50 logloss
- Notes on what changed

---

## Canonical Definitions (Single Source of Truth)

### Pick Tiers

- **Scored**: every row for which the model generates `p_final`. This is the broadest population and includes all resolved predictions in `projection_predictions`.
- **Actionable (Published)**: scored rows where `p_pick = max(p_final, 1 - p_final) >= 0.57`. These are the picks surfaced to users and used for hit-rate reporting.
- **Placed**: (future) rows representing actual bets placed. Not yet tracked.

### Push and Void Handling

- **Push**: `actual_value == line_score` exactly. `over_label = NULL`, `outcome = "push"`, `is_correct = NULL`. Excluded from all accuracy/logloss computations.
- **Void**: game has boxscores but player has no row (DNP). Forced to push. Same exclusion rules apply.
- **Resolved**: `outcome IN ('over', 'under')` and `over_label IS NOT NULL` and `actual_value IS NOT NULL`.

### Line Evaluation

- The line used for evaluation is `line_score` in `projection_predictions`, which is the line at decision time (when the prediction was made). This is aliased as `line_at_decision` in the canonical view.
- Outcome resolution in `app/db/prediction_logs.py` compares `actual_value` (from boxscores) against `line_score` to determine over/under/push.

### Exact Row Filters Used in Monitoring

All monitoring scripts, API endpoints, and drift checks use the canonical view `vw_resolved_picks_canonical` which applies these filters:

```sql
WHERE outcome IN ('over', 'under')
  AND over_label IS NOT NULL
  AND actual_value IS NOT NULL
```

---

## Canonical Metric Pipeline

### DB View

All metric consumers MUST query `vw_resolved_picks_canonical` (defined in migration `0011_canonical_resolved_view.py`).

Direct queries to `projection_predictions` for metric computation are NOT allowed except as a fallback when the view does not exist (pre-migration).

### Consumers Using the Canonical View

| Script/Endpoint | Purpose |
|---|---|
| `scripts/ops/monitor_model_health.py` | Rolling accuracy, logloss, inversion tests, calibration diagnostics |
| `app/api/stats.py` `/hit-rate` | Dashboard hit rate and rolling charts |
| `scripts/ops/check_drift.py` | Drift detection (performance, distribution, calibration) |
| `scripts/ops/ablation_report.py` | Per-component comparison metrics |

### Metrics Version

All report JSON outputs include a `metrics_version` string (currently `"2.0.0"`). Bump this version when changing:
- Push/void handling rules
- Threshold values
- Population filters
- Metric formulas

---

## Required Logging Fields Checklist

The following columns in `projection_predictions` are required for evaluation integrity:

- `prob_over` (Numeric) -- post-shrink p_final
- `p_raw` (Numeric) -- pre-shrink ensemble probability (added in migration 0010)
- `line_score` (Numeric) -- line at decision time
- `decision_time` (DateTime) -- when the prediction was made
- `stat_type` (Text) -- stat category (PTS, REB, AST, 3PM, etc.)
- `over_label` (Integer) -- 1 if actual > line, 0 if actual < line, NULL for push
- `outcome` (Text) -- "over", "under", or "push"
- `is_correct` (Boolean) -- whether the pick direction matched the outcome
- `actual_value` (Numeric) -- resolved stat value from boxscores
- `resolved_at` (DateTime) -- when the outcome was resolved
- `n_eff` (Numeric) -- effective sample size (data quality indicator)
- `rank_score` (Numeric) -- risk-adjusted confidence score
- `p_forecast_cal`, `p_nn`, `p_tabdl`, `p_lr`, `p_xgb`, `p_lgbm` -- individual expert probabilities

---

## Suggested Next Technical Priorities

1. Run inversion tests
- Execute `monitor_model_health.py` and check if `1-p_final` outperforms `p_final` in the inversion test output.

2. Verify context priors
- Run `app/ml/context_prior.py::compute_context_priors_from_db()` to generate `models/context_priors.json` and inspect per-stat-type base rates.

3. Tune abstain threshold
- Analyze the trade-off between coverage and hit rate at different `PICK_THRESHOLD` values.

4. Run ablation report
- Execute `scripts/ops/ablation_report.py` and verify hybrid final outperforms simpler components.

5. Segment calibration review
- Check the per-stat-type calibration diagnostics in the health report for systematic miscalibration.

---

## Fast File Map

- API serving: `app/main.py`, `app/api/picks.py`, `app/api/stats.py`
- Scoring pipeline: `app/services/scoring.py`
- Context prior: `app/ml/context_prior.py`
- Forecast math: `app/modeling/stat_forecast.py`, `app/modeling/forecast_calibration.py`
- Ensembles: `app/modeling/online_ensemble.py`, `app/modeling/thompson_ensemble.py`, `app/modeling/gating_model.py`, `app/modeling/hybrid_ensemble.py`
- Outcome resolution: `app/db/prediction_logs.py`, `scripts/ops/resolve_projection_outcomes.py`
- Training entrypoints: `scripts/ml/train_baseline_model.py`, `scripts/ml/train_nn_model.py`, `scripts/ml/train_tabdl_model.py`, `scripts/ml/train_xgb_model.py`, `scripts/ml/train_lgbm_model.py`, `scripts/ml/train_online_ensemble.py`, `scripts/ml/train_meta_learner.py`
- Monitoring: `scripts/ops/monitor_model_health.py`, `scripts/ops/check_drift.py`, `scripts/ops/ablation_report.py`
- Schedules: `scripts/cron_collect.sh`, `scripts/cron_train.sh`, `modal_app.py`
- Frontend API integration: `frontend/lib/api.ts`
- Canonical DB view: `alembic/versions/0011_canonical_resolved_view.py`
