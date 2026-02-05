# Scripts

The `scripts/` directory is organized by domain:

- `scripts/prizepicks/`: PrizePicks ingestion + normalization.
- `scripts/nba/`: NBA stats ingestion (nba.com/stats).
- `scripts/calibration/`: forecast backtest + calibration artifacts.
- `scripts/ml/`: model training + inference (LR, NN, ensemble).
- `scripts/pipelines/`: end-to-end pipelines that run multiple steps.
- `scripts/ops/`: maintenance/ops utilities (email, validation, caches).

## Stable Entry Points

Run scripts as modules from the repo root, for example:

- `python -m scripts.prizepicks.collect_prizepicks`
- `python -m scripts.nba.fetch_nba_stats --date-from 2026-02-01 --date-to 2026-02-03`
- `python -m scripts.ml.train_nn_model`
- `python -m scripts.ml.run_top_picks_ensemble --log-decisions`
- `python -m scripts.ml.train_online_ensemble --source db --days-back 90`
- `python -m scripts.ops.resolve_projection_outcomes --days-back 21`
