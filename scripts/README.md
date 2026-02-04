# Scripts

The `scripts/` directory is organized by domain:

- `scripts/prizepicks/`: PrizePicks ingestion + normalization.
- `scripts/nba/`: NBA stats ingestion (nba.com/stats).
- `scripts/calibration/`: forecast backtest + calibration artifacts.
- `scripts/ml/`: model training + inference (LR, NN, ensemble).
- `scripts/pipelines/`: end-to-end pipelines that run multiple steps.
- `scripts/ops/`: maintenance/ops utilities (email, validation, caches).

## Stable Entry Points

For convenience and backwards compatibility, the old paths (e.g. `scripts/train_nn_model.py`)
still exist as thin wrappers that call into the organized subpackages above.

