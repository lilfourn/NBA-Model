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
- `python -m scripts.ml.train_online_ensemble --source db --days-back 90 --log-weight-history`
- `python -m scripts.ml.prune_model_artifacts --keep 2`
- `python -m scripts.ops.resolve_projection_outcomes --days-back 21`

## Class Weighting (Next Retraining Cycle)

The models currently do not correct for class imbalance in the training data.
Since UNDER outcomes are more common (~53%) than OVER in the resolved dataset,
all models learn a slight UNDER bias that compounds through the ensemble.

Apply these changes during the next training cycle:

1. **TabDL**: Set `use_pos_weight: true` in the training config. The code
   already computes `pos_weight = max(1.0, neg / max(pos, 1.0))` but it is
   disabled by default.

2. **NN (GRU + Attention)**: Add `pos_weight` to `BCEWithLogitsLoss`:
   ```python
   pos_weight = torch.tensor([neg_count / pos_count])
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

3. **XGBoost**: Add `scale_pos_weight` to the training params:
   ```python
   params["scale_pos_weight"] = neg_count / pos_count
   ```

4. **LightGBM**: Add `scale_pos_weight` to the training params:
   ```python
   params["scale_pos_weight"] = neg_count / pos_count
   ```

5. **Meta-learner (LogisticRegression)**: Add `class_weight='balanced'`:
   ```python
   LogisticRegression(class_weight='balanced', ...)
   ```
