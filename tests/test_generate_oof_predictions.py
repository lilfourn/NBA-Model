from __future__ import annotations

from scripts.ml.generate_oof_predictions import _prequential_fold_indices


def test_prequential_folds_use_past_only_training() -> None:
    folds = _prequential_fold_indices(n_rows=20, n_folds=5)
    for train_idx, test_idx in folds:
        if not train_idx or not test_idx:
            continue
        assert max(train_idx) < min(test_idx)


def test_prequential_folds_have_disjoint_train_test() -> None:
    folds = _prequential_fold_indices(n_rows=23, n_folds=4)
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_prequential_folds_cover_all_rows_once_in_test() -> None:
    n_rows = 31
    folds = _prequential_fold_indices(n_rows=n_rows, n_folds=6)
    test_rows: list[int] = []
    for _, test_idx in folds:
        test_rows.extend(test_idx)
    assert sorted(test_rows) == list(range(n_rows))


def test_prequential_folds_handle_small_inputs() -> None:
    assert _prequential_fold_indices(n_rows=1, n_folds=5) == []
    assert _prequential_fold_indices(n_rows=10, n_folds=0) == []
