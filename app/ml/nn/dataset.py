from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class NNDataset(Dataset):
    frame: Any
    numeric: Any
    sequences: np.ndarray
    cat_maps: dict[str, dict[str, int]]
    cat_keys: list[str]
    _cat_ids: torch.Tensor = field(init=False, repr=False)
    _numeric: torch.Tensor = field(init=False, repr=False)
    _sequences: torch.Tensor = field(init=False, repr=False)
    _labels: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows = len(self.frame)
        cat_dim = len(self.cat_keys)
        cat_ids = np.zeros((rows, cat_dim), dtype=np.int64)
        for idx, key in enumerate(self.cat_keys):
            mapping = self.cat_maps.get(key, {})
            if key in self.frame.columns:
                values = self.frame[key].astype(str)
            else:
                values = pd.Series([""], index=self.frame.index, dtype="object")
            mapped = values.map(mapping).fillna(0).astype(np.int64).to_numpy()
            cat_ids[:, idx] = mapped
        self._cat_ids = torch.from_numpy(cat_ids)

        numeric_arr = self.numeric.to_numpy(dtype=np.float32, copy=False)
        self._numeric = torch.from_numpy(np.ascontiguousarray(numeric_arr))

        seq_arr = np.asarray(self.sequences, dtype=np.float32)
        self._sequences = torch.from_numpy(np.ascontiguousarray(seq_arr))

        if "over" in self.frame.columns:
            labels = pd.to_numeric(self.frame["over"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        else:
            labels = np.zeros((rows,), dtype=np.float32)
        self._labels = torch.from_numpy(labels.reshape(-1, 1))

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int):
        return (
            self._cat_ids[idx],
            self._numeric[idx],
            self._sequences[idx],
            self._labels[idx],
        )
