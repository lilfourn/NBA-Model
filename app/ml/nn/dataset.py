from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NNDataset(Dataset):
    frame: Any
    numeric: Any
    sequences: np.ndarray
    cat_maps: dict[str, dict[str, int]]
    cat_keys: list[str]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        cat_vals = []
        for key in self.cat_keys:
            value = row.get(key)
            mapping = self.cat_maps.get(key, {})
            cat_vals.append(mapping.get(str(value), 0))
        cat_ids = torch.tensor(cat_vals, dtype=torch.long)

        num = self.numeric.iloc[idx].to_numpy(dtype=np.float32)
        num_tensor = torch.tensor(num, dtype=torch.float32)

        seq_tensor = torch.tensor(self.sequences[idx], dtype=torch.float32)

        y = float(row.get("over", 0))
        y_tensor = torch.tensor([y], dtype=torch.float32)

        return cat_ids, num_tensor, seq_tensor, y_tensor
