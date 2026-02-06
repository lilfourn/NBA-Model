from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def default_embedding_dims(cat_cardinalities: Sequence[int], *, max_dim: int = 16) -> list[int]:
    dims: list[int] = []
    for cardinality in cat_cardinalities:
        base = int(round(float(max(2, cardinality)) ** 0.5))
        dims.append(min(max_dim, max(4, base)))
    return dims


class TabularMLPClassifier(nn.Module):
    """Simple embedding + MLP classifier for mixed tabular data."""

    def __init__(
        self,
        *,
        num_numeric: int,
        cat_cardinalities: Sequence[int],
        cat_emb_dims: Sequence[int],
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(cat_cardinalities) != len(cat_emb_dims):
            raise ValueError("cat_cardinalities and cat_emb_dims must be the same length")

        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), int(emb_dim)) for cardinality, emb_dim in zip(cat_cardinalities, cat_emb_dims)]
        )

        input_dim = int(num_numeric) + int(sum(int(dim) for dim in cat_emb_dims))
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden = int(hidden_dim)
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, cat_ids: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        x_num = x_num.float()
        parts = [x_num]
        if self.embeddings:
            for idx, emb in enumerate(self.embeddings):
                parts.append(emb(cat_ids[:, idx]))
        x = torch.cat(parts, dim=1)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)

