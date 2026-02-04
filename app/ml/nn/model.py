from __future__ import annotations

import torch
from torch import nn


class GRUAttentionTabularClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_numeric: int,
        cat_cardinalities: list[int],
        cat_emb_dims: list[int],
        seq_d_in: int = 2,
        seq_hidden: int = 64,
        attn_dim: int = 32,
        mlp_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(cat_cardinalities) != len(cat_emb_dims):
            raise ValueError("cat_cardinalities and cat_emb_dims must match length")

        self.emb_layers = nn.ModuleList(
            [nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, cat_emb_dims)]
        )
        cat_out_dim = sum(cat_emb_dims)

        self.gru = nn.GRU(input_size=seq_d_in, hidden_size=seq_hidden, batch_first=True)

        self.attn = nn.Sequential(
            nn.Linear(seq_hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

        in_dim = num_numeric + cat_out_dim + seq_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, cat_ids: torch.Tensor, x_num: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        embs = []
        for idx, emb in enumerate(self.emb_layers):
            embs.append(emb(cat_ids[:, idx]))
        cat_vec = torch.cat(embs, dim=1)

        h_seq, _ = self.gru(x_seq)
        scores = self.attn(h_seq).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(h_seq * alpha.unsqueeze(-1), dim=1)

        feats = torch.cat([x_num, cat_vec, context], dim=1)
        return self.mlp(feats)
