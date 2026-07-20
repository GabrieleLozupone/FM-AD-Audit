from __future__ import annotations

import torch
from torch import nn


class AttentiveHead(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_outputs: int = 1,
        n_covariates: int = 0,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(dim + n_covariates, num_outputs)

    def forward(self, tokens: torch.Tensor, covariates: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query.expand(tokens.size(0), -1, -1)
        pooled, attn_weights = self.attn(q, tokens, tokens)
        pooled = self.norm(pooled.squeeze(1))
        if covariates is not None and covariates.numel() > 0:
            pooled = torch.cat([pooled, covariates], dim=1)
        pooled = self.dropout(pooled)
        return self.linear(pooled), attn_weights
