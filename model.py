from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n: int = 20
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class GraphConnectivityTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.read_in = nn.Linear(config.n, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.read_out = nn.Linear(config.d_model, config.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape [B, n, n], got {x.shape}")
        if x.shape[-1] != self.config.n or x.shape[-2] != self.config.n:
            raise ValueError(
                f"Expected last two dims to be ({self.config.n}, {self.config.n}), got {x.shape[-2:]}"
            )
        h = self.read_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        logits = self.read_out(h)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits

    @torch.no_grad()
    def predict_binary(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        logits = self.forward(x)
        return (logits > threshold).to(torch.int64)
