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
    # Attention kind: "softmax" (default, classical) or "normalized_relu"
    # (= Ye et al. 2026 / "Right Data" paper: α = (1/n)·ReLU(QK^T/√d_h)).
    # The normalized-ReLU variant is the canonical choice for the
    # matrix-powering construction in the Disentangled-Transformer paper.
    attn_kind: str = "softmax"
    # ── RoBERTa-faithful options (used only by RobertaGraphTransformer) ──
    # "post" = BERT/RoBERTa post-LayerNorm (LN after residual add);
    # "pre"  = pre-LayerNorm (the idealised Definition A.1 form).
    norm_style: str = "pre"
    layer_norm_eps: float = 1e-5      # RoBERTa default
    init_std: float = 0.02            # RoBERTa weight-init std
    # Read-out for connectivity logits:
    #   "linear"     -> R_ij from h_i · W_out (the paper's read-out), or
    #   "similarity" -> R_ij = scale * <h_i_norm, h_j_norm> + bias  (spectral /
    #                   Laplacian-style: connectivity == embedding similarity).
    readout: str = "linear"


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 attn_kind: str = "softmax") -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if attn_kind not in ("softmax", "normalized_relu"):
            raise ValueError(f"unknown attn_kind: {attn_kind}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_kind = attn_kind
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
        if self.attn_kind == "softmax":
            attn = F.softmax(scores, dim=-1)
        else:   # normalized_relu  -- (1/n) * ReLU(QK^T/√d_h)
            attn = F.relu(scores) / n
        self.last_attn = attn.detach()        # [B, heads, n, n] real learned weights
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0,
                 attn_kind: str = "softmax") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, attn_kind=attn_kind)
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
                    attn_kind=config.attn_kind,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.readout_kind = getattr(config, "readout", "linear")
        if self.readout_kind == "similarity":
            # connectivity == cosine similarity of node embeddings
            self.sim_scale = nn.Parameter(torch.tensor(10.0))
            self.sim_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.read_out = nn.Linear(config.d_model, config.n)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Node embeddings H = h^(L) (after the final LayerNorm)."""
        if x.ndim != 3 or x.shape[-1] != self.config.n or x.shape[-2] != self.config.n:
            raise ValueError(f"Expected input [B, {self.config.n}, {self.config.n}], got {x.shape}")
        h = self.read_in(x)
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h)

    def hidden_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return [h^0 (read-in), h^1, ..., h^L, final_norm(h^L)] for analysis."""
        states = []
        h = self.read_in(x); states.append(h)
        for block in self.blocks:
            h = block(h); states.append(h)
        states.append(self.final_norm(h))
        return states

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """The node embeddings the read-out sees (H = h^(L))."""
        return self._trunk(x)

    def forward_and_embeddings(self, x: torch.Tensor):
        h = self._trunk(x)
        if self.readout_kind == "similarity":
            hn = F.normalize(h, dim=-1)
            logits = self.sim_scale * torch.matmul(hn, hn.transpose(-1, -2)) + self.sim_bias
        else:
            logits = self.read_out(h)
            logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_and_embeddings(x)[0]

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The real learned attention weights, one tensor [B, heads, n, n] per layer."""
        self.forward(x)
        return [blk.attn.last_attn for blk in self.blocks]

    @torch.no_grad()
    def predict_binary(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        logits = self.forward(x)
        return (logits > threshold).to(torch.int64)


def laplacian_smoothness(H: torch.Tensor, adj_no_loops: torch.Tensor) -> torch.Tensor:
    """Mean per-graph Laplacian Dirichlet energy of node embeddings:
        Tr(H^T L H) = sum_{(i,j) in E} ||h_i - h_j||^2 ,   L = D - A,
    normalised by the number of (undirected) edges. `adj_no_loops` is the 0/1
    adjacency WITHOUT self-loops, shape [B, n, n]."""
    deg = adj_no_loops.sum(-1)                                   # [B, n]
    LH = deg.unsqueeze(-1) * H - torch.matmul(adj_no_loops, H)   # (D - A) H
    energy = (H * LH).sum(dim=(-2, -1))                          # Tr(H^T L H) per graph
    n_edges = adj_no_loops.sum(dim=(-2, -1)) / 2.0 + 1e-6
    return (energy / n_edges).mean()


class GraphBinaryClassifier(nn.Module):
    """Graph-level binary classifier sharing the exact same trunk as
    GraphConnectivityTransformer (read-in, normalized-ReLU/softmax attention
    blocks, final LayerNorm). Instead of an n x n read-out, it mean-pools the
    node embeddings and maps them to a single logit.

    Used for the "1 chain vs 2 chains" task (= 1 vs 2 connected components):
    target 1 -> two components (2chain), target 0 -> one component (1chain).
    """

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
                    attn_kind=config.attn_kind,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.n or x.shape[-2] != self.config.n:
            raise ValueError(f"Expected input [B, {self.config.n}, {self.config.n}], got {x.shape}")
        h = self.read_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        pooled = h.mean(dim=1)                 # mean over nodes -> [B, d_model]
        return self.head(pooled).squeeze(-1)   # -> [B]


# ──────────────────────────────────────────────────────────────────────────────
# RoBERTa-faithful variant
#
# Ye et al. 2026 (App. D.1) state: "we adopt the implementation from RoBERTa
# (Liu et al., 2019) with single-head per-layer and using normalized ReLU
# activation function as defined in Definition A.1." RoBERTa is a concrete
# BERT-style encoder; "adopting" it brings engineering details that the clean
# A.1 math omits. This module reproduces those details faithfully:
#   * BERT/RoBERTa post-LayerNorm blocks (LN after the residual add)
#   * GELU feed-forward with intermediate width d_ff (= 4*d for RoBERTa)
#   * biases in every linear layer; embedding-level LayerNorm + dropout
#   * dropout 0.1 (config.dropout); LayerNorm eps 1e-5; weight init N(0, 0.02)
# and the two graph-specific adaptations:
#   * a linear read-in (n -> d) instead of token/positional embeddings
#     (Definition A.1 uses NO positional encoding)
#   * normalized-ReLU attention  alpha = (1/n) ReLU(QK^T / sqrt(d_h))
#   * a linear read-out (d -> n) giving the n x n connectivity logits, with
#     NO final LayerNorm and NO output symmetrisation (matching A.1 exactly).
# norm_style="pre" is also supported to compare against the idealised A.1 form.
# ──────────────────────────────────────────────────────────────────────────────


class _ReluSelfAttention(nn.Module):
    """Single-/multi-head self-attention with normalized-ReLU (or softmax)
    scores. Returns the context only (the output dense lives in the block)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, attn_kind: str) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if attn_kind not in ("softmax", "normalized_relu"):
            raise ValueError(f"unknown attn_kind: {attn_kind}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_kind = attn_kind
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.attn_kind == "softmax":
            attn = F.softmax(scores, dim=-1)
        else:
            attn = F.relu(scores) / n        # (1/n) ReLU(QK^T / sqrt(d_h))
        self.last_attn = attn.detach()        # [B, heads, n, n] real learned weights
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        return out


class _RobertaBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, eps = cfg.d_model, cfg.layer_norm_eps
        self.norm_style = cfg.norm_style
        self.attn = _ReluSelfAttention(d, cfg.n_heads, cfg.dropout, cfg.attn_kind)
        self.attn_dense = nn.Linear(d, d)
        self.attn_ln = nn.LayerNorm(d, eps=eps)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.intermediate = nn.Linear(d, cfg.d_ff)
        self.output = nn.Linear(cfg.d_ff, d)
        self.out_ln = nn.LayerNorm(d, eps=eps)
        self.out_drop = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.norm_style == "post":            # RoBERTa / BERT
            a = self.attn_drop(self.attn_dense(self.attn(h)))
            h = self.attn_ln(h + a)
            f = self.out_drop(self.output(self.act(self.intermediate(h))))
            h = self.out_ln(h + f)
        else:                                    # pre-norm (A.1 idealisation)
            a = self.attn_drop(self.attn_dense(self.attn(self.attn_ln(h))))
            h = h + a
            f = self.out_drop(self.output(self.act(self.intermediate(self.out_ln(h)))))
            h = h + f
        return h


class RobertaGraphTransformer(nn.Module):
    """RoBERTa-style encoder for graph connectivity (n x n logits)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.read_in = nn.Linear(config.n, config.d_model)
        self.emb_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.emb_drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([_RobertaBlock(config) for _ in range(config.n_layers)])
        self.readout_kind = getattr(config, "readout", "linear")
        if self.readout_kind == "similarity":
            # connectivity == cosine similarity of node embeddings (spectral view)
            self.sim_scale = nn.Parameter(torch.tensor(10.0))
            self.sim_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.read_out = nn.Linear(config.d_model, config.n)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.n or x.shape[-2] != self.config.n:
            raise ValueError(f"Expected input [B, {self.config.n}, {self.config.n}], got {x.shape}")
        h = self.emb_drop(self.emb_ln(self.read_in(x)))
        for block in self.blocks:
            h = block(h)
        return h

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """The node embeddings the read-out sees (H = h^(L))."""
        return self._trunk(x)

    def forward_and_embeddings(self, x: torch.Tensor):
        h = self._trunk(x)
        if self.readout_kind == "similarity":
            hn = F.normalize(h, dim=-1)
            logits = self.sim_scale * torch.matmul(hn, hn.transpose(-1, -2)) + self.sim_bias
        else:
            logits = self.read_out(h)   # [B, n, n], no final norm, no symmetrisation
        return logits, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_and_embeddings(x)[0]

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The real learned attention weights, one tensor [B, heads, n, n] per layer."""
        self.forward(x)
        return [blk.attn.last_attn for blk in self.blocks]

    @torch.no_grad()
    def predict_binary(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        return (self.forward(x) > threshold).to(torch.int64)
