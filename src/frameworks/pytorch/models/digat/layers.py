"""DIGAT-specific layers: scatter utilities and attention modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ======================================================================
# Scatter utilities (replace torch_scatter)
# ======================================================================


def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Per-group softmax without torch_scatter.

    Args:
        src: (B, N) scores.
        index: (B, N) int group assignments in [0, num_groups).
        num_groups: Total number of groups.

    Returns:
        (B, N) softmax-normalised within each group.
    """
    batch_size, _ = src.shape
    idx = index.long()

    group_max = torch.full(
        (batch_size, num_groups), float("-inf"), device=src.device, dtype=src.dtype
    )
    group_max.scatter_reduce_(1, idx, src, reduce="amax", include_self=True)
    element_max = group_max.gather(1, idx)  # (batch_size, num_items)

    exp_src = torch.exp(src - element_max)

    group_sum = torch.zeros(batch_size, num_groups, device=src.device, dtype=src.dtype)
    group_sum.scatter_add_(1, idx, exp_src)
    element_sum = group_sum.gather(1, idx)  # (batch_size, num_items)

    return exp_src / (element_sum + 1e-10)


def scatter_sum(
    src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int
) -> torch.Tensor:
    """Grouped sum without torch_scatter.

    Args:
        src: (B, N, D) values.
        index: (B, N) int group assignments in [0, dim_size).
        dim: Dimension to scatter along (must be 1).
        dim_size: Output size along the scatter dimension.

    Returns:
        (B, dim_size, D) summed values per group.
    """
    batch_size, _, feat_dim = src.shape
    out = torch.zeros(
        batch_size, dim_size, feat_dim, device=src.device, dtype=src.dtype
    )
    idx = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(dim, idx, src)


# ======================================================================
# Attention layers
# ======================================================================


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: K/Q projections → softmax → pool."""

    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super().__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.scale = math.sqrt(float(attention_dim))

    def forward(
        self,
        features: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = (
            torch.bmm(self.K(features), self.Q(query).unsqueeze(2)).squeeze(2)
            / self.scale
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention for sequence pooling."""

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.affine = nn.Linear(feature_dim, attention_dim, bias=True)
        self.project = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.project(torch.tanh(self.affine(features))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (news encoder MSA)."""

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(float(head_dim))
        self.W_Q = nn.Linear(d_model, self.out_dim, bias=True)
        self.W_K = nn.Linear(d_model, self.out_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out = (
            torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.out_dim)
        )
        return out
