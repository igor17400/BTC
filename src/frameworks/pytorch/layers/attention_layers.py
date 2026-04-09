"""Attention and masking layers for PyTorch news recommendation models.

Core layers used by NRMS, NAML, LSTUR, and PP-Rec:
- :class:`AdditiveAttention` — soft-alignment attention (all models).
- :class:`AttentivePoolingQKY` — key/value-split attention (PP-Rec CPJA).
- :class:`ComputeMasking` / :class:`OverwriteMasking` — nn.Module masking.
- :func:`compute_mask` / :func:`overwrite_mask` — functional masking utils.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pure-function masking utilities (isomorphic with Keras/JAX)
# ---------------------------------------------------------------------------


def compute_mask(inputs: torch.Tensor) -> torch.Tensor:
    """Compute a boolean mask where non-zero positions are True."""
    return (inputs != 0).float()


def overwrite_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out positions indicated by a mask."""
    return values * mask.unsqueeze(-1)


class AdditiveAttention(nn.Module):
    """Soft-alignment attention layer (additive / Bahdanau-style).

    Args:
        input_dim: Last dimension of the input tensor (feature size).
        query_vec_dim: Hidden dimension of the attention mechanism.
    """

    def __init__(self, input_dim: int, query_vec_dim: int = 200):
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, query_vec_dim))
        self.b = nn.Parameter(torch.zeros(query_vec_dim))
        self.q = nn.Parameter(torch.empty(query_vec_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention_hidden = torch.tanh(torch.matmul(inputs, self.W) + self.b)
        attention_scores = torch.matmul(attention_hidden, self.q).squeeze(-1)
        attention = torch.exp(attention_scores)
        if mask is not None:
            attention = attention * mask.float()
        attention_weights = attention / (attention.sum(dim=-1, keepdim=True) + 1e-7)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        weighted_input = inputs * attention_weights_expanded
        return weighted_input.sum(dim=1)


class AttentivePoolingQKY(nn.Module):
    """Attentive pooling where attention keys differ from values (PyTorch).

    Used in PP-Rec's Content-Popularity Joint Attention (CPJA).

    Args:
        key_dim: Last dimension of ``key_input``.
        query_vec_dim: Hidden dimension of the attention MLP.
    """

    def __init__(self, key_dim: int, query_vec_dim: int = 200):
        super().__init__()
        self.W = nn.Parameter(torch.empty(key_dim, query_vec_dim))
        self.b = nn.Parameter(torch.zeros(query_vec_dim))
        self.q = nn.Parameter(torch.empty(query_vec_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(
        self,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_hidden = torch.tanh(torch.matmul(key_input, self.W) + self.b)
        attention_scores = torch.matmul(attention_hidden, self.q).squeeze(-1)
        attention = torch.exp(attention_scores)
        if mask is not None:
            attention = attention * mask.float()
        attention_weights = attention / (attention.sum(dim=-1, keepdim=True) + 1e-7)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        weighted_input = value_input * attention_weights_expanded
        return weighted_input.sum(dim=1)


class ComputeMasking(nn.Module):
    """Produce a boolean mask where ``inputs != 0``."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs != 0).float()


class OverwriteMasking(nn.Module):
    """Zero out positions according to a mask."""

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values * mask.unsqueeze(-1)
