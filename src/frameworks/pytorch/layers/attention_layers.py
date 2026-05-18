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


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with explicit ``head_dim`` (PyTorch).

    Mirrors :class:`flax.nnx.MultiHeadAttention` when called with
    ``in_features=D, qkv_features=num_heads*head_dim, out_features=D``.
    Each of Q/K/V is its own ``nn.Linear`` (not packed) so weight transfer
    between frameworks stays trivial.

    Why not ``nn.MultiheadAttention``: ``nn.MultiheadAttention`` derives
    ``head_dim = embed_dim / num_heads`` from ``embed_dim`` and silently
    ignores any explicit ``head_dim``. When ``embed_dim`` is smaller than
    ``num_heads * head_dim`` (e.g. PP-Rec's entity branch where the entity
    embedding is 100-d but the paper asks for 20×20-d heads) the
    PyTorch port ends up with 5-dim heads instead of the configured 20.
    See :mod:`src.frameworks.pytorch.models.pprec` for the original bug.

    Args:
        in_features: Last-dim size of the input.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. Internal Q/K/V width is
            ``num_heads * head_dim``.
        dropout_rate: Dropout applied to attention weights.
        out_features: Last-dim size of the output. Defaults to
            ``in_features`` (matches Flax NNX behaviour).
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float = 0.0,
        out_features: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        total_dim = num_heads * head_dim
        out_features = out_features if out_features is not None else in_features

        self.W_q = nn.Linear(in_features, total_dim)
        self.W_k = nn.Linear(in_features, total_dim)
        self.W_v = nn.Linear(in_features, total_dim)
        self.W_o = nn.Linear(total_dim, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, S, in_features)`` input.
            key_padding_mask: Optional ``(B, S)`` bool mask where ``True``
                marks padded positions. Rows that are entirely padded are
                unmasked at position 0 internally to avoid an all-``-inf``
                softmax (which produces NaN); their outputs are zeroed
                after the projection.

        Returns:
            ``(B, S, out_features)`` output.
        """
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(x).reshape(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        K = self.W_k(x).reshape(B, S, H, D).transpose(1, 2)
        V = self.W_v(x).reshape(B, S, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D**0.5)  # (B, H, S, S)

        if key_padding_mask is not None:
            fully_padded = key_padding_mask.all(dim=-1)  # (B,)
            kpm = key_padding_mask
            if fully_padded.any():
                kpm = key_padding_mask.clone()
                kpm[fully_padded, 0] = False
            # True = padded → set to -inf so softmax zeroes it
            mask = kpm[:, None, None, :]  # (B, 1, 1, S)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, S, D)
        out = out.transpose(1, 2).reshape(B, S, H * D)  # (B, S, H*D)
        out = self.W_o(out)

        if key_padding_mask is not None and fully_padded.any():
            # Outputs along the query axis are also meaningless for
            # fully-padded rows — zero them so downstream layers see 0.
            mask_q = (~fully_padded).to(out.dtype).view(B, 1, 1)
            out = out * mask_q

        return out


class CrossAttention(nn.Module):
    """Multi-head cross-attention that supports different Q and KV dimensions.

    PyTorch mirror of :class:`src.frameworks.jax.layers.attention_layers.CrossAttention`.
    Each of Q/K/V is its own ``nn.Linear`` so a query projection of
    ``(q_dim → num_heads*head_dim)`` is folded into ``W_q`` — avoiding the
    redundant pre-projection that ``nn.MultiheadAttention(embed_dim,
    kdim, vdim)`` requires when q_dim differs from the output dim.

    Args:
        q_dim: Feature dim of query inputs.
        kv_dim: Feature dim of key/value inputs.
        num_heads: Number of attention heads.
        head_dim: Dim per head. Total output = ``num_heads * head_dim``.
        dropout_rate: Dropout on attention weights.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        total_dim = num_heads * head_dim

        self.W_q = nn.Linear(q_dim, total_dim)
        self.W_k = nn.Linear(kv_dim, total_dim)
        self.W_v = nn.Linear(kv_dim, total_dim)
        self.W_o = nn.Linear(total_dim, total_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: ``(B, Sq, q_dim)`` query tensor.
            key_value: ``(B, Skv, kv_dim)`` key/value tensor.
            key_padding_mask: Optional ``(B, Skv)`` bool mask where ``True``
                marks padded keys. Rows that are entirely padded are
                unmasked at position 0 internally; the corresponding
                outputs along the query axis are zeroed afterwards.

        Returns:
            ``(B, Sq, num_heads * head_dim)`` output.
        """
        B = query.shape[0]
        Sq = query.shape[1]
        Skv = key_value.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(query).reshape(B, Sq, H, D).transpose(1, 2)  # (B, H, Sq, D)
        K = self.W_k(key_value).reshape(B, Skv, H, D).transpose(1, 2)
        V = self.W_v(key_value).reshape(B, Skv, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D**0.5)  # (B, H, Sq, Skv)

        if key_padding_mask is not None:
            fully_padded = key_padding_mask.all(dim=-1)  # (B,)
            kpm = key_padding_mask
            if fully_padded.any():
                kpm = key_padding_mask.clone()
                kpm[fully_padded, 0] = False
            mask = kpm[:, None, None, :]  # (B, 1, 1, Skv)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, Sq, D)
        out = out.transpose(1, 2).reshape(B, Sq, H * D)
        out = self.W_o(out)

        if key_padding_mask is not None and fully_padded.any():
            mask_q = (~fully_padded).to(out.dtype).view(B, 1, 1)
            out = out * mask_q

        return out


class ComputeMasking(nn.Module):
    """Produce a boolean mask where ``inputs != 0``."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs != 0).float()


class OverwriteMasking(nn.Module):
    """Zero out positions according to a mask."""

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values * mask.unsqueeze(-1)
