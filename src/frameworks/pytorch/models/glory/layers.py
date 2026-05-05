"""GLORY-specific layers: attention primitives + GatedGraphConv.

Pure PyTorch — no ``torch_geometric`` dependency.  This keeps the
model self-contained and makes the future JAX / Keras ports direct
line-by-line translations.

References:
- Yang et al., "Going Beyond Local: Global Graph-Enhanced Personalized
  News Recommendations", RecSys 2023.
- Li et al., "Gated Graph Sequence Neural Networks" (GGNN), ICLR 2016.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ======================================================================
# Attention primitives (mirror GLORY's ``src/models/base/layers.py``)
# ======================================================================


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention used inside ``MultiHeadAttention``.

    Note: GLORY's reference uses a non-standard masking scheme — scores
    are exponentiated before masking, then re-normalised.  We replicate
    that behaviour exactly to match the paper's initialization and
    training dynamics.
    """

    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask.unsqueeze(dim=-2)
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(attn, V)


class MultiHeadAttention(nn.Module):
    """Multi-head attention matching GLORY's Q/K/V projection choice."""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        head_num: int,
        head_dim: int,
        residual: bool = False,
    ):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.residual = residual
        out_dim = head_num * head_dim

        self.W_Q = nn.Linear(key_size, out_dim, bias=True)
        self.W_K = nn.Linear(query_size, out_dim, bias=False)
        self.W_V = nn.Linear(value_size, out_dim, bias=True)
        self.attn = ScaledDotProductAttention(head_dim)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_V.bias)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.head_num, -1)
        q = (
            self.W_Q(Q)
            .view(batch_size, -1, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.W_K(K)
            .view(batch_size, -1, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.W_V(V)
            .view(batch_size, -1, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        ctx = self.attn(q, k, v, mask)
        out = (
            ctx.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head_num * self.head_dim)
        )
        return out + Q if self.residual else out


class AttentionPooling(nn.Module):
    """Additive attention pooling over a sequence of vectors.

    Follows GLORY's implementation exactly:
    ``alpha = softmax(v^T tanh(W x)) / (sum + eps)`` with multiplicative
    masking (consistent with ``ScaledDotProductAttention``).
    """

    def __init__(self, emb_size: int, hidden_size: int):
        super().__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(
            self.att_fc1.weight,
            gain=nn.init.calculate_gain("tanh"),
        )
        nn.init.zeros_(self.att_fc1.bias)
        nn.init.xavier_uniform_(self.att_fc2.weight)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, N, E)
        e = torch.tanh(self.att_fc1(x))
        alpha = torch.exp(self.att_fc2(e))  # (B, N, 1)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        return torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)


# ======================================================================
# DotProduct scorer
# ======================================================================


class DotProduct(nn.Module):
    """Batched dot product for candidate scoring."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # left: (B, C, D), right: (B, D)  →  (B, C)
        return torch.bmm(left, right.unsqueeze(-1)).squeeze(-1)


# ======================================================================
# GatedGraphConv (pure PyTorch, no torch_geometric)
# ======================================================================


def _scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Segment sum along dim 0: ``out[index[i]] += src[i]``.

    Args:
        src: ``(E, D)`` source values (edge features).
        index: ``(E,)`` int target indices in ``[0, dim_size)``.
        dim_size: Output size along dim 0.

    Returns:
        ``(dim_size, D)`` aggregated output.
    """
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(0, idx, src)


class GatedGraphConv(nn.Module):
    """Gated Graph Neural Network (Li et al. 2016).

    Pure-PyTorch replacement for ``torch_geometric.nn.GatedGraphConv``.
    Shared weight ``W`` across all ``num_layers`` propagation steps,
    with a single ``GRUCell`` mixing messages into node states (matches
    GLORY's original configuration which uses the PyG default).

    Implementation:
        for t in range(num_layers):
            m_i = Σ_{j ∈ N(i)} W · h_j        # weighted neighbor sum
            h_i = GRU(m_i, h_i)               # gated update

    Args:
        out_channels: Output feature dimension.  ``in_channels`` may be
            smaller; inputs are zero-padded to ``out_channels``.
        num_layers: Number of propagation steps.
        aggr: Aggregation function; only ``"add"`` supported (matches
            GLORY's config).
    """

    def __init__(
        self,
        out_channels: int,
        num_layers: int = 3,
        aggr: str = "add",
        bias: bool = True,
    ):
        super().__init__()
        if aggr != "add":
            raise ValueError(
                f"Only aggr='add' supported; got {aggr!r}. "
                f"GLORY's reference uses add aggregation."
            )
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.weight = nn.Parameter(torch.empty(num_layers, out_channels, out_channels))
        self.rnn = nn.GRUCell(out_channels, out_channels, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for i in range(self.num_layers):
            nn.init.xavier_uniform_(self.weight[i])
        # GRUCell is self-initialised by PyTorch default.

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Propagate features across a graph.

        Args:
            x: ``(N, F_in)`` node features.  Will be zero-padded to
                ``(N, out_channels)`` if ``F_in < out_channels``.
            edge_index: ``(2, E)`` int tensor of directed edges
                ``(src, dst)`` — messages flow src → dst.

        Returns:
            ``(N, out_channels)`` updated node features.
        """
        if x.shape[1] > self.out_channels:
            raise ValueError(
                f"GatedGraphConv: input feature dim {x.shape[1]} exceeds "
                f"out_channels={self.out_channels}."
            )
        if x.shape[1] < self.out_channels:
            zero = x.new_zeros(x.shape[0], self.out_channels - x.shape[1])
            x = torch.cat([x, zero], dim=-1)

        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        for i in range(self.num_layers):
            # Message: W_i @ h[src]  (per-layer weight)
            m = x @ self.weight[i]  # (N, D)
            # Aggregate at destination nodes.
            agg = _scatter_add(m[src], dst, num_nodes)  # (N, D)
            # Gated update.
            x = self.rnn(agg, x)  # (N, D)

        return x


# ======================================================================
# Entity encoders (used when ``use_entity=True``)
# ======================================================================


class EntityEncoder(nn.Module):
    """Local entity encoder: emb -> dropout -> MHA -> LN -> drop -> pool -> LN -> Linear -> LeakyReLU.

    Mirrors :class:`src.frameworks.jax.models.glory.layers.EntityEncoder`.
    Projects ``entity_dim`` to ``news_dim``.
    """

    def __init__(
        self,
        entity_dim: int,
        news_dim: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        head_num = entity_dim // head_dim
        self.dropout1 = nn.Dropout(dropout_rate)
        self.msa = MultiHeadAttention(
            entity_dim,
            entity_dim,
            entity_dim,
            head_num,
            head_dim,
        )
        self.layernorm1 = nn.LayerNorm(entity_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.attn_pool = AttentionPooling(entity_dim, attention_hidden_dim)
        self.layernorm2 = nn.LayerNorm(entity_dim)
        self.linear = nn.Linear(entity_dim, news_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        entity_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode entities.

        Args:
            entity_input: ``(B, N, E_ent, entity_dim)`` entity embeddings.
            mask: optional ``(B*N, E_ent)`` mask.

        Returns:
            ``(B, N, news_dim)`` entity representations.
        """
        B, N = entity_input.shape[:2]
        x = entity_input.reshape(B * N, entity_input.shape[2], entity_input.shape[3])
        x = self.dropout1(x)
        x = self.msa(x, x, x, mask)
        x = self.layernorm1(x)
        x = self.dropout2(x)
        x = self.attn_pool(x, mask)
        x = self.layernorm2(x)
        x = self.act(self.linear(x))
        return x.view(B, N, -1)


class GlobalEntityEncoder(nn.Module):
    """Global entity encoder: emb -> dropout -> MHA -> LN -> drop -> pool -> LN.

    Same as :class:`EntityEncoder` but WITHOUT the final ``Linear +
    LeakyReLU``. Output dim = ``head_num * head_dim = news_dim``.
    """

    def __init__(
        self,
        entity_dim: int,
        head_num: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.news_dim = head_num * head_dim
        self.dropout1 = nn.Dropout(dropout_rate)
        self.msa = MultiHeadAttention(
            entity_dim,
            entity_dim,
            entity_dim,
            head_num,
            head_dim,
        )
        self.layernorm1 = nn.LayerNorm(self.news_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.attn_pool = AttentionPooling(self.news_dim, attention_hidden_dim)
        self.layernorm2 = nn.LayerNorm(self.news_dim)

    def forward(
        self,
        entity_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode neighbor entities.

        Args:
            entity_input: ``(B, N, E_ent, entity_dim)`` entity embeddings.
            mask: optional ``(B*N, E_ent)`` mask.

        Returns:
            ``(B, N, news_dim)`` entity representations.
        """
        B, N = entity_input.shape[:2]
        x = entity_input.reshape(B * N, entity_input.shape[2], entity_input.shape[3])
        x = self.dropout1(x)
        x = self.msa(x, x, x, mask)
        x = self.layernorm1(x)
        x = self.dropout2(x)
        x = self.attn_pool(x, mask)
        x = self.layernorm2(x)
        return x.view(B, N, self.news_dim)
