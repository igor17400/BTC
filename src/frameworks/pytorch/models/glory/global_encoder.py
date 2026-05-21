"""GLORY global encoders (PyTorch).

Hosts the auxiliary encoders used on the "global" / KG enrichment path:

- :class:`GatedGraphConv` — pure-PyTorch port of
  ``torch_geometric.nn.GatedGraphConv`` (Li et al. 2016) used to
  propagate features across the precomputed news graph.
- :class:`EntityEncoder` and :class:`GlobalEntityEncoder` — entity
  branch (only built when ``use_entity=True``).

Pure PyTorch — no ``torch_geometric`` dependency.  References:
Yang et al., "Going Beyond Local…", RecSys 2023; Li et al., GGNN, ICLR 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .news_encoder import AttentionPooling, MultiHeadAttention

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

    Implementation::

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
            # Message: W_i @ h[src] (per-layer weight).
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
    """Local entity encoder.

    Pipeline: ``emb → dropout → MHA → LN → drop → pool → LN → Linear → LeakyReLU``.
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
    """Global entity encoder.

    Same pipeline as :class:`EntityEncoder` but WITHOUT the final
    ``Linear + LeakyReLU``. Output dim = ``head_num * head_dim = news_dim``.
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
