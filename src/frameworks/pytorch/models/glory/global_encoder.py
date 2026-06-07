"""GLORY global encoders (PyTorch).

Hosts the auxiliary encoders used on the "global" / KG enrichment path:

- ``GatedGraphConv`` — re-exported from ``torch_geometric.nn`` to
  match the reference implementation at
  ``reference_codes/glory_official/src/models/GLORY.py:30``
  (``GatedGraphConv(news_dim, num_layers=3, aggr='add')``).
- :class:`EntityEncoder` and :class:`GlobalEntityEncoder` — entity
  branch (only built when ``use_entity=True``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv

from .news_encoder import AttentionPooling, MultiHeadAttention

__all__ = ["GatedGraphConv", "EntityEncoder", "GlobalEntityEncoder"]


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
