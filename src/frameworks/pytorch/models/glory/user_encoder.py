"""GLORY user encoder + per-clicked-news view fusion (PyTorch).

``ClickEncoder`` fuses 2 (title, graph) or 3 (title, graph, entity)
views per clicked news into a single per-slot vector; ``UserEncoder``
then pools those vectors into a user representation via MHA + attention
pooling over the history.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import GLORYConfig

from .news_encoder import AttentionPooling, MultiHeadAttention


class ClickEncoder(nn.Module):
    """Fuse per-clicked-news views via attention pooling.

    Stacks 2 views (title, graph) or 3 views (title, graph, entity)
    depending on whether ``entity_emb`` is provided.
    """

    def __init__(self, config: GLORYConfig):
        super().__init__()
        self.news_dim = config.head_num * config.head_dim
        self.attn_pool = AttentionPooling(self.news_dim, config.attention_hidden_dim)

    def forward(
        self,
        title_emb: torch.Tensor,  # (B, N, D)
        graph_emb: torch.Tensor,  # (B, N, D)
        entity_emb: torch.Tensor | None = None,  # (B, N, D)
    ) -> torch.Tensor:
        B, N = title_emb.shape[:2]
        if entity_emb is not None:
            stacked = torch.stack(
                [title_emb, graph_emb, entity_emb],
                dim=-2,
            )  # (B, N, 3, D)
            num_views = 3
        else:
            stacked = torch.stack([title_emb, graph_emb], dim=-2)  # (B, N, 2, D)
            num_views = 2
        stacked = stacked.view(B * N, num_views, self.news_dim)
        fused = self.attn_pool(stacked)  # (B*N, D)
        return fused.view(B, N, self.news_dim)


class UserEncoder(nn.Module):
    """Pool a sequence of clicked-news embeddings into a user vector."""

    def __init__(self, config: GLORYConfig):
        super().__init__()
        self.news_dim = config.head_num * config.head_dim
        self.msa = MultiHeadAttention(
            self.news_dim,
            self.news_dim,
            self.news_dim,
            config.head_num,
            config.head_dim,
        )
        self.attn_pool = AttentionPooling(self.news_dim, config.attention_hidden_dim)

    def forward(
        self,
        clicked_news: torch.Tensor,  # (B, H, D)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.msa(clicked_news, clicked_news, clicked_news, mask)
        return self.attn_pool(h, mask)
