"""NAML user encoder (PyTorch) — additive attention over history.

Pipeline:
    news_encoder(history) -> AdditiveAttention with mask -> user vector
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import NAMLConfig

from ...layers import AdditiveAttention
from .news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    """Encode user browsing history into a single user vector."""

    def __init__(self, config: NAMLConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.user_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.user_attention_query_dim,
        )

    def forward(
        self, history_packed: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode a packed history tensor.

        Args:
            history_packed: ``(B, H, 3)`` int with columns
                ``[news_idx | category_id | subcategory_id]``.
            training: Controls dropout.

        Returns:
            ``(B, cnn_filter_num)`` user vectors.
        """
        history_repr = self.news_encoder(history_packed, training=training)
        history_mask = self.news_encoder.valid_mask(history_packed)  # (B, H)
        return self.user_attention(history_repr, mask=history_mask)
