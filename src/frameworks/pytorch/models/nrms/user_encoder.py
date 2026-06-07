"""NRMS user encoder (PyTorch) — encoder-agnostic.

Encodes user browsing history into a single user vector via MHSA +
additive pool over the history slots. Delegates per-slot encoding and
slot-validity masking to the news encoder so this module is unaware of
whether GloVe or PLM is downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import NRMSConfig

from ...layers import AdditiveAttention
from .news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    """NRMS user encoder.

    Pipeline:
        news_encoder(*history) -> MHSA -> AdditiveAttention -> user vector
    """

    def __init__(self, config: NRMSConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.browsed_news_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.user_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.user_additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
        )

    def forward(
        self, history_news_idx: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode user browsing history.

        Args:
            history_news_idx: ``(B, H)`` int tensor of parsed news ids.
            training: Controls dropout.

        Returns:
            ``(B, embedding_size)`` user representations.
        """
        history_repr = self.news_encoder(
            history_news_idx, training=training
        )  # (B, H, D)
        valid = self.news_encoder.valid_mask(history_news_idx)  # (B, H)

        key_padding_mask = ~valid  # True = padded
        fully_empty = ~valid.any(dim=-1)  # (B,) users with no history
        if fully_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_empty, 0] = False

        y, _ = self.browsed_news_attention(
            history_repr,
            history_repr,
            history_repr,
            key_padding_mask=key_padding_mask,
        )

        user_repr = self.user_additive_attention(y, mask=valid)

        if fully_empty.any():
            user_repr = user_repr.masked_fill(fully_empty.unsqueeze(-1), 0.0)

        return user_repr
