"""NAML user encoder (Flax NNX) — additive attention over history."""

from __future__ import annotations

import jax
from flax import nnx

from src.core.models.configs import NAMLConfig

from ...layers import AdditiveAttention
from .news_encoder import NewsEncoder


class UserEncoder(nnx.Module):
    """Encode user browsing history into a single user vector."""

    def __init__(
        self,
        config: NAMLConfig,
        news_encoder: NewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder
        self.user_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.user_attention_query_dim,
            rngs=rngs,
        )

    def __call__(
        self, history_packed: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """history_packed: ``(B, H, 3)`` int [news_idx, category, subcategory]."""
        history_repr = self.news_encoder(history_packed, training=training)
        history_mask = self.news_encoder.valid_mask(history_packed)
        return self.user_attention(history_repr, mask=history_mask)
