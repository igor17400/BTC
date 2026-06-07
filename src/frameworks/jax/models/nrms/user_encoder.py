"""NRMS user encoder (Flax NNX) — encoder-agnostic.

Encodes user browsing history into a single user vector via MHA +
additive pool over the history slots. Delegates per-slot encoding and
slot-validity masking to the news encoder so this module is unaware of
GloVe vs PLM.
"""

from __future__ import annotations

import jax
from flax import nnx

from src.core.models.configs import NRMSConfig

from ...layers import AdditiveAttention
from .news_encoder import NewsEncoder


class UserEncoder(nnx.Module):
    """NRMS user encoder.

    Pipeline:
        news_encoder(*history) -> MHA -> AdditiveAttention -> user vector
    """

    def __init__(
        self,
        config: NRMSConfig,
        news_encoder: NewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder

        self.browsed_news_attention = nnx.MultiHeadAttention(
            num_heads=config.user_num_heads,
            in_features=config.embedding_size,
            qkv_features=config.embedding_size,
            decode=False,
            rngs=rngs,
        )
        self.user_additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self, history_news_idx: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Encode user browsing history.

        Args:
            history_news_idx: ``(B, H)`` int array of parsed news ids.
            training: Enable dropout.

        Returns:
            ``(B, embedding_size)`` user representations.
        """
        history_repr = self.news_encoder(
            history_news_idx, training=training
        )  # (B, H, D)
        valid = self.news_encoder.valid_mask(history_news_idx)  # (B, H)
        attn_mask = valid[:, None, None, :]

        y = self.browsed_news_attention(
            history_repr,
            history_repr,
            mask=attn_mask,
            deterministic=not training,
        )

        return self.user_additive_attention(y, mask=valid)
