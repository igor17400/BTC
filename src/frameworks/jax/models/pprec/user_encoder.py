"""PP-Rec user encoder (Flax NNX) — Content-Popularity Joint Attention.

Mirror of the PyTorch user encoder.

    news_encoder(history) -> (B, H, news_dim)
        -> MHSA over H slots
    popularity_embedding(history_ctr) -> (B, H, pop_emb_dim)
    Concat[user_vecs, pop_emb] -> AttentivePoolingQKY
        -> user vector (B, news_dim)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import PPRecConfig

from ...layers import AttentivePoolingQKY
from .news_encoder import NewsEncoder


class UserEncoder(nnx.Module):
    """Popularity-aware user encoder with CPJA."""

    def __init__(
        self,
        config: PPRecConfig,
        news_encoder: NewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder

        self.user_mhsa = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.news_dim,
            qkv_features=config.num_heads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        self.popularity_embedding = nnx.Embed(
            num_embeddings=config.popularity_embedding_bins,
            features=config.popularity_embedding_dim,
            rngs=rngs,
        )
        cpja_key_dim = config.news_dim + config.popularity_embedding_dim
        self.cpja = AttentivePoolingQKY(
            key_dim=cpja_key_dim,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        history_features: jax.Array,
        history_ctr: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode user history into a user vector.

        Args:
            history_features: ``(B, H)`` plain news_idx, or ``(B, H, k)``
                packed ``[news_idx | entities | category]``.
            history_ctr: ``(B, H)`` int discretised CTR (optional).
        """
        user_vecs = self.news_encoder(
            history_features, training=training
        )  # (B, H, news_dim)
        history_keep = self.news_encoder.valid_mask(history_features)  # (B, H)
        attn_mask = history_keep[:, None, None, :]
        user_vecs = self.user_mhsa(
            user_vecs, user_vecs, mask=attn_mask, deterministic=not training
        )

        B, H = history_keep.shape
        if history_ctr is None:
            history_ctr = jnp.zeros((B, H), dtype=jnp.int32)
        pop_emb = self.popularity_embedding(history_ctr.astype(jnp.int32))

        key_input = jnp.concatenate([user_vecs, pop_emb], axis=-1)
        return self.cpja(key_input, user_vecs, mask=history_keep)
