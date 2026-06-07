"""GLORY user encoder + per-clicked-news view fusion (Flax NNX).

Mirror of the PyTorch ``user_encoder`` module. ``ClickEncoder`` fuses 2
(title, graph) or 3 (title, graph, entity) views per clicked news into a
single per-slot vector; ``UserEncoder`` then pools those vectors into a
user representation via MHA + attention pooling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import GLORYConfig

from .news_encoder import AttentionPooling, MultiHeadAttention


class ClickEncoder(nnx.Module):
    """Fuse per-clicked-news views via attention pooling.

    Stacks 2 views (title, graph) or 3 views (title, graph, entity)
    depending on whether entity_emb is provided.
    """

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.attn_pool = AttentionPooling(
            self.news_dim,
            config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        title_emb: jax.Array,  # (B, N, D)
        graph_emb: jax.Array,  # (B, N, D)
        entity_emb: jax.Array | None = None,  # (B, N, D)
    ) -> jax.Array:
        B, N = title_emb.shape[:2]
        if entity_emb is not None:
            stacked = jnp.stack(
                [title_emb, graph_emb, entity_emb],
                axis=-2,
            )  # (B, N, 3, D)
            num_views = 3
        else:
            stacked = jnp.stack(
                [title_emb, graph_emb],
                axis=-2,
            )  # (B, N, 2, D)
            num_views = 2
        stacked = stacked.reshape(B * N, num_views, self.news_dim)
        fused = self.attn_pool(stacked)  # (B*N, D)
        return fused.reshape(B, N, self.news_dim)


class UserEncoder(nnx.Module):
    """Pool a sequence of clicked-news embeddings into a user vector."""

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.msa = MultiHeadAttention(
            self.news_dim,
            self.news_dim,
            self.news_dim,
            config.head_num,
            config.head_dim,
            rngs=rngs,
        )
        self.attn_pool = AttentionPooling(
            self.news_dim,
            config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        clicked_news: jax.Array,  # (B, H, D)
        mask: jax.Array | None = None,
    ) -> jax.Array:
        h = self.msa(clicked_news, clicked_news, clicked_news, mask)
        return self.attn_pool(h, mask)
