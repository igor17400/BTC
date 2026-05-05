"""TCCM-faithful JAX/Flax NNX layers.

Port of ``src/frameworks/pytorch/models/tccm/layers.py``.

* :class:`TCCMPopularityEncoder` — per-token bucketed CTR with
  cross-attention, content scorer MLP, and reciprocal-power time module.
* :class:`TCCMActivityGater` — Dense(128, tanh) → Dense(64) →
  Dense(1, sigmoid).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import TCCMConfig

from ...layers import AdditiveAttention

# ---------------------------------------------------------------------------
# Popularity encoder (paper ``get_popularity_encoder``)
# ---------------------------------------------------------------------------


class TCCMPopularityEncoder(nnx.Module):
    """Content-aware popularity encoder for TCCM (JAX port)."""

    def __init__(self, config: TCCMConfig, *, rngs: nnx.Rngs):
        self.config = config

        self.token_pop_embedding = nnx.Embed(
            num_embeddings=config.pop_token_embedding_bins,
            features=config.pop_token_embedding_dim,
            rngs=rngs,
        )
        self.title_dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.entity_dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        proj_dim = config.pop_num_heads * config.pop_head_dim
        self.title_proj = nnx.Linear(
            config.pop_token_embedding_dim, proj_dim, rngs=rngs
        )
        self.entity_proj = nnx.Linear(
            config.pop_token_embedding_dim, proj_dim, rngs=rngs
        )

        def _mha():
            return nnx.MultiHeadAttention(
                num_heads=config.pop_num_heads,
                in_features=proj_dim,
                qkv_features=proj_dim,
                decode=False,
                rngs=rngs,
            )

        self.title_co_mhca = _mha()
        self.entity_co_mhca = _mha()
        self.title_mhsa = _mha()
        self.entity_mhsa = _mha()

        self.title_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.entity_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.fusion_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        self.title_attention = AdditiveAttention(
            input_dim=proj_dim,
            query_vec_dim=proj_dim,
            rngs=rngs,
        )
        self.entity_attention = AdditiveAttention(
            input_dim=proj_dim,
            query_vec_dim=proj_dim,
            rngs=rngs,
        )
        self.fusion_attention = AdditiveAttention(
            input_dim=proj_dim,
            query_vec_dim=proj_dim,
            rngs=rngs,
        )

        c1, c2, c3 = config.pop_content_dims
        self.content_d1 = nnx.Linear(proj_dim, c1, rngs=rngs)
        self.content_d2 = nnx.Linear(c1, c2, rngs=rngs)
        self.content_d3 = nnx.Linear(c2, c3, rngs=rngs)
        self.content_out = nnx.Linear(c3, 1, rngs=rngs)

        self.time_embedding = nnx.Embed(
            num_embeddings=config.timeliness_embedding_bins,
            features=config.timeliness_embedding_dim,
            rngs=rngs,
        )
        r1, r2 = config.pop_recency_dims
        self.time_d1 = nnx.Linear(config.timeliness_embedding_dim, r1, rngs=rngs)
        self.time_d2 = nnx.Linear(r1, r2, rngs=rngs)
        self.time_out = nnx.Linear(r2, 1, rngs=rngs)
        self.timeliness_lambda = float(config.timeliness_lambda)

    def __call__(
        self,
        bucket_input: jax.Array,
        time_input: jax.Array,
        title_len: int,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            bucket_input: ``(B, T+E)`` int.
            time_input: ``(B,)`` int.
            title_len: ``T`` so the layer can split the entity slice.

        Returns:
            ``(B,)`` popularity score.
        """
        det = not training
        title_buckets = bucket_input[:, :title_len]
        entity_buckets = bucket_input[:, title_len:]

        title_emb = self.title_dropout1(
            self.token_pop_embedding(title_buckets), deterministic=det
        )
        entity_emb = self.entity_dropout1(
            self.token_pop_embedding(entity_buckets), deterministic=det
        )
        title_h = self.title_proj(title_emb)
        entity_h = self.entity_proj(entity_emb)

        title_co = self.title_co_mhca(title_h, entity_h, deterministic=det)
        entity_co = self.entity_co_mhca(entity_h, title_h, deterministic=det)

        title_self = self.title_mhsa(title_h, title_h, deterministic=det)
        title_seq = title_self + title_co
        title_seq = self.title_dropout2(title_seq, deterministic=det)
        title_vec = self.title_attention(title_seq)

        entity_self = self.entity_mhsa(entity_h, entity_h, deterministic=det)
        entity_seq = entity_self + entity_co
        entity_seq = self.entity_dropout2(entity_seq, deterministic=det)
        entity_vec = self.entity_attention(entity_seq)

        stacked = jnp.stack([title_vec, entity_vec], axis=1)
        stacked = self.fusion_dropout(stacked, deterministic=det)
        pop_vec = self.fusion_attention(stacked)

        x = jnp.tanh(self.content_d1(pop_vec))
        x = self.content_d2(x)
        x = self.content_d3(x)
        s_p = jax.nn.sigmoid(self.content_out(x)).squeeze(-1)

        time_emb = self.time_embedding(time_input)
        r = jnp.tanh(self.time_d1(time_emb))
        r = self.time_d2(r)
        t_prime = jax.nn.sigmoid(self.time_out(r)).squeeze(-1)
        t_factor = jnp.clip(t_prime, a_min=1e-6) ** (-self.timeliness_lambda)
        return s_p * t_factor


# ---------------------------------------------------------------------------
# Activity gater
# ---------------------------------------------------------------------------


class TCCMActivityGater(nnx.Module):
    """Per-user gate balancing relevance vs. popularity."""

    def __init__(self, config: TCCMConfig, *, rngs: nnx.Rngs):
        d1, d2 = config.activity_gate_dims
        self.dense1 = nnx.Linear(config.news_dim, d1, rngs=rngs)
        self.dense2 = nnx.Linear(d1, d2, rngs=rngs)
        self.dense_out = nnx.Linear(d2, 1, rngs=rngs)

    def __call__(self, user_vec: jax.Array) -> jax.Array:
        x = jnp.tanh(self.dense1(user_vec))
        x = self.dense2(x)
        return jax.nn.sigmoid(self.dense_out(x)).squeeze(-1)
