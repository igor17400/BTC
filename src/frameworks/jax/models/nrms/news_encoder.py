"""NRMS news encoder (Flax NNX) — encoder-agnostic, IP2-style.

Mirror of the PyTorch news encoder. MHSA runs at the text encoder's
native dim (300 for GloVe, 768 for BERT-base); the additive-pool
output is projected to ``embedding_size`` (the model's news_dim)
when the dims differ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import NRMSConfig

from ...layers import AdditiveAttention, TextEncoder


class _Identity(nnx.Module):
    """Pass-through used when no post-pool projection is needed."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class NewsEncoder(nnx.Module):
    """NRMS per-news encoder (JAX). Operates at ``text_encoder.output_dim``.

    Args:
        config: NRMS hyperparameters. ``news_num_heads`` MUST divide
            ``text_encoder.output_dim``.
        text_encoder: Configured :class:`TextEncoder`.
        rngs: ``nnx.Rngs``.
    """

    def __init__(
        self,
        config: NRMSConfig,
        text_encoder: TextEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        text_dim = text_encoder.output_dim
        if text_dim % config.news_num_heads != 0:
            raise ValueError(
                f"news_num_heads={config.news_num_heads} does not divide "
                f"text_encoder.output_dim={text_dim}. Override "
                "spec.model.architecture.news_encoder.num_heads in the "
                f"experiment yaml to a divisor of {text_dim} "
                "(e.g. 12 for BERT-base 768d -> head_dim 64)."
            )
        self.config = config
        self.text_dim = int(text_dim)
        self.news_dim = config.embedding_size
        self.text_encoder = text_encoder

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.multi_head_attention = nnx.MultiHeadAttention(
            num_heads=config.news_num_heads,
            in_features=text_dim,
            qkv_features=text_dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=text_dim,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )
        if text_dim != self.news_dim:
            self.projection: nnx.Module = nnx.Linear(text_dim, self.news_dim, rngs=rngs)
        else:
            self.projection = _Identity()

    @staticmethod
    def valid_mask(news_idx: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed id is non-zero."""
        return jnp.not_equal(news_idx, 0)

    def __call__(self, news_idx: jax.Array, *, training: bool = False) -> jax.Array:
        """Encode news titles."""
        leading_shape = news_idx.shape
        flat_idx = news_idx.reshape(-1)

        tokens, mask = self.text_encoder(flat_idx)  # (N*, T, text_dim), (N*, T)

        y = self.dropout1(tokens, deterministic=not training)

        valid = jnp.not_equal(mask, 0)
        attn_mask = valid[:, None, None, :]

        y = self.multi_head_attention(y, y, mask=attn_mask, deterministic=not training)
        y = self.dropout2(y, deterministic=not training)

        pooled = self.additive_attention(y, mask=valid)  # (N*, text_dim)
        news_repr = self.projection(pooled)  # (N*, news_dim)
        return news_repr.reshape(*leading_shape, self.news_dim)
