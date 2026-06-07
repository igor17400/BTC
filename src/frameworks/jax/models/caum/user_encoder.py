"""CAUM user encoder (Flax NNX) — fallback mean-pool over history.

Mirror of the PyTorch user encoder. CAUM's *real* user modelling is
candidate-aware and lives in :mod:`.inter_model`. This module exists so
the model satisfies the :class:`BaseModel` contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import CAUMConfig

from .news_encoder import NewsEncoder


class UserEncoder(nnx.Module):
    """Mean-pool user encoder for the BaseModel contract."""

    def __init__(self, config: CAUMConfig, news_encoder: NewsEncoder):
        self.config = config
        self.news_encoder = news_encoder

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """``inputs``: ``(B, H, k)`` packed → ``(B, news_dim)``."""
        news_embeds = self.news_encoder(inputs, training=training)  # (B, H, D)
        mask = self.news_encoder.valid_mask(inputs).astype(news_embeds.dtype)
        count = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0)
        return jnp.sum(news_embeds * mask[:, :, None], axis=1) / count
