"""NRMS (Neural News Recommendation with Multi-Head Self-Attention) -- Flax NNX.

Reference: Wu et al., "Neural News Recommendation with Multi-Head
Self-Attention", EMNLP 2019.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import NRMSConfig
from ..layers import AdditiveAttention
from .base import BaseModel


# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------

class NewsEncoder(nnx.Module):
    """Encode a news article from its title token sequence.

    Pipeline: Embedding -> Dropout -> MultiHeadAttention -> Dropout
    -> AdditiveAttention -> news vector.
    """

    def __init__(
        self,
        config: NRMSConfig,
        embedding_layer: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding_layer = embedding_layer

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.multi_head_attention = nnx.MultiHeadAttention(
            num_heads=config.multiheads,
            in_features=config.embedding_size,
            qkv_features=config.multiheads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            inputs: ``(batch, title_length)`` int32 token ids.
            training: Whether to enable dropout.

        Returns:
            ``(batch, embedding_size)`` news representations.
        """
        embedded = self.embedding_layer(inputs)  # (B, T, E)
        y = self.dropout1(embedded, deterministic=not training)

        # Padding mask for attention: True where tokens are non-zero
        padding_mask = jnp.not_equal(inputs, 0)  # (B, T)

        # Flax NNX MultiHeadAttention expects a mask of shape (B, 1, T_q, T_kv)
        # where True means "attend to this position".
        attn_mask = padding_mask[:, None, None, :]  # (B, 1, 1, T)

        y = self.multi_head_attention(y, y, mask=attn_mask, deterministic=not training)
        y = self.dropout2(y, deterministic=not training)

        news_repr = self.additive_attention(y, mask=padding_mask)
        return news_repr


# ---------------------------------------------------------------------------
# User encoder
# ---------------------------------------------------------------------------

class UserEncoder(nnx.Module):
    """Encode a user from browsing history.

    Pipeline: TimeDistributed(NewsEncoder) -> MultiHeadAttention
    -> AdditiveAttention -> user vector.
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
            num_heads=config.multiheads,
            in_features=config.embedding_size,
            qkv_features=config.multiheads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        self.user_additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            inputs: ``(batch, history_length, title_length)`` int32 token ids.
            training: Enable dropout.

        Returns:
            ``(batch, embedding_size)`` user representations.
        """
        batch_size, hist_len, title_len = inputs.shape

        # TimeDistributed: encode every news item in history
        flat_inputs = inputs.reshape(batch_size * hist_len, title_len)
        flat_vecs = self.news_encoder(flat_inputs, training=training)
        click_title_presents = flat_vecs.reshape(batch_size, hist_len, -1)

        # History mask: True where at least one token is non-zero
        history_mask = jnp.any(jnp.not_equal(inputs, 0), axis=-1)  # (B, H)

        attn_mask = history_mask[:, None, None, :]  # (B, 1, 1, H)

        y = self.browsed_news_attention(
            click_title_presents,
            click_title_presents,
            mask=attn_mask,
            deterministic=not training,
        )

        user_repr = self.user_additive_attention(y, mask=history_mask)
        return user_repr


# ---------------------------------------------------------------------------
# Full NRMS model
# ---------------------------------------------------------------------------

class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention.

    Combines a shared word embedding, news encoder, user encoder, and
    dot-product scoring.
    """

    def __init__(
        self,
        processed_news: Dict[str, Any],
        config: NRMSConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = NRMSConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        # Shared word embedding initialised from pre-trained weights
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        vocab_size = int(processed_news["vocab_size"])
        self.embedding_layer = nnx.Embed(
            num_embeddings=vocab_size,
            features=config.embedding_size,
            rngs=rngs,
        )
        # Overwrite the randomly-initialised embedding table
        self.embedding_layer.embedding.value = jnp.asarray(embeddings_matrix)

        # Sub-modules
        self.news_encoder = NewsEncoder(config, self.embedding_layer, rngs=rngs)
        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)

    # ---- Scoring helpers ------------------------------------------------

    def score_training_batch(
        self,
        hist_tokens: jax.Array,
        cand_tokens: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score a training batch (softmax over candidates).

        Args:
            hist_tokens: ``(B, H, T)`` user history.
            cand_tokens: ``(B, C, T)`` candidate news.
            training: Enable dropout.

        Returns:
            ``(B, C)`` softmax probabilities.
        """
        user_repr = self.user_encoder(hist_tokens, training=training)  # (B, E)

        B, C, T = cand_tokens.shape
        flat_cands = cand_tokens.reshape(B * C, T)
        flat_vecs = self.news_encoder(flat_cands, training=training)
        cand_repr = flat_vecs.reshape(B, C, -1)  # (B, C, E)

        scores = jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)  # (B, C)
        return jax.nn.softmax(scores, axis=-1)

    def score_multiple_candidates(
        self,
        hist_tokens: jax.Array,
        cand_tokens: jax.Array,
    ) -> jax.Array:
        """Score multiple candidates at inference time (sigmoid).

        Returns:
            ``(B, C)`` sigmoid scores.
        """
        user_repr = self.user_encoder(hist_tokens, training=False)

        B, C, T = cand_tokens.shape
        flat_cands = cand_tokens.reshape(B * C, T)
        flat_vecs = self.news_encoder(flat_cands, training=False)
        cand_repr = flat_vecs.reshape(B, C, -1)

        scores = jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)
        return jax.nn.sigmoid(scores)

    # ---- Unified call ---------------------------------------------------

    def __call__(
        self,
        inputs: Dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Unified forward pass dispatching on input keys.

        Training expects ``{"hist_tokens": ..., "cand_tokens": ...}``.
        Inference expects the same keys but runs without dropout.
        """
        hist = inputs["hist_tokens"]
        cand = inputs["cand_tokens"]

        if training:
            return self.score_training_batch(hist, cand, training=True)
        else:
            return self.score_multiple_candidates(hist, cand)
