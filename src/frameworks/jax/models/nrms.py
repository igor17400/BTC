"""NRMS (Neural News Recommendation with Multi-Head Self-Attention) -- Flax NNX.

Reference: Wu et al., "Neural News Recommendation with Multi-Head
Self-Attention", EMNLP 2019.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import NRMSConfig

from ..layers import AdditiveAttention, PLMTokenNewsEncoder
from .base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(nnx.Module):
    """GloVe-token news encoder.

    Pipeline: Embedding → Dropout → MultiHeadAttention → Dropout → AdditiveAttention.

    Honors the shared encoder contract:
        - ``__call__(features, training=...) -> (*leading, news_dim)`` for any
          leading shape ``(*leading, T)``.
        - ``valid_mask(features) -> bool of shape (*leading,)``.

    A GloVe news slot is "valid" if any token in its title is non-zero.
    """

    def __init__(
        self,
        config: NRMSConfig,
        embedding_layer: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_dim = config.embedding_size
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

    @staticmethod
    def valid_mask(features: jax.Array) -> jax.Array:
        """A GloVe news slot is valid when any token is non-zero."""
        return jnp.any(jnp.not_equal(features, 0), axis=-1)

    def __call__(self, features: jax.Array, *, training: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            features: ``(*leading, T)`` int32 token ids.
            training: Enable dropout.

        Returns:
            ``(*leading, news_dim)`` news vectors.
        """
        leading_shape = features.shape[:-1]
        T = features.shape[-1]
        inputs = features.reshape(-1, T)  # (B*, T)

        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded, deterministic=not training)

        padding_mask = jnp.not_equal(inputs, 0)  # (B*, T)
        attn_mask = padding_mask[:, None, None, :]  # (B*, 1, 1, T)

        y = self.multi_head_attention(y, y, mask=attn_mask, deterministic=not training)
        y = self.dropout2(y, deterministic=not training)

        news_repr = self.additive_attention(y, mask=padding_mask)  # (B*, news_dim)
        return news_repr.reshape(*leading_shape, self.news_dim)


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

    def __call__(
        self, history_features: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Forward pass.

        Args:
            history_features: Any shape the underlying ``news_encoder``
                accepts, with the leading axes being ``(batch, history_length)``.
                The encoder handles per-slot encoding; this method is
                encoder-agnostic.
            training: Enable dropout.

        Returns:
            ``(batch, embedding_size)`` user representations.
        """
        click_title_presents = self.news_encoder(history_features, training=training)
        history_mask = self.news_encoder.valid_mask(history_features)  # (B, H)
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
        processed_news: dict[str, Any],
        config: NRMSConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = NRMSConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        encoder_type = getattr(config.encoder, "type", "glove")
        if encoder_type == "glove":
            # Shared word embedding initialised from pre-trained weights.
            embeddings_matrix = np.asarray(processed_news["embeddings"])
            vocab_size = int(processed_news["vocab_size"])
            self.embedding_layer = nnx.Embed(
                num_embeddings=vocab_size,
                features=config.embedding_size,
                rngs=rngs,
            )
            self.embedding_layer.embedding.value = jnp.asarray(embeddings_matrix)
            self.news_encoder = NewsEncoder(config, self.embedding_layer, rngs=rngs)
        else:
            # PLM mode — token-level cache + model-side pooler.
            if "plm_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"NRMS with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id']. Call "
                    "attach_plm_embeddings(..., level='token') in the runner."
                )
            pooler = config.text_pooler
            self.news_encoder = PLMTokenNewsEncoder(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                news_dim=config.embedding_size,
                pooler_type=pooler.type,
                attention_query_dim=pooler.attention_query_dim,
                num_heads=pooler.num_heads,
                head_dim=pooler.head_dim,
                dropout_rate=pooler.dropout_rate,
                rngs=rngs,
            )

        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)

    # ---- Scoring ---------------------------------------------------------

    def score_training_batch(
        self,
        history_features: jax.Array,
        candidate_features: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score a training batch (raw logits — loss handles softmax).

        Encoder-agnostic: the encoder handles arbitrary leading shapes,
        so we just call it with the natural (B, H, ...) and (B, C, ...)
        leading axes and let it produce (B, H, D) / (B, C, D) directly.

        Args:
            history_features: ``(B, H, *)`` history news features.
            candidate_features: ``(B, C, *)`` candidate news features.
            training: Enable dropout.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        user_repr = self.user_encoder(history_features, training=training)  # (B, E)
        cand_repr = self.news_encoder(
            candidate_features, training=training
        )  # (B, C, E)
        scores = jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)  # (B, C)
        return scores

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Forward pass for training. Returns raw logits.

        Uses uniform feature keys: ``hist_features`` and ``cand_features``.
        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        return self.score_training_batch(
            inputs["hist_features"], inputs["cand_features"], training=training
        )
