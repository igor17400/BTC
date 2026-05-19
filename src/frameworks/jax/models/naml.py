"""NAML (Neural News Recommendation with Attentive Multi-View Learning) -- Flax NNX.

Reference: Wu et al., "Neural News Recommendation with Attentive Multi-View
Learning", IJCAI 2019.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import NAMLConfig

from ..layers import AdditiveAttention, PLMTokenCNNEncoder, apply_activation
from .base import BaseModel

# ---------------------------------------------------------------------------
# View encoders
# ---------------------------------------------------------------------------


class TitleEncoder(nnx.Module):
    """Encode a news title: Embedding -> Dropout -> Conv1D -> Dropout -> AdditiveAttention."""

    def __init__(
        self,
        config: NAMLConfig,
        embedding_layer: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding_layer = embedding_layer

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        # Conv1D: (batch, seq, features_in) -> (batch, seq, features_out)
        self.cnn = nnx.Conv(
            in_features=config.embedding_size,
            out_features=config.cnn_filter_num,
            kernel_size=(config.cnn_kernel_size,),
            padding="SAME",
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.word_attention_query_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Args: inputs (B, title_len) int32.  Returns: (B, cnn_filter_num)."""
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded, deterministic=not training)
        y = apply_activation(self.cnn(y), self.config.activation)
        y = self.dropout2(y, deterministic=not training)

        padding_mask = jnp.not_equal(inputs, 0)
        return self.additive_attention(y, mask=padding_mask)


class AbstractEncoder(nnx.Module):
    """Encode a news abstract: same pipeline as TitleEncoder."""

    def __init__(
        self,
        config: NAMLConfig,
        embedding_layer: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding_layer = embedding_layer

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.cnn = nnx.Conv(
            in_features=config.embedding_size,
            out_features=config.cnn_filter_num,
            kernel_size=(config.cnn_kernel_size,),
            padding="SAME",
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.word_attention_query_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Args: inputs (B, abstract_len) int32.  Returns: (B, cnn_filter_num)."""
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded, deterministic=not training)
        y = apply_activation(self.cnn(y), self.config.activation)
        y = self.dropout2(y, deterministic=not training)

        padding_mask = jnp.not_equal(inputs, 0)
        return self.additive_attention(y, mask=padding_mask)


class CategoryEncoder(nnx.Module):
    """Embed + project a category ID to ``cnn_filter_num`` dimensions."""

    def __init__(
        self,
        config: NAMLConfig,
        num_categories: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding = nnx.Embed(
            num_embeddings=num_categories + 1,
            features=config.category_embedding_dim,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            in_features=config.category_embedding_dim,
            out_features=config.cnn_filter_num,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Args: inputs (B, 1) int32.  Returns: (B, cnn_filter_num)."""
        embedded = self.embedding(inputs)  # (B, 1, cat_dim)
        projected = apply_activation(self.projection(embedded), self.config.activation)
        return jnp.squeeze(projected, axis=1)  # (B, cnn_filter_num)


class SubcategoryEncoder(nnx.Module):
    """Embed + project a subcategory ID to ``cnn_filter_num`` dimensions."""

    def __init__(
        self,
        config: NAMLConfig,
        num_subcategories: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding = nnx.Embed(
            num_embeddings=num_subcategories + 1,
            features=config.subcategory_embedding_dim,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            in_features=config.subcategory_embedding_dim,
            out_features=config.cnn_filter_num,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Args: inputs (B, 1) int32.  Returns: (B, cnn_filter_num)."""
        embedded = self.embedding(inputs)
        projected = apply_activation(self.projection(embedded), self.config.activation)
        return jnp.squeeze(projected, axis=1)


# ---------------------------------------------------------------------------
# Multi-view news encoder
# ---------------------------------------------------------------------------


class NewsEncoder(nnx.Module):
    """Combine title, abstract, category, subcategory views with attention.

    Encoder-aware unpacking: in GloVe mode the input is the original
    concatenated token layout; in PLM mode the input is a 3-column
    tensor ``[news_idx, category_id, subcategory_id]`` and the text
    encoders are :class:`PLMTokenCNNEncoder` lookups.
    """

    def __init__(
        self,
        config: NAMLConfig,
        title_encoder: nnx.Module,
        abstract_encoder: nnx.Module,
        category_encoder: CategoryEncoder,
        subcategory_encoder: SubcategoryEncoder,
        *,
        encoder_type: str = "glove",
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.encoder_type = encoder_type
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        self.view_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.view_attention_query_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Args:
            GloVe mode: ``inputs (B, title_len + abstract_len + 2)`` int32.
            PLM mode:   ``inputs (B, 3)`` = ``[news_idx, category, subcategory]``.

        Returns: ``(B, cnn_filter_num)``.
        """
        cfg = self.config
        if self.encoder_type == "glove":
            title_tokens = inputs[:, : cfg.max_title_length]
            abstract_tokens = inputs[
                :, cfg.max_title_length : cfg.max_title_length + cfg.max_abstract_length
            ]
            category_id = inputs[
                :,
                cfg.max_title_length + cfg.max_abstract_length : cfg.max_title_length
                + cfg.max_abstract_length
                + 1,
            ]
            subcategory_id = inputs[
                :, cfg.max_title_length + cfg.max_abstract_length + 1 :
            ]
            title_vec = self.title_encoder(title_tokens, training=training)
            abstract_vec = self.abstract_encoder(abstract_tokens, training=training)
        else:
            # PLM mode: column 0 = parsed news id (drives BOTH PLM lookups).
            news_idx = inputs[:, 0]
            category_id = inputs[:, 1:2]
            subcategory_id = inputs[:, 2:3]
            title_vec = self.title_encoder(news_idx, training=training)
            abstract_vec = self.abstract_encoder(news_idx, training=training)

        category_vec = self.category_encoder(category_id, training=training)
        subcategory_vec = self.subcategory_encoder(subcategory_id, training=training)

        views = jnp.stack(
            [title_vec, abstract_vec, category_vec, subcategory_vec], axis=1
        )
        return self.view_attention(views)


# ---------------------------------------------------------------------------
# User encoder
# ---------------------------------------------------------------------------


class UserEncoder(nnx.Module):
    """Encode a user from browsing history via TimeDistributed + AdditiveAttention."""

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

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Args:
            inputs ``(B, H, feature_size)`` int32. GloVe mode uses the long
            packed-token layout; PLM mode uses ``[news_idx, cat, subcat]``.
        Returns: ``(B, cnn_filter_num)``.
        """
        B, H, F = inputs.shape

        # TimeDistributed news encoding
        flat = inputs.reshape(B * H, F)
        flat_vecs = self.news_encoder(flat, training=training)
        news_vecs = flat_vecs.reshape(B, H, -1)

        # Validity mask: encoder-dependent.
        if self.news_encoder.encoder_type == "glove":
            title_tokens = inputs[:, :, : self.config.max_title_length]
            history_mask = jnp.any(jnp.not_equal(title_tokens, 0), axis=-1)
        else:
            history_mask = jnp.not_equal(inputs[:, :, 0], 0)

        return self.user_attention(news_vecs, mask=history_mask)


# ---------------------------------------------------------------------------
# Full NAML model
# ---------------------------------------------------------------------------


class NAML(BaseModel):
    """Neural News Recommendation with Attentive Multi-View Learning."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NAMLConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = NAMLConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        encoder_type = getattr(getattr(config, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        if encoder_type == "glove":
            # Shared word embedding initialised from pretrained GloVe.
            embeddings_matrix = np.asarray(processed_news["embeddings"])
            vocab_size = int(processed_news["vocab_size"])
            self.embedding_layer = nnx.Embed(
                num_embeddings=vocab_size,
                features=config.embedding_size,
                rngs=rngs,
            )
            self.embedding_layer.embedding.value = jnp.asarray(embeddings_matrix)
            self.title_encoder = TitleEncoder(config, self.embedding_layer, rngs=rngs)
            self.abstract_encoder = AbstractEncoder(
                config, self.embedding_layer, rngs=rngs
            )
        else:
            # PLM mode — token-level lookups + NAML's Conv1d/AddAttn pipeline.
            for required in (
                "plm_token_embeddings_by_id",
                "plm_attention_mask_by_id",
                "plm_abstract_token_embeddings_by_id",
                "plm_abstract_attention_mask_by_id",
            ):
                if required not in processed_news:
                    raise KeyError(
                        f"NAML with encoder.type='{encoder_type}' requires "
                        f"processed_news['{required}']. Make sure the encoder "
                        "yaml sets text_field='title' and "
                        "text_field_abstract='abstract' and the runner has "
                        "called attach_plm_embeddings for both."
                    )
            plm_dim = int(processed_news["plm_dim"])
            self.title_encoder = PLMTokenCNNEncoder(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                news_dim=config.cnn_filter_num,
                plm_dim=plm_dim,
                cnn_kernel_size=config.cnn_kernel_size,
                dropout_rate=config.dropout_rate,
                word_attention_query_dim=config.word_attention_query_dim,
                activation=config.activation,
                rngs=rngs,
            )
            self.abstract_encoder = PLMTokenCNNEncoder(
                processed_news["plm_abstract_token_embeddings_by_id"],
                processed_news["plm_abstract_attention_mask_by_id"],
                news_dim=config.cnn_filter_num,
                plm_dim=plm_dim,
                cnn_kernel_size=config.cnn_kernel_size,
                dropout_rate=config.dropout_rate,
                word_attention_query_dim=config.word_attention_query_dim,
                activation=config.activation,
                rngs=rngs,
            )

        self.category_encoder = CategoryEncoder(
            config, int(processed_news["num_categories"]), rngs=rngs
        )
        self.subcategory_encoder = SubcategoryEncoder(
            config, int(processed_news["num_subcategories"]), rngs=rngs
        )

        self.news_encoder = NewsEncoder(
            config,
            self.title_encoder,
            self.abstract_encoder,
            self.category_encoder,
            self.subcategory_encoder,
            encoder_type=encoder_type,
            rngs=rngs,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)

    # ---- Scoring helpers ------------------------------------------------

    def score_training_batch(
        self,
        hist_concat: jax.Array,
        cand_concat: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score training batch (raw logits — loss handles softmax).

        Args:
            hist_concat: ``(B, H, F)`` concatenated user history features.
            cand_concat: ``(B, C, F)`` concatenated candidate features.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        user_repr = self.user_encoder(hist_concat, training=training)  # (B, D)

        B, C, F = cand_concat.shape
        flat_cands = cand_concat.reshape(B * C, F)
        flat_vecs = self.news_encoder(flat_cands, training=training)
        cand_repr = flat_vecs.reshape(B, C, -1)  # (B, C, D)

        scores = jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)
        return scores  # raw logits; loss handles softmax

    # ---- Helpers to build concatenated inputs ---------------------------

    def _concat_features(self, inputs: dict[str, jax.Array], prefix: str) -> jax.Array:
        """Build a single concatenated tensor from separate feature arrays.

        GloVe mode: ``[title_tokens | abstract_tokens | cat | subcat]``
            along the last dim.
        PLM mode: ``[news_idx | cat | subcat]`` along the last dim.
        """
        if self.encoder_type != "glove":
            news_idx = inputs[f"{prefix}_features"]
            parts = [jnp.expand_dims(news_idx, axis=-1)]
            if f"{prefix}_category" in inputs:
                cat = inputs[f"{prefix}_category"]
                if cat.ndim == news_idx.ndim:
                    cat = jnp.expand_dims(cat, axis=-1)
                parts.append(cat)
            if f"{prefix}_subcategory" in inputs:
                sub = inputs[f"{prefix}_subcategory"]
                if sub.ndim == news_idx.ndim:
                    sub = jnp.expand_dims(sub, axis=-1)
                parts.append(sub)
            return jnp.concatenate(parts, axis=-1)

        parts = [inputs[f"{prefix}_tokens"], inputs[f"{prefix}_abstract_tokens"]]

        cat = inputs[f"{prefix}_category"]
        if cat.ndim < inputs[f"{prefix}_tokens"].ndim:
            cat = jnp.expand_dims(cat, axis=-1)
        parts.append(cat)

        subcat = inputs[f"{prefix}_subcategory"]
        if subcat.ndim < inputs[f"{prefix}_tokens"].ndim:
            subcat = jnp.expand_dims(subcat, axis=-1)
        parts.append(subcat)

        return jnp.concatenate(parts, axis=-1)

    # ---- Unified call ---------------------------------------------------

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Forward pass for training. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        hist_concat = self._concat_features(inputs, "hist")
        cand_concat = self._concat_features(inputs, "cand")
        return self.score_training_batch(hist_concat, cand_concat, training=training)
