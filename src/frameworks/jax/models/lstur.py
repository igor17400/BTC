"""LSTUR (Long- and Short-term User Representations) -- Flax NNX.

Reference: An et al., "Neural News Recommendation with Long- and Short-term
User Representations", ACL 2019.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import LSTURConfig
from ..layers import AdditiveAttention, compute_mask, overwrite_mask
from .base import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_activation(x: jax.Array, activation: str) -> jax.Array:
    if activation == "relu":
        return jax.nn.relu(x)
    elif activation == "tanh":
        return jnp.tanh(x)
    elif activation == "gelu":
        return jax.nn.gelu(x)
    return x


# ---------------------------------------------------------------------------
# Optional category / subcategory encoders (simple embedding)
# ---------------------------------------------------------------------------

class CategoryEncoder(nnx.Module):
    """Simple embedding for category IDs (LSTUR variant -- no projection)."""

    def __init__(self, config: LSTURConfig, num_categories: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(
            num_embeddings=num_categories + 1,
            features=config.category_embedding_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        embedded = self.embedding(inputs)  # (B, 1, dim)
        return jnp.squeeze(embedded, axis=1)  # (B, dim)


class SubcategoryEncoder(nnx.Module):
    """Simple embedding for subcategory IDs."""

    def __init__(self, config: LSTURConfig, num_subcategories: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(
            num_embeddings=num_subcategories + 1,
            features=config.subcategory_embedding_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        embedded = self.embedding(inputs)
        return jnp.squeeze(embedded, axis=1)


# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------

class NewsEncoder(nnx.Module):
    """CNN-based news encoder with optional category/subcategory concatenation.

    Pipeline: Embedding -> Dropout -> Conv1D -> Dropout -> Masking
    -> AdditiveAttention.  When category/subcategory encoders are provided
    their vectors are concatenated to the title vector.
    """

    def __init__(
        self,
        config: LSTURConfig,
        embedding_layer: nnx.Embed,
        *,
        rngs: nnx.Rngs,
        category_encoder: CategoryEncoder | None = None,
        subcategory_encoder: SubcategoryEncoder | None = None,
    ):
        self.config = config
        self.embedding_layer = embedding_layer
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

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
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def _process_title(
        self, title_tokens: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Encode title tokens through CNN + attention.

        Returns: ``(B, cnn_filter_num)``.
        """
        embedded = self.embedding_layer(title_tokens)
        y = self.dropout1(embedded, deterministic=not training)
        y = _apply_activation(self.cnn(y), self.config.cnn_activation)
        y = self.dropout2(y, deterministic=not training)

        # Masking: zero out padded positions
        mask = compute_mask(title_tokens).astype(y.dtype)
        y = overwrite_mask(y, mask)

        padding_mask = jnp.not_equal(title_tokens, 0)
        return self.additive_attention(y, mask=padding_mask)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Forward pass.

        ``inputs`` may be title-only ``(B, T)`` or concatenated
        ``(B, T + 2)`` containing ``[title | category | subcategory]``.
        """
        has_extra = inputs.shape[-1] > self.config.max_title_length

        if has_extra and (
            self.category_encoder is not None or self.subcategory_encoder is not None
        ):
            title_tokens = inputs[:, : self.config.max_title_length]
            category_id = inputs[
                :,
                self.config.max_title_length : self.config.max_title_length + 1,
            ]
            subcategory_id = inputs[:, self.config.max_title_length + 1 :]

            title_vec = self._process_title(title_tokens, training=training)
            parts = [title_vec]

            if self.category_encoder is not None:
                parts.append(
                    self.category_encoder(category_id, training=training)
                )
            if self.subcategory_encoder is not None:
                parts.append(
                    self.subcategory_encoder(subcategory_id, training=training)
                )

            if len(parts) > 1:
                return jnp.concatenate(parts, axis=-1)
            return title_vec
        else:
            if has_extra:
                title_tokens = inputs[:, : self.config.max_title_length]
            else:
                title_tokens = inputs
            return self._process_title(title_tokens, training=training)


# ---------------------------------------------------------------------------
# User encoder
# ---------------------------------------------------------------------------

class UserEncoder(nnx.Module):
    """GRU-based user encoder with long-term user embeddings.

    Two variants:
    * ``"ini"`` -- user embedding initialises the GRU hidden state.
    * ``"con"`` -- GRU output is concatenated with user embedding and
      projected through a linear layer.
    """

    def __init__(
        self,
        config: LSTURConfig,
        news_encoder: NewsEncoder,
        num_users: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder

        # Long-term user embedding
        self.user_embedding = nnx.Embed(
            num_embeddings=num_users,
            features=config.gru_unit,
            rngs=rngs,
        )
        # Initialise user embeddings to zeros (as in the Keras version)
        self.user_embedding.embedding.value = jnp.zeros_like(
            self.user_embedding.embedding.value
        )

        # Compute news encoder output dim for GRU input size
        news_out_dim = config.cnn_filter_num
        if config.use_category:
            news_out_dim += config.category_embedding_dim
        if config.use_subcategory:
            news_out_dim += config.subcategory_embedding_dim

        # GRU via nnx.RNN + nnx.GRUCell
        self.gru = nnx.RNN(
            nnx.GRUCell(
                in_features=news_out_dim,
                hidden_size=config.gru_unit,
                rngs=rngs,
            ),
        )

        # Dense projection for "con" variant
        if config.type == "con":
            self.concat_dense = nnx.Linear(
                in_features=config.gru_unit * 2,
                out_features=config.gru_unit,
                rngs=rngs,
            )

    def __call__(
        self,
        history_tokens: jax.Array,
        user_indices: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            history_tokens: ``(B, H, T)`` browsing history token ids.
            user_indices: ``(B,)`` or ``(B, 1)`` user index.
            training: Enable dropout.

        Returns:
            ``(B, gru_unit)`` user representations.
        """
        # User embedding
        if user_indices.ndim == 1:
            user_indices = jnp.expand_dims(user_indices, axis=-1)
        long_u_emb = self.user_embedding(user_indices)  # (B, 1, gru_unit)
        long_u_emb = jnp.squeeze(long_u_emb, axis=1)  # (B, gru_unit)

        # TimeDistributed news encoding
        B, H, T = history_tokens.shape
        flat = history_tokens.reshape(B * H, T)
        flat_vecs = self.news_encoder(flat, training=training)
        click_presents = flat_vecs.reshape(B, H, -1)  # (B, H, news_dim)

        if self.config.type == "ini":
            # Use user embedding as initial GRU state
            # nnx.RNN returns all hidden states; take the last one
            all_states = self.gru(click_presents, initial_carry=long_u_emb)
            user_present = all_states[:, -1, :]  # (B, gru_unit)
        elif self.config.type == "con":
            all_states = self.gru(click_presents)
            short_emb = all_states[:, -1, :]  # (B, gru_unit)
            concat_emb = jnp.concatenate([short_emb, long_u_emb], axis=-1)
            user_present = self.concat_dense(concat_emb)
        else:
            raise ValueError(f"Invalid user encoder type: {self.config.type}")

        return user_present


# ---------------------------------------------------------------------------
# Full LSTUR model
# ---------------------------------------------------------------------------

class LSTUR(BaseModel):
    """Neural News Recommendation with Long- and Short-term User Representations."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        num_users: int,
        config: LSTURConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = LSTURConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id
        self.num_users = num_users

        # Shared word embedding
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        vocab_size = int(processed_news["vocab_size"])
        self.embedding_layer = nnx.Embed(
            num_embeddings=vocab_size,
            features=config.embedding_size,
            rngs=rngs,
        )
        self.embedding_layer.embedding.value = jnp.asarray(embeddings_matrix)

        # Optional category / subcategory encoders
        cat_enc: CategoryEncoder | None = None
        subcat_enc: SubcategoryEncoder | None = None
        if config.use_category:
            cat_enc = CategoryEncoder(
                config, int(processed_news["num_categories"]), rngs=rngs
            )
        if config.use_subcategory:
            subcat_enc = SubcategoryEncoder(
                config, int(processed_news["num_subcategories"]), rngs=rngs
            )

        # News encoder
        self.news_encoder = NewsEncoder(
            config,
            self.embedding_layer,
            rngs=rngs,
            category_encoder=cat_enc,
            subcategory_encoder=subcat_enc,
        )

        # User encoder
        self.user_encoder = UserEncoder(
            config, self.news_encoder, num_users, rngs=rngs
        )

    # ---- Scoring helpers ------------------------------------------------

    def score_training_batch(
        self,
        hist_tokens: jax.Array,
        user_ids: jax.Array,
        cand_tokens: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score a training batch (softmax over candidates).

        Returns: ``(B, C)`` softmax probabilities.
        """
        user_repr = self.user_encoder(
            hist_tokens, user_ids, training=training
        )  # (B, D)

        B, C, T = cand_tokens.shape
        flat_cands = cand_tokens.reshape(B * C, T)
        flat_vecs = self.news_encoder(flat_cands, training=training)
        cand_repr = flat_vecs.reshape(B, C, -1)

        scores = jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)
        return jax.nn.softmax(scores, axis=-1)

    def score_multiple_candidates(
        self,
        hist_tokens: jax.Array,
        user_ids: jax.Array,
        cand_tokens: jax.Array,
    ) -> jax.Array:
        """Score multiple candidates at inference (sigmoid)."""
        user_repr = self.user_encoder(hist_tokens, user_ids, training=False)

        B, C, T = cand_tokens.shape
        flat_cands = cand_tokens.reshape(B * C, T)
        flat_vecs = self.news_encoder(flat_cands, training=False)
        cand_repr = flat_vecs.reshape(B, C, -1)

        scores = jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)
        return jax.nn.sigmoid(scores)

    # ---- Helpers for concatenated category/subcategory inputs -----------

    def _maybe_concat_category(
        self, inputs: dict[str, jax.Array], prefix: str, base_key: str
    ) -> jax.Array:
        """If category/subcategory data is present, concatenate it."""
        tokens = inputs[base_key]
        cat_key = f"{prefix}_category"
        subcat_key = f"{prefix}_subcategory"

        if cat_key in inputs and subcat_key in inputs:
            cat = inputs[cat_key]
            if cat.ndim < tokens.ndim:
                cat = jnp.expand_dims(cat, axis=-1)
            subcat = inputs[subcat_key]
            if subcat.ndim < tokens.ndim:
                subcat = jnp.expand_dims(subcat, axis=-1)
            tokens = jnp.concatenate([tokens, cat, subcat], axis=-1)

        return tokens

    # ---- Unified call ---------------------------------------------------

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Unified forward pass.

        Expects ``hist_tokens``, ``cand_tokens``, ``user_ids``.
        Optionally: ``hist_category``, ``hist_subcategory``,
        ``cand_category``, ``cand_subcategory``.
        """
        hist_tokens = self._maybe_concat_category(inputs, "hist", "hist_tokens")
        cand_tokens = self._maybe_concat_category(inputs, "cand", "cand_tokens")
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))

        if training:
            return self.score_training_batch(
                hist_tokens, user_ids, cand_tokens, training=True
            )
        else:
            return self.score_multiple_candidates(hist_tokens, user_ids, cand_tokens)
