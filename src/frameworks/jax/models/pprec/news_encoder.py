"""PP-Rec news encoder (Flax NNX) — encoder-agnostic ``co1`` recipe.

Mirror of the PyTorch :class:`NewsEncoder`. Title MHSA + entity MHSA +
bidirectional cross-attention (MHCA) between title words and entities,
then concat + Dense fusion to ``news_dim``.

When ``text_encoder.output_dim`` differs from ``config.embedding_size``
(PLM mode), a Linear projects down to ``embedding_size`` before MHSA —
keeps every downstream MHA / cross-attn dim equal to the paper's
``embedding_size`` so ``num_heads`` doesn't need re-tuning per encoder.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import PPRecConfig

from ...layers import AdditiveAttention, CrossAttention, TextEncoder


class NewsEncoder(nnx.Module):
    """PP-Rec news encoder (paper ``co1`` variant) — Flax NNX."""

    def __init__(
        self,
        config: PPRecConfig,
        text_encoder: TextEncoder,
        entity_embedding_layer: nnx.Embed | None = None,
        category_embedding_layer: nnx.Embed | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.text_encoder = text_encoder
        self.entity_embedding = entity_embedding_layer
        self.category_embedding = category_embedding_layer

        text_dim = text_encoder.output_dim
        emb_dim = config.embedding_size
        self.text_dim = int(text_dim)
        self.text_projection = (
            nnx.Linear(text_dim, emb_dim, rngs=rngs) if text_dim != emb_dim else None
        )

        co_out_dim = config.co_num_heads * config.co_head_dim

        # --- Word (title) branch — MHSA at embedding_size ---
        self.word_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.word_mhsa = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=emb_dim,
            qkv_features=config.num_heads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        # Flax NNX MHA has out_features = in_features by default, so the
        # MHSA output stays at emb_dim regardless of qkv_features.
        word_concat_dim = emb_dim + (
            co_out_dim if entity_embedding_layer is not None else 0
        )
        self.word_proj = nnx.Linear(word_concat_dim, config.news_dim, rngs=rngs)
        self.word_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.word_attention = AdditiveAttention(
            input_dim=config.news_dim,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

        # --- Entity branch + bidirectional MHCA ---
        if entity_embedding_layer is not None:
            self.entity_mhsa = nnx.MultiHeadAttention(
                num_heads=config.num_heads,
                in_features=config.entity_embedding_dim,
                qkv_features=config.num_heads * config.head_dim,
                decode=False,
                rngs=rngs,
            )
            entity_concat_dim = config.entity_embedding_dim + co_out_dim
            self.entity_proj = nnx.Linear(entity_concat_dim, config.news_dim, rngs=rngs)
            self.entity_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
            self.entity_attention = AdditiveAttention(
                input_dim=config.news_dim,
                query_vec_dim=config.attention_hidden_dim,
                rngs=rngs,
            )

            self.title_mhca = CrossAttention(
                q_dim=emb_dim,
                kv_dim=config.entity_embedding_dim,
                num_heads=config.co_num_heads,
                head_dim=config.co_head_dim,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
            self.entity_mhca = CrossAttention(
                q_dim=config.entity_embedding_dim,
                kv_dim=emb_dim,
                num_heads=config.co_num_heads,
                head_dim=config.co_head_dim,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )

        # --- Category branch ---
        if category_embedding_layer is not None:
            self.category_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # --- Fusion ---
        fusion_dim = config.news_dim
        if entity_embedding_layer is not None:
            fusion_dim += config.news_dim
        if category_embedding_layer is not None:
            fusion_dim += config.category_embedding_dim
        self.fusion_dense = nnx.Linear(fusion_dim, config.news_dim, rngs=rngs)

        self._n_entity = (
            config.max_entities if entity_embedding_layer is not None else 0
        )
        self._n_cols = 1 + self._n_entity + int(category_embedding_layer is not None)

    @staticmethod
    def valid_mask(packed_features: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero."""
        if packed_features.ndim >= 2 and packed_features.shape[-1] > 1:
            return jnp.not_equal(packed_features[..., 0], 0)
        return jnp.not_equal(packed_features, 0)

    def _split_columns(
        self, inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        """Return ``(news_idx, entity_indices, category_id)`` from inputs."""
        if self._n_cols == 1:
            news_idx = (
                inputs if inputs.shape[-1] != 1 or inputs.ndim == 1 else inputs[..., 0]
            )
            return news_idx, None, None

        news_idx = inputs[..., 0]
        col = 1
        entity_indices: jax.Array | None = None
        category_id: jax.Array | None = None
        if self.entity_embedding is not None:
            entity_indices = inputs[..., col : col + self._n_entity]
            col += self._n_entity
        if self.category_embedding is not None:
            category_id = inputs[..., col]
        return news_idx, entity_indices, category_id

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Encode a batch of news articles.

        Args:
            inputs: Packed ``(..., k)`` int with column order
                ``[news_idx | entities | category]`` (trailing columns
                drop when their branch is disabled).
        """
        news_idx, entity_indices, category_id = self._split_columns(inputs)
        leading_shape = news_idx.shape

        flat_news_idx = news_idx.reshape(-1)
        # --- Title path ---
        tokens, title_mask_raw = self.text_encoder(
            flat_news_idx
        )  # (N*, T, text_dim), (N*, T)
        title_keep = jnp.not_equal(title_mask_raw, 0)
        title_emb = (
            tokens if self.text_projection is None else self.text_projection(tokens)
        )
        title_emb = self.word_dropout(title_emb, deterministic=not training)

        has_entity = self.entity_embedding is not None
        has_category = self.category_embedding is not None

        if has_entity:
            assert entity_indices is not None
            flat_entities = entity_indices.reshape(-1, self._n_entity)
            entity_keep = jnp.not_equal(flat_entities, 0)
            entity_emb = self.entity_embedding(flat_entities)  # (N*, E, ent_dim)

            entity_kv_mask = entity_keep[:, None, None, :]
            title_co = self.title_mhca(
                title_emb, entity_emb, mask=entity_kv_mask, deterministic=not training
            )
            title_kv_mask = title_keep[:, None, None, :]
            entity_co = self.entity_mhca(
                entity_emb, title_emb, mask=title_kv_mask, deterministic=not training
            )

        # --- Title self-attention + concat ---
        title_self_mask = title_keep[:, None, None, :]
        title_vecs = self.word_mhsa(
            title_emb, title_emb, mask=title_self_mask, deterministic=not training
        )
        if has_entity:
            title_vecs = jnp.concatenate([title_vecs, title_co], axis=-1)
        title_vecs = self.word_proj(title_vecs)
        title_vecs = self.word_dropout2(title_vecs, deterministic=not training)
        title_vec = self.word_attention(title_vecs, mask=title_keep)

        vecs: list[jax.Array] = [title_vec]

        # --- Entity self-attention + concat ---
        if has_entity:
            entity_self_mask = entity_keep[:, None, None, :]
            entity_vecs = self.entity_mhsa(
                entity_emb,
                entity_emb,
                mask=entity_self_mask,
                deterministic=not training,
            )
            entity_vecs = jnp.concatenate([entity_vecs, entity_co], axis=-1)
            entity_vecs = self.entity_proj(entity_vecs)
            entity_vecs = self.entity_dropout(entity_vecs, deterministic=not training)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_keep)
            vecs.append(entity_vec)

        # --- Category branch ---
        if has_category:
            assert category_id is not None
            cat_emb = self.category_embedding(category_id.reshape(-1))
            cat_emb = self.category_dropout(cat_emb, deterministic=not training)
            vecs.append(cat_emb)

        # --- Fusion ---
        fused = jnp.concatenate(vecs, axis=-1) if len(vecs) > 1 else title_vec
        news_vec = self.fusion_dense(fused)
        return news_vec.reshape(*leading_shape, self.config.news_dim)
