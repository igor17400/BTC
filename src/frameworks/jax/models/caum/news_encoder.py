"""CAUM news encoder (Flax NNX) — encoder-agnostic.

Mirror of the PyTorch CAUM news encoder. Three parallel branches feed a
final fusion Linear:

  - Title branch:    text_encoder(news_idx) -> (tokens, mask)
                         -> Dropout -> MHSA -> Dropout
                         -> AdditiveAttention with mask
  - Entity branch:   Embedding -> Dropout -> MHSA -> Dropout
                         -> AdditiveAttention with mask
  - Category branch: Embedding -> Dropout -> Linear

Input layout (packed, last dim):
    [news_idx | entity_id_1 .. entity_id_E | category_id]
    Trailing columns drop when their flag is False.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import CAUMConfig

from ...layers import AdditiveAttention, TextEncoder


class _EntityBranch(nnx.Module):
    """Entity pipeline: Embedding -> Dropout -> MHSA -> Dropout -> AdditivePool."""

    def __init__(self, config: CAUMConfig, num_entities: int, *, rngs: nnx.Rngs):
        self.config = config
        self.embedding = nnx.Embed(
            num_embeddings=num_entities,
            features=config.entity_embedding_dim,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        entity_out_dim = config.entity_num_heads * config.entity_head_dim
        self.mhsa = nnx.MultiHeadAttention(
            num_heads=config.entity_num_heads,
            in_features=config.entity_embedding_dim,
            qkv_features=entity_out_dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=config.entity_embedding_dim,
            query_vec_dim=config.news_attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(self, entity_indices: jax.Array, *, training: bool) -> jax.Array:
        """``entity_indices``: ``(N*, E)`` ints → ``(N*, entity_embedding_dim)``."""
        entity_mask = jnp.not_equal(entity_indices, 0)  # (N*, E)
        emb = self.dropout1(self.embedding(entity_indices), deterministic=not training)
        attn_mask = entity_mask[:, None, None, :]
        y = self.mhsa(emb, emb, mask=attn_mask, deterministic=not training)
        y = self.dropout2(y, deterministic=not training)
        return self.additive_attention(y, mask=entity_mask)


class _CategoryBranch(nnx.Module):
    """Category pipeline: Embedding -> Dropout -> Linear."""

    def __init__(self, config: CAUMConfig, num_categories: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(
            num_embeddings=num_categories + 1,
            features=config.category_embedding_dim,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.dense = nnx.Linear(
            config.category_embedding_dim,
            config.category_embedding_dim,
            rngs=rngs,
        )

    def __call__(self, category_id: jax.Array, *, training: bool) -> jax.Array:
        """``category_id``: ``(N*,)`` ints → ``(N*, category_embedding_dim)``."""
        emb = self.dropout(self.embedding(category_id), deterministic=not training)
        return self.dense(emb)


class NewsEncoder(nnx.Module):
    """CAUM news encoder. Outputs ``(*leading, news_dim)``."""

    def __init__(
        self,
        config: CAUMConfig,
        text_encoder: TextEncoder,
        num_entities: int = 0,
        num_categories: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        text_dim = text_encoder.output_dim
        if text_dim % config.news_num_heads != 0:
            raise ValueError(
                f"CAUM title MHSA: text_encoder.output_dim={text_dim} is not "
                f"divisible by news_num_heads={config.news_num_heads}. Override "
                "spec.model.architecture.news_encoder.num_heads in the experiment "
                f"yaml to a divisor of {text_dim} (e.g. 12 for BERT-base 768d → "
                "head_dim 64)."
            )

        self.config = config
        self.text_encoder = text_encoder
        self.text_dim = int(text_dim)

        self.title_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.title_mhsa = nnx.MultiHeadAttention(
            num_heads=config.news_num_heads,
            in_features=text_dim,
            qkv_features=text_dim,
            decode=False,
            rngs=rngs,
        )
        self.title_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.title_attention = AdditiveAttention(
            input_dim=text_dim,
            query_vec_dim=config.news_attention_hidden_dim,
            rngs=rngs,
        )

        self.entity_branch = (
            _EntityBranch(config, num_entities, rngs=rngs)
            if config.use_entity
            else None
        )
        self.category_branch = (
            _CategoryBranch(config, num_categories, rngs=rngs)
            if config.use_category
            else None
        )

        fusion_in = text_dim
        if self.entity_branch is not None:
            fusion_in += config.entity_embedding_dim
        if self.category_branch is not None:
            fusion_in += config.category_embedding_dim
        self.fusion_dense = nnx.Linear(fusion_in, config.news_dim, rngs=rngs)

        self._n_entity = config.max_entities if config.use_entity else 0
        self._n_cols = 1 + self._n_entity + int(config.use_category)

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
        if self.entity_branch is not None:
            entity_indices = inputs[..., col : col + self._n_entity]
            col += self._n_entity
        if self.category_branch is not None:
            category_id = inputs[..., col]
        return news_idx, entity_indices, category_id

    def _encode_title(self, news_idx: jax.Array, *, training: bool) -> jax.Array:
        """Encode title tokens through TextEncoder + MHSA + additive pool.

        Args:
            news_idx: ``(N*,)`` int tensor of parsed news ids.

        Returns:
            ``(N*, text_dim)`` title vectors.
        """
        tokens, mask = self.text_encoder(news_idx)  # (N*, T, text_dim), (N*, T)
        y = self.title_dropout(tokens, deterministic=not training)

        valid_mask = jnp.not_equal(mask, 0)
        attn_mask = valid_mask[:, None, None, :]
        y = self.title_mhsa(y, y, mask=attn_mask, deterministic=not training)
        y = self.title_dropout2(y, deterministic=not training)
        return self.title_attention(y, mask=valid_mask)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Encode a batch of news articles.

        Args:
            inputs: Packed ``(..., k)`` int with column order
                ``[news_idx | entities | category]`` (trailing columns
                drop when their flag is False), or plain ``(...,)`` int
                when no structured views are enabled.
        """
        news_idx, entity_indices, category_id = self._split_columns(inputs)
        leading_shape = news_idx.shape

        flat_news_idx = news_idx.reshape(-1)
        title_vec = self._encode_title(flat_news_idx, training=training)

        parts: list[jax.Array] = [title_vec]
        if self.entity_branch is not None:
            assert entity_indices is not None
            flat_entities = entity_indices.reshape(-1, self._n_entity)
            parts.append(self.entity_branch(flat_entities, training=training))
        if self.category_branch is not None:
            assert category_id is not None
            parts.append(
                self.category_branch(category_id.reshape(-1), training=training)
            )

        fused = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else title_vec
        news_vec = self.fusion_dense(fused)
        return news_vec.reshape(*leading_shape, self.config.news_dim)
