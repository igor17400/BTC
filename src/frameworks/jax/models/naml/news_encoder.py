"""NAML news encoder (Flax NNX) — encoder-agnostic, multi-view.

Mirror of the PyTorch news encoder. Four parallel view encoders
(title, abstract, category, subcategory) feed a view-level
:class:`AdditiveAttention`. The two text views consume an injected
:class:`TextEncoder` whose ``output_dim`` is the native text dim;
the CNN handles the per-view projection to ``cnn_filter_num``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import NAMLConfig

from ...layers import AdditiveAttention, TextEncoder, apply_activation


class _TextViewEncoder(nnx.Module):
    """Title / abstract pipeline: text_encoder -> Conv -> additive pool."""

    def __init__(
        self,
        config: NAMLConfig,
        text_encoder: TextEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.text_encoder = text_encoder

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        # NNX Conv expects (B, T, in_features) -> (B, T, out_features).
        self.cnn = nnx.Conv(
            in_features=text_encoder.output_dim,
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

    def __call__(self, news_idx: jax.Array, *, training: bool = False) -> jax.Array:
        """news_idx: ``(N*,)`` ints → ``(N*, cnn_filter_num)`` vector."""
        tokens, mask = self.text_encoder(news_idx)
        y = self.dropout1(tokens, deterministic=not training)
        y = apply_activation(self.cnn(y), self.config.activation)
        y = self.dropout2(y, deterministic=not training)

        valid = jnp.not_equal(mask, 0)
        return self.additive_attention(y, mask=valid)


class _StructuredViewEncoder(nnx.Module):
    """Category / subcategory pipeline: Embedding -> Linear -> activation."""

    def __init__(
        self,
        config: NAMLConfig,
        num_classes: int,
        embedding_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding = nnx.Embed(
            num_embeddings=num_classes + 1,
            features=embedding_dim,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            in_features=embedding_dim,
            out_features=config.cnn_filter_num,
            rngs=rngs,
        )

    def __call__(self, class_id: jax.Array, *, training: bool = False) -> jax.Array:
        """class_id: ``(N*,)`` ints → ``(N*, cnn_filter_num)`` vector."""
        embedded = self.embedding(class_id)
        return apply_activation(self.projection(embedded), self.config.activation)


class NewsEncoder(nnx.Module):
    """Multi-view news encoder. Outputs ``(*leading, cnn_filter_num)``."""

    def __init__(
        self,
        config: NAMLConfig,
        title_text_encoder: TextEncoder,
        abstract_text_encoder: TextEncoder | None,
        num_categories: int,
        num_subcategories: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_dim = config.cnn_filter_num

        self.title_view = _TextViewEncoder(config, title_text_encoder, rngs=rngs)
        # Optional abstract view — absent when process_abstract=False, in which
        # case the view-attention pools over title + category + subcategory only.
        self.abstract_view = (
            _TextViewEncoder(config, abstract_text_encoder, rngs=rngs)
            if abstract_text_encoder is not None
            else None
        )
        self.category_view = _StructuredViewEncoder(
            config, num_categories, config.category_embedding_dim, rngs=rngs
        )
        self.subcategory_view = _StructuredViewEncoder(
            config, num_subcategories, config.subcategory_embedding_dim, rngs=rngs
        )
        self.view_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.view_attention_query_dim,
            rngs=rngs,
        )

    @staticmethod
    def valid_mask(packed_features: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero."""
        return jnp.not_equal(packed_features[..., 0], 0)

    def __call__(
        self, packed_features: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Encode a batch of news articles.

        Args:
            packed_features: ``(*leading, 3)`` int with column order
                ``[news_idx | category_id | subcategory_id]``.
        """
        leading_shape = packed_features.shape[:-1]
        flat = packed_features.reshape(-1, 3)
        news_idx = flat[:, 0]
        category_id = flat[:, 1]
        subcategory_id = flat[:, 2]

        title_vec = self.title_view(news_idx, training=training)
        category_vec = self.category_view(category_id, training=training)
        subcategory_vec = self.subcategory_view(subcategory_id, training=training)

        # Stack views: (N*, num_views, cnn_filter_num). Abstract is included
        # only when present (process_abstract=True), giving 4 views; otherwise
        # 3. View-attention pools over whatever views are stacked.
        view_vecs = [title_vec]
        if self.abstract_view is not None:
            view_vecs.append(self.abstract_view(news_idx, training=training))
        view_vecs += [category_vec, subcategory_vec]

        views = jnp.stack(view_vecs, axis=1)
        pooled = self.view_attention(views)
        return pooled.reshape(*leading_shape, self.news_dim)
