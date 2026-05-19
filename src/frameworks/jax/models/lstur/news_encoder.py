"""LSTUR news encoder (Flax NNX) — encoder-agnostic.

Mirror of the PyTorch LSTUR news encoder. CNN-over-tokens title pipeline
+ optional category/subcategory embeddings. The two text-pipeline modes
(GloVe or PLM) flow through the same :class:`TextEncoder` interface; the
CNN handles the projection to ``cnn_filter_num``.

Pipeline (single code path for GloVe and PLM):
    text_encoder(news_idx) -> (tokens, mask)          # tokens @ text_dim
        -> Dropout -> Conv(text_dim -> cnn_filter_num)
        -> activation -> Dropout
        -> AdditiveAttention with mask                # pool @ cnn_filter_num
    cat_embedding(category_id)                        # @ cat_dim   (optional)
    subcat_embedding(subcategory_id)                  # @ subcat_dim (optional)
    -> concat -> news vector @ output_dim
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import LSTURConfig

from ...layers import AdditiveAttention, TextEncoder, apply_activation


class _CategoryEmbedding(nnx.Module):
    """LSTUR-style structured-view embedding (no projection)."""

    def __init__(self, num_classes: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(
            num_embeddings=num_classes + 1,
            features=embedding_dim,
            rngs=rngs,
        )

    def __call__(self, class_id: jax.Array) -> jax.Array:
        """``class_id``: ``(N*,)`` ints → ``(N*, embedding_dim)`` vector."""
        return self.embedding(class_id)


class NewsEncoder(nnx.Module):
    """LSTUR news encoder. Outputs ``(*leading, output_dim)``.

    Args:
        config: LSTUR hyperparameters.
        text_encoder: :class:`TextEncoder` for the title cache (GloVe or PLM).
        num_categories: Category vocab size (excludes padding index 0).
            Ignored when ``config.use_category`` is False.
        num_subcategories: Subcategory vocab size (excludes padding index 0).
            Ignored when ``config.use_subcategory`` is False.
        rngs: ``nnx.Rngs`` for module initialisation.
    """

    def __init__(
        self,
        config: LSTURConfig,
        text_encoder: TextEncoder,
        num_categories: int = 0,
        num_subcategories: int = 0,
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
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

        self.category_embedding = (
            _CategoryEmbedding(num_categories, config.category_embedding_dim, rngs=rngs)
            if config.use_category
            else None
        )
        self.subcategory_embedding = (
            _CategoryEmbedding(
                num_subcategories, config.subcategory_embedding_dim, rngs=rngs
            )
            if config.use_subcategory
            else None
        )

        self._n_cols = 1 + int(config.use_category) + int(config.use_subcategory)

    @property
    def output_dim(self) -> int:
        """Final per-news vector dim — used to size the user encoder GRU."""
        dim = self.config.cnn_filter_num
        if self.category_embedding is not None:
            dim += self.config.category_embedding_dim
        if self.subcategory_embedding is not None:
            dim += self.config.subcategory_embedding_dim
        return dim

    @staticmethod
    def valid_mask(features: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero.

        Accepts both packed ``(..., k)`` and plain ``(...,)`` int tensors.
        """
        if features.ndim >= 2 and features.shape[-1] <= 4:
            return jnp.not_equal(features[..., 0], 0)
        return jnp.not_equal(features, 0)

    def _split_columns(
        self, inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        """Return ``(news_idx, category_id, subcategory_id)`` from inputs."""
        if self._n_cols == 1:
            news_idx = (
                inputs if inputs.shape[-1] != 1 or inputs.ndim == 1 else inputs[..., 0]
            )
            return news_idx, None, None

        news_idx = inputs[..., 0]
        col = 1
        category_id: jax.Array | None = None
        subcategory_id: jax.Array | None = None
        if self.category_embedding is not None:
            category_id = inputs[..., col]
            col += 1
        if self.subcategory_embedding is not None:
            subcategory_id = inputs[..., col]
        return news_idx, category_id, subcategory_id

    def _encode_title(self, news_idx: jax.Array, *, training: bool) -> jax.Array:
        """Encode title tokens through TextEncoder + CNN + additive pool.

        Args:
            news_idx: ``(N*,)`` int tensor of parsed news ids.

        Returns:
            ``(N*, cnn_filter_num)`` title vectors.
        """
        tokens, mask = self.text_encoder(news_idx)  # (N*, T, text_dim), (N*, T)
        y = self.dropout1(tokens, deterministic=not training)
        y = apply_activation(self.cnn(y), self.config.cnn_activation)
        y = self.dropout2(y, deterministic=not training)

        valid = jnp.not_equal(mask, 0)
        return self.additive_attention(y, mask=valid)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Encode a batch of news articles.

        Args:
            inputs: Packed ``(..., k)`` int with column order
                ``[news_idx | category_id | subcategory_id]`` (trailing
                columns drop when their flag is False), or plain
                ``(...,)`` int when no structured views are enabled.
            training: Controls dropout.

        Returns:
            ``(..., output_dim)`` news vectors.
        """
        news_idx, category_id, subcategory_id = self._split_columns(inputs)
        leading_shape = news_idx.shape

        flat_news_idx = news_idx.reshape(-1)
        title_vec = self._encode_title(flat_news_idx, training=training)

        parts: list[jax.Array] = [title_vec]
        if self.category_embedding is not None:
            assert category_id is not None
            parts.append(self.category_embedding(category_id.reshape(-1)))
        if self.subcategory_embedding is not None:
            assert subcategory_id is not None
            parts.append(self.subcategory_embedding(subcategory_id.reshape(-1)))

        news_vec = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else title_vec
        return news_vec.reshape(*leading_shape, self.output_dim)
