"""Unified text-encoder interface for news models (Flax NNX).

Mirror of :mod:`src.frameworks.pytorch.layers.text_encoder`. Same
contract: ``tokens, mask = text_encoder(news_idx)`` with
``tokens.shape == (..., max_length, output_dim)`` and
``mask.shape == (..., max_length)`` int8.

The frozen caches (GloVe word-id table and PLM token cache) are stored
as :class:`FrozenCache` so ``nnx.value_and_grad`` and the optimizer
ignore them — no zero-gradient buffer is allocated on backward, which
otherwise OOMs the GPU on multi-GB PLM caches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .plm_token import FrozenCache


class TextEncoder(nnx.Module):
    """Protocol class for the model-facing text encoder contract.

    Subclasses MUST:
      - Set ``self.output_dim: int`` and ``self.max_length: int``.
      - Implement ``__call__(news_idx) -> (tokens, mask)`` where
        ``tokens`` is ``(..., max_length, output_dim)`` float and
        ``mask`` is ``(..., max_length)`` int8 (1 = valid token).
    """

    output_dim: int
    max_length: int

    def __call__(  # pragma: no cover - protocol
        self, news_idx: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError


class GloveTextEncoder(TextEncoder):
    """GloVe word lookup keyed by news_idx (Flax NNX).

    Holds the precomputed per-news word-id matrix as a frozen
    :class:`FrozenCache` plus a trainable ``nnx.Embed(vocab_size,
    embedding_dim)`` initialised from pretrained GloVe vectors.

    Args:
        word_id_matrix: ``(num_news, T)`` int — per-news word IDs.
        vocab_size: Word vocabulary size.
        embedding_dim: Output channel dim (= ``output_dim``).
        pretrained_embeddings: Optional ``(vocab_size, embedding_dim)``
            float matrix to initialise the embedding weights.
        rngs: ``nnx.Rngs`` for module initialisation.
    """

    def __init__(
        self,
        word_id_matrix: np.ndarray,
        *,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings: np.ndarray | None = None,
        rngs: nnx.Rngs,
    ):
        word_id_matrix = np.asarray(word_id_matrix)
        if word_id_matrix.ndim != 2:
            raise ValueError(
                f"word_id_matrix must be 2D (num_news, T); got shape "
                f"{word_id_matrix.shape}"
            )
        _, T = word_id_matrix.shape
        self.output_dim = int(embedding_dim)
        self.max_length = int(T)

        # Frozen per-news word-id table (int32) — int storage keeps the
        # cache tiny; cast to int32 at lookup is free.
        self.word_ids = FrozenCache(jnp.asarray(word_id_matrix, dtype=jnp.int32))

        self.embedding = nnx.Embed(
            num_embeddings=int(vocab_size),
            features=int(embedding_dim),
            rngs=rngs,
        )
        if pretrained_embeddings is not None:
            pretrained = np.asarray(pretrained_embeddings)
            if pretrained.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    f"pretrained_embeddings shape {pretrained.shape} != "
                    f"(vocab_size={vocab_size}, embedding_dim={embedding_dim})"
                )
            self.embedding.embedding.value = jnp.asarray(pretrained, dtype=jnp.float32)

    @staticmethod
    def valid_mask(news_idx: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed id is non-zero."""
        return jnp.not_equal(news_idx, 0)

    def __call__(self, news_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        leading_shape = news_idx.shape
        flat_idx = news_idx.astype(jnp.int32).reshape(-1)
        ids = jax.lax.stop_gradient(self.word_ids.value[flat_idx])  # (N*, T)
        tokens = self.embedding(ids)  # (N*, T, D)
        mask = jnp.not_equal(ids, 0).astype(jnp.int8)  # (N*, T)
        return (
            tokens.reshape(*leading_shape, self.max_length, self.output_dim),
            mask.reshape(*leading_shape, self.max_length),
        )


class PLMTextEncoder(TextEncoder):
    """Frozen-cached PLM token lookup keyed by news_idx (Flax NNX).

    Returns tokens at the **native** PLM hidden size (e.g. 768 for
    BERT-base). The model's per-news pipeline runs at this dim and is
    responsible for projecting down to its target ``news_dim`` at the
    end of pool when needed — IP2-style.

    Caches are stored as :class:`FrozenCache` in bfloat16 to halve GPU
    footprint; lookup up-casts to fp32 so downstream compute stays in
    fp32.

    Args:
        plm_token_embeddings_by_id: ``(max_id+1, T, plm_dim)`` float.
        plm_attention_mask_by_id: ``(max_id+1, T)`` int (0/1).
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
    ):
        cache = np.asarray(plm_token_embeddings_by_id)
        if cache.ndim != 3:
            raise ValueError(
                f"plm_token_embeddings_by_id must be 3D (num_news, T, plm_dim); "
                f"got shape {cache.shape}"
            )
        _, T, plm_dim = cache.shape
        self.plm_dim = int(plm_dim)
        self.output_dim = int(plm_dim)
        self.max_length = int(T)

        self.token_embeddings = FrozenCache(jnp.asarray(cache, dtype=jnp.bfloat16))
        self.attention_mask = FrozenCache(
            jnp.asarray(np.asarray(plm_attention_mask_by_id), dtype=jnp.int8)
        )

    @staticmethod
    def valid_mask(news_idx: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed id is non-zero."""
        return jnp.not_equal(news_idx, 0)

    def __call__(self, news_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        leading_shape = news_idx.shape
        ids = news_idx.astype(jnp.int32).reshape(-1)
        tokens = jax.lax.stop_gradient(
            self.token_embeddings.value[ids].astype(jnp.float32)
        )
        mask = jax.lax.stop_gradient(self.attention_mask.value[ids])
        return (
            tokens.reshape(*leading_shape, self.max_length, self.output_dim),
            mask.reshape(*leading_shape, self.max_length),
        )
