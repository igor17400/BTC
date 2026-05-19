"""JAX/Flax NNX token-level frozen-PLM news encoders.

Mirror of :mod:`src.frameworks.pytorch.layers.plm_token`. Provides:

- :class:`PLMTokenNewsEncoder` — frozen token lookup + trainable
  :class:`Pooler` + Linear projection (NRMS-style).
- :class:`PLMTokenCNNEncoder` — frozen token lookup + Conv1d +
  AdditiveAttention (NAML / LSTUR title view).
- :class:`PLMTokenLookup` — drop-in nn.Embedding replacement that
  returns ``(B, T, D)`` features + ``(B, T)`` mask (PP-Rec, CAUM, ...).

The frozen caches are wrapped in :class:`FrozenCache` (a non-``Param``
:class:`nnx.Variable` subclass) so ``nnx.value_and_grad`` and the
optimizer skip them — no multi-GB zero-gradient buffer is allocated
for the cache during backward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .attention_layers import AdditiveAttention
from .layer_utils import apply_activation
from .plm_pooler import Pooler


class FrozenCache(nnx.Variable):
    """A non-trainable ``nnx.Variable`` for frozen PLM caches.

    The default Flax NNX gradient/optimizer filter is ``nnx.Param``;
    anything that's an ``nnx.Variable`` but NOT a Param is skipped. We
    use this to keep the 6–10 GB PLM token / abstract caches alive on
    the device while making ``value_and_grad`` ignore them — otherwise
    XLA allocates a same-size zero-gradient buffer on backward and the
    abstract cache alone OOMs the 40 GB GPU.
    """

    pass


class PLMTokenNewsEncoder(nnx.Module):
    """Frozen-PLM token-level news encoder + pooler + projection (NRMS).

    Args:
        plm_token_embeddings_by_id: ``(max_parsed_id + 1, T, plm_dim)``
            float32 — token-level frozen PLM outputs, indexed by parsed
            news id. Row 0 is the padding zero block.
        plm_attention_mask_by_id: ``(max_parsed_id + 1, T)`` int8 (0/1).
        news_dim: Output dim after the trainable projection.
        pooler_type / pooler_*: See :class:`Pooler`.
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
        *,
        news_dim: int,
        pooler_type: str = "attention",
        attention_query_dim: int = 200,
        num_heads: int = 6,
        head_dim: int = 128,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ):
        max_id, T, plm_dim = plm_token_embeddings_by_id.shape
        self.plm_dim = int(plm_dim)
        self.news_dim = int(news_dim)
        self.max_length = int(T)

        # Frozen lookup tables in bfloat16 — halves cache footprint
        # on the GPU. Cast back to fp32 at lookup so the pooler /
        # projection stay in fp32.
        self.token_embeddings = FrozenCache(
            jnp.asarray(plm_token_embeddings_by_id, dtype=jnp.bfloat16)
        )
        self.attention_mask = FrozenCache(
            jnp.asarray(plm_attention_mask_by_id, dtype=jnp.int8)
        )

        self.pooler = Pooler(
            plm_dim=plm_dim,
            pooler_type=pooler_type,
            attention_query_dim=attention_query_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.projection = nnx.Linear(plm_dim, news_dim, rngs=rngs)

    @staticmethod
    def valid_mask(features: jax.Array) -> jax.Array:
        """A PLM news slot is valid when its parsed id is non-zero."""
        return jnp.not_equal(features, 0)

    def __call__(
        self,
        features: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Look up token-level frozen vectors, pool, project.

        Args:
            features: int tensor of any shape ``(...,)`` of parsed news ids.

        Returns:
            ``(..., news_dim)`` float tensor.
        """
        leading_shape = features.shape
        ids = features.astype(jnp.int32).reshape(-1)
        tokens = jax.lax.stop_gradient(
            self.token_embeddings.value[ids].astype(jnp.float32)
        )
        mask = jax.lax.stop_gradient(self.attention_mask.value[ids])
        pooled = self.pooler(tokens, mask, training=training)
        out = self.projection(pooled)
        return out.reshape(*leading_shape, self.news_dim)


class PLMTokenLookup(nnx.Module):
    """Frozen token-level PLM lookup with optional projection (JAX/Flax NNX).

    Mirror of :class:`src.frameworks.pytorch.layers.plm_token.PLMTokenLookup`.
    Returns BOTH the per-token features and the attention mask — slots in
    as a replacement for a GloVe ``nnx.Embed`` layer in models that need
    token-level features (PP-Rec, CAUM, ...).
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
        *,
        plm_dim: int,
        output_dim: int | None = None,
        rngs: nnx.Rngs,
    ):
        max_id, T, in_dim = plm_token_embeddings_by_id.shape
        if in_dim != plm_dim:
            raise ValueError(
                f"plm_dim mismatch: cache has {in_dim}, config says {plm_dim}"
            )
        self.plm_dim = int(plm_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(plm_dim)
        self.max_length = int(T)

        # Caches stored as bfloat16 to halve GPU footprint vs fp32.
        # We up-cast to fp32 at the lookup site so downstream compute
        # (Linear / Conv1d / MHA) stays in fp32. The cache itself is
        # 3.2 GB (title) or 5 GB (abstract) instead of 6.4/10 GB.
        self.token_embeddings = FrozenCache(
            jnp.asarray(plm_token_embeddings_by_id, dtype=jnp.bfloat16)
        )
        self.attention_mask = FrozenCache(
            jnp.asarray(plm_attention_mask_by_id, dtype=jnp.int8)
        )

        if output_dim is not None and output_dim != plm_dim:
            self.projection: nnx.Module = nnx.Linear(plm_dim, output_dim, rngs=rngs)
        else:
            self.projection = _Identity()

    def __call__(
        self,
        news_idx: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        leading_shape = news_idx.shape
        ids = news_idx.astype(jnp.int32).reshape(-1)
        tokens = jax.lax.stop_gradient(
            self.token_embeddings.value[ids].astype(jnp.float32)
        )
        mask = jax.lax.stop_gradient(self.attention_mask.value[ids])
        out = self.projection(tokens)
        return (
            out.reshape(*leading_shape, self.max_length, self.output_dim),
            mask.reshape(*leading_shape, self.max_length),
        )


class _Identity(nnx.Module):
    """Pass-through used when no projection is needed."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class PLMTokenCNNEncoder(nnx.Module):
    """NAML/LSTUR-style token-level PLM news encoder.

    Frozen token lookup + Conv1d + AdditiveAttention (NAML / LSTUR's
    original per-token modelling pipeline, with BERT tokens replacing
    GloVe tokens).
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
        *,
        news_dim: int,
        plm_dim: int,
        cnn_kernel_size: int = 3,
        dropout_rate: float = 0.2,
        word_attention_query_dim: int = 200,
        activation: str = "relu",
        rngs: nnx.Rngs,
    ):
        max_id, T, in_dim = plm_token_embeddings_by_id.shape
        if in_dim != plm_dim:
            raise ValueError(
                f"plm_dim mismatch: cache has {in_dim}, config says {plm_dim}"
            )
        self.plm_dim = int(plm_dim)
        self.news_dim = int(news_dim)
        self.max_length = int(T)
        self.activation = activation

        # bfloat16 cache — halves the abstract-view 10 GB → 5 GB.
        self.token_embeddings = FrozenCache(
            jnp.asarray(plm_token_embeddings_by_id, dtype=jnp.bfloat16)
        )
        self.attention_mask = FrozenCache(
            jnp.asarray(plm_attention_mask_by_id, dtype=jnp.int8)
        )

        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        # NNX Conv expects (B, T, in_features) → (B, T, out_features).
        self.cnn = nnx.Conv(
            in_features=plm_dim,
            out_features=news_dim,
            kernel_size=(cnn_kernel_size,),
            padding="SAME",
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=news_dim,
            query_vec_dim=word_attention_query_dim,
            rngs=rngs,
        )

    @staticmethod
    def valid_mask(features: jax.Array) -> jax.Array:
        """A PLM news slot is valid when its parsed id is non-zero."""
        return jnp.not_equal(features, 0)

    def __call__(
        self,
        features: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Look up token-level frozen vectors and run a CNN + AddAttn pool.

        Args:
            features: int tensor of any shape ``(...,)`` of parsed news ids.

        Returns:
            ``(..., news_dim)`` float tensor.
        """
        leading_shape = features.shape
        ids = features.astype(jnp.int32).reshape(-1)
        tokens = jax.lax.stop_gradient(
            self.token_embeddings.value[ids].astype(jnp.float32)
        )
        mask = jax.lax.stop_gradient(self.attention_mask.value[ids]).astype(jnp.bool_)

        y = self.dropout1(tokens, deterministic=not training)
        y = apply_activation(self.cnn(y), self.activation)
        y = self.dropout2(y, deterministic=not training)

        out = self.additive_attention(y, mask=mask)  # (N*, news_dim)
        return out.reshape(*leading_shape, self.news_dim)
