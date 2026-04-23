"""DIGAT-specific layers for JAX/Flax NNX: scatter utilities and attention modules.

Parallels ``src.frameworks.pytorch.models.digat.layers`` one-for-one so
both framework implementations are structurally identical.  Scatter
operations are implemented with pure JAX primitives (``jnp.max`` +
masked reductions) so there is no dependency on ``jax_scatter`` or
similar.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


# ======================================================================
# Container utilities
# ======================================================================


class ModuleStack(nnx.Module):
    """Index-accessed container for a list of ``nnx.Module`` instances.

    Equivalent to ``nnx.List`` (introduced in later Flax versions) but
    works with Flax 0.10.x.  Each member is stored as a distinct
    attribute so NNX registers it as a child module.
    """

    def __init__(self, modules: list[nnx.Module]):
        for i, m in enumerate(modules):
            setattr(self, f"_m{i}", m)
        self._len = len(modules)

    def __getitem__(self, index: int) -> nnx.Module:
        return getattr(self, f"_m{index}")

    def __len__(self) -> int:
        return self._len


# ======================================================================
# Scatter utilities (batched, pure JAX)
# ======================================================================


def scatter_softmax(
    src: jax.Array, index: jax.Array, num_groups: int
) -> jax.Array:
    """Per-group softmax without ``torch_scatter``.

    Args:
        src: ``(B, N)`` scores.
        index: ``(B, N)`` int group assignments in ``[0, num_groups)``.
        num_groups: Total number of groups.

    Returns:
        ``(B, N)`` softmax-normalised within each group.
    """
    batch_size, num_items = src.shape
    idx = index.astype(jnp.int32)

    # Max per group (for numerical stability): use a one-hot mask to
    # restrict each group's max computation to the items that belong
    # to it.
    group_range = jnp.arange(num_groups)[None, None, :]          # (1, 1, G)
    mask = idx[:, :, None] == group_range                         # (B, N, G)
    neg_inf = jnp.finfo(src.dtype).min
    masked = jnp.where(mask, src[:, :, None], neg_inf)            # (B, N, G)
    group_max = jnp.max(masked, axis=1)                            # (B, G)
    element_max = jnp.take_along_axis(group_max, idx, axis=1)     # (B, N)

    exp_src = jnp.exp(src - element_max)

    # Sum per group via ``jnp.add.at`` equivalent using one-hot scatter.
    group_sum = jnp.sum(
        jnp.where(mask, exp_src[:, :, None], 0.0), axis=1
    )                                                              # (B, G)
    element_sum = jnp.take_along_axis(group_sum, idx, axis=1)     # (B, N)

    return exp_src / (element_sum + 1e-10)


def scatter_sum(
    src: jax.Array, index: jax.Array, dim: int, dim_size: int
) -> jax.Array:
    """Grouped sum without ``torch_scatter``.

    Args:
        src: ``(B, N, D)`` values.
        index: ``(B, N)`` int group assignments in ``[0, dim_size)``.
        dim: Dimension to scatter along (must be 1).
        dim_size: Output size along the scatter dimension.

    Returns:
        ``(B, dim_size, D)`` summed values per group.
    """
    if dim != 1:
        raise ValueError("scatter_sum only supports dim=1.")

    idx = index.astype(jnp.int32)
    # One-hot: (B, N, dim_size)
    one_hot = jax.nn.one_hot(idx, dim_size, dtype=src.dtype)
    # (B, N, dim_size, 1) * (B, N, 1, D) → sum over N → (B, dim_size, D)
    return jnp.einsum("bnk,bnd->bkd", one_hot, src)


# ======================================================================
# Attention layers
# ======================================================================


class ScaledDotProductAttention(nnx.Module):
    """Scaled dot-product attention: K/Q projections → softmax → pool."""

    def __init__(
        self,
        feature_dim: int,
        query_dim: int,
        attention_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.K = nnx.Linear(feature_dim, attention_dim, use_bias=False, rngs=rngs)
        self.Q = nnx.Linear(query_dim, attention_dim, use_bias=True, rngs=rngs)
        self.scale = math.sqrt(float(attention_dim))

    def __call__(
        self,
        features: jax.Array,
        query: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        # features: (B, N, F), query: (B, Q)
        keys = self.K(features)                             # (B, N, A)
        q = self.Q(query)[:, :, None]                       # (B, A, 1)
        scores = jnp.squeeze(jnp.matmul(keys, q), axis=-1) / self.scale  # (B, N)
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        weights = jax.nn.softmax(scores, axis=1)[:, None, :]  # (B, 1, N)
        return jnp.squeeze(jnp.matmul(weights, features), axis=1)  # (B, F)


class AdditiveAttention(nnx.Module):
    """Additive (Bahdanau) attention for sequence pooling."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.affine = nnx.Linear(feature_dim, attention_dim, use_bias=True, rngs=rngs)
        self.project = nnx.Linear(attention_dim, 1, use_bias=False, rngs=rngs)

    def __call__(
        self, features: jax.Array, mask: jax.Array | None = None
    ) -> jax.Array:
        # features: (B, N, F)
        scores = jnp.squeeze(self.project(jnp.tanh(self.affine(features))), axis=-1)  # (B, N)
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        weights = jax.nn.softmax(scores, axis=1)[:, None, :]           # (B, 1, N)
        return jnp.squeeze(jnp.matmul(weights, features), axis=1)      # (B, F)


class MultiHeadSelfAttention(nnx.Module):
    """Multi-head self-attention (news encoder MSA).

    Custom implementation (not ``nnx.MultiHeadAttention``) to match the
    PyTorch DIGAT reference exactly: a bias-free K projection and a
    biased Q/V projection, plus manual head reshape/transpose.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(float(head_dim))
        self.W_Q = nnx.Linear(d_model, self.out_dim, use_bias=True, rngs=rngs)
        self.W_K = nnx.Linear(d_model, self.out_dim, use_bias=False, rngs=rngs)
        self.W_V = nnx.Linear(d_model, self.out_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        # (B, L, H*D) → (B, L, H, D) → (B, H, L, D)
        Q = self.W_Q(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        attn = jax.nn.softmax(
            jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale, axis=-1
        )
        out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, self.out_dim)
        return out
