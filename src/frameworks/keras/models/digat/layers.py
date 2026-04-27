"""DIGAT-specific layers for Keras 3: scatter utilities and attention modules.

Parallels ``src.frameworks.pytorch.models.digat.layers`` one-for-one so
both framework implementations are structurally identical.  Scatter
operations are implemented with pure ``keras.ops`` primitives (one_hot,
einsum, etc.) so the code is backend-agnostic.
"""

from __future__ import annotations

import math

import keras
from keras import layers, ops


# ======================================================================
# Scatter utilities (batched, pure keras.ops)
# ======================================================================


def scatter_softmax(src, index, num_groups):
    """Per-group softmax without torch_scatter.

    Args:
        src: ``(B, N)`` scores.
        index: ``(B, N)`` int group assignments in ``[0, num_groups)``.
        num_groups: Total number of groups.

    Returns:
        ``(B, N)`` softmax-normalised within each group.
    """
    idx = ops.cast(index, "int32")

    # Max per group (for numerical stability): use a one-hot mask to
    # restrict each group's max computation to the items that belong to it.
    group_range = ops.arange(num_groups)  # (G,)
    group_range = ops.reshape(group_range, (1, 1, num_groups))  # (1, 1, G)
    mask = ops.equal(idx[:, :, None], group_range)  # (B, N, G)  bool

    neg_inf = -1e9
    masked = ops.where(mask, src[:, :, None], neg_inf)  # (B, N, G)
    group_max = ops.max(masked, axis=1)  # (B, G)
    element_max = ops.take_along_axis(group_max, idx, axis=1)  # (B, N)

    exp_src = ops.exp(src - element_max)

    # Sum per group via one-hot scatter.
    mask_f = ops.cast(mask, exp_src.dtype)  # (B, N, G)
    group_sum = ops.sum(mask_f * exp_src[:, :, None], axis=1)  # (B, G)
    element_sum = ops.take_along_axis(group_sum, idx, axis=1)  # (B, N)

    return exp_src / (element_sum + 1e-10)


def scatter_sum(src, index, dim, dim_size):
    """Grouped sum without torch_scatter.

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

    idx = ops.cast(index, "int32")
    # One-hot: (B, N, dim_size)
    one_hot = ops.one_hot(idx, dim_size, dtype=src.dtype)
    # (B, N, dim_size) x (B, N, D) -> (B, dim_size, D) via einsum
    return ops.einsum("bnk,bnd->bkd", one_hot, src)


# ======================================================================
# Attention layers
# ======================================================================


class ScaledDotProductAttention(layers.Layer):
    """Scaled dot-product attention: K/Q projections -> softmax -> pool."""

    def __init__(self, feature_dim, query_dim, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.query_dim = query_dim
        self.attention_dim = attention_dim
        self.scale = math.sqrt(float(attention_dim))

        self.K = layers.Dense(attention_dim, use_bias=False, name="K")
        self.Q = layers.Dense(attention_dim, use_bias=True, name="Q")

    def call(self, features, query, mask=None):
        """
        Args:
            features: (B, N, F)
            query: (B, Q)
            mask: (B, N) optional

        Returns:
            (B, F)
        """
        keys = self.K(features)  # (B, N, A)
        q = ops.expand_dims(self.Q(query), axis=2)  # (B, A, 1)
        scores = ops.squeeze(ops.matmul(keys, q), axis=-1) / self.scale  # (B, N)
        if mask is not None:
            scores = ops.where(ops.equal(mask, 0), -1e9, scores)
        weights = ops.softmax(scores, axis=1)  # (B, N)
        weights = ops.expand_dims(weights, axis=1)  # (B, 1, N)
        return ops.squeeze(ops.matmul(weights, features), axis=1)  # (B, F)


class DIGATAdditiveAttention(layers.Layer):
    """Additive (Bahdanau) attention for sequence pooling.

    Named ``DIGATAdditiveAttention`` to avoid collision with the
    framework-level ``AdditiveAttention`` in ``src.frameworks.keras.layers``.
    """

    def __init__(self, feature_dim, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim

        self.affine = layers.Dense(attention_dim, use_bias=True, name="affine")
        self.project = layers.Dense(1, use_bias=False, name="project")

    def call(self, features, mask=None):
        """
        Args:
            features: (B, N, F)
            mask: (B, N) optional

        Returns:
            (B, F)
        """
        scores = ops.squeeze(self.project(ops.tanh(self.affine(features))), axis=-1)  # (B, N)
        if mask is not None:
            scores = ops.where(ops.equal(mask, 0), -1e9, scores)
        weights = ops.softmax(scores, axis=1)  # (B, N)
        weights = ops.expand_dims(weights, axis=1)  # (B, 1, N)
        return ops.squeeze(ops.matmul(weights, features), axis=1)  # (B, F)


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention (news encoder MSA).

    Custom implementation (not ``layers.MultiHeadAttention``) to match the
    PyTorch DIGAT reference exactly: a bias-free K projection and biased
    Q/V projections, plus manual head reshape/transpose.
    """

    def __init__(self, d_model, num_heads, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(float(head_dim))

        self.W_Q = layers.Dense(self.out_dim, use_bias=True, name="W_Q")
        self.W_K = layers.Dense(self.out_dim, use_bias=False, name="W_K")
        self.W_V = layers.Dense(self.out_dim, use_bias=True, name="W_V")

    def call(self, x):
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, out_dim)
        """
        B = ops.shape(x)[0]
        L = ops.shape(x)[1]
        H, D = self.num_heads, self.head_dim

        # (B, L, H*D) -> (B, L, H, D) -> (B, H, L, D)
        Q = ops.transpose(ops.reshape(self.W_Q(x), (B, L, H, D)), (0, 2, 1, 3))
        K = ops.transpose(ops.reshape(self.W_K(x), (B, L, H, D)), (0, 2, 1, 3))
        V = ops.transpose(ops.reshape(self.W_V(x), (B, L, H, D)), (0, 2, 1, 3))

        # (B, H, L, D) x (B, H, D, L) -> (B, H, L, L)
        attn = ops.softmax(
            ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) / self.scale,
            axis=-1,
        )
        # (B, H, L, L) x (B, H, L, D) -> (B, H, L, D) -> (B, L, H, D) -> (B, L, H*D)
        out = ops.reshape(
            ops.transpose(ops.matmul(attn, V), (0, 2, 1, 3)),
            (B, L, self.out_dim),
        )
        return out
