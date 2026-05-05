"""GLORY-specific layers for Keras 3: attention primitives + GatedGraphConv.

Parallels ``src.frameworks.pytorch.models.glory.layers`` one-for-one so
all framework implementations are structurally identical.  Graph scatter
operations use pure ``keras.ops`` primitives.

References:
- Yang et al., "Going Beyond Local: Global Graph-Enhanced Personalized
  News Recommendations", RecSys 2023.
- Li et al., "Gated Graph Sequence Neural Networks" (GGNN), ICLR 2016.
"""

from __future__ import annotations

import math

import keras
import numpy as np
from keras import layers, ops

# ======================================================================
# Attention primitives (mirror GLORY's ``src/models/base/layers.py``)
# ======================================================================


class ScaledDotProductAttention(keras.layers.Layer):
    """Scaled dot-product attention used inside ``MultiHeadAttention``.

    Note: GLORY's reference uses a non-standard masking scheme -- scores
    are exponentiated before masking, then re-normalised.  We replicate
    that behaviour exactly to match the paper's initialization and
    training dynamics.
    """

    def __init__(self, d_k: int, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k

    def call(self, Q, K, V, attn_mask=None):
        scores = ops.matmul(Q, ops.swapaxes(K, -1, -2)) / math.sqrt(self.d_k)
        # Numerically stable exp: subtract max to prevent overflow.
        # Mathematically equivalent to raw exp() after normalization.
        scores = scores - ops.max(scores, axis=-1, keepdims=True)
        scores = ops.exp(scores)
        if attn_mask is not None:
            scores = scores * ops.expand_dims(attn_mask, axis=-2)
        attn = scores / (ops.sum(scores, axis=-1, keepdims=True) + 1e-8)
        return ops.matmul(attn, V)


class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention matching GLORY's Q/K/V projection choice."""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        head_num: int,
        head_dim: int,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_num = head_num
        self.head_dim = head_dim
        self.residual = residual
        out_dim = head_num * head_dim

        self.W_Q = layers.Dense(
            out_dim,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
        self.W_K = layers.Dense(
            out_dim,
            use_bias=False,
            kernel_initializer="glorot_uniform",
        )
        self.W_V = layers.Dense(
            out_dim,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
        self.attn = ScaledDotProductAttention(head_dim)

    def call(self, Q, K=None, V=None, mask=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = ops.shape(Q)[0]
        H, D = self.head_num, self.head_dim

        if mask is not None:
            mask = ops.broadcast_to(
                ops.expand_dims(mask, axis=1),
                (batch_size, H, ops.shape(mask)[-1]),
            )

        q = ops.transpose(
            ops.reshape(self.W_Q(Q), (batch_size, -1, H, D)),
            (0, 2, 1, 3),
        )
        k = ops.transpose(
            ops.reshape(self.W_K(K), (batch_size, -1, H, D)),
            (0, 2, 1, 3),
        )
        v = ops.transpose(
            ops.reshape(self.W_V(V), (batch_size, -1, H, D)),
            (0, 2, 1, 3),
        )

        ctx = self.attn(q, k, v, mask)
        out = ops.reshape(
            ops.transpose(ctx, (0, 2, 1, 3)),
            (batch_size, -1, H * D),
        )
        return out + Q if self.residual else out


class AttentionPooling(keras.layers.Layer):
    """Additive attention pooling over a sequence of vectors.

    Follows GLORY's implementation exactly:
    ``alpha = softmax(v^T tanh(W x)) / (sum + eps)`` with multiplicative
    masking (consistent with ``ScaledDotProductAttention``).
    """

    def __init__(self, emb_size: int, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.att_fc1 = layers.Dense(
            hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.GlorotUniform(),
            bias_initializer="zeros",
        )
        self.att_fc2 = layers.Dense(
            1,
            use_bias=True,
            kernel_initializer=keras.initializers.GlorotUniform(),
        )

    def call(self, x, attn_mask=None):
        # x: (B, N, E)
        e = ops.tanh(self.att_fc1(x))
        raw = self.att_fc2(e)  # (B, N, 1)
        # Numerically stable exp: subtract max to prevent overflow.
        alpha = ops.exp(raw - ops.max(raw, axis=1, keepdims=True))
        if attn_mask is not None:
            alpha = alpha * ops.expand_dims(attn_mask, axis=2)
        alpha = alpha / (ops.sum(alpha, axis=1, keepdims=True) + 1e-8)
        # (B, E, N) @ (B, N, 1) -> (B, E, 1) -> (B, E)
        return ops.squeeze(ops.matmul(ops.swapaxes(x, 1, 2), alpha), axis=-1)


# ======================================================================
# DotProduct scorer
# ======================================================================


class DotProduct(keras.layers.Layer):
    """Batched dot product for candidate scoring."""

    def call(self, left, right):
        # left: (B, C, D), right: (B, D)  ->  (B, C)
        return ops.squeeze(ops.matmul(left, ops.expand_dims(right, axis=-1)), axis=-1)


# ======================================================================
# GatedGraphConv (pure Keras ops, no torch_geometric)
# ======================================================================


def _scatter_add(src, index, dim_size):
    """Segment sum along dim 0: ``out[index[i]] += src[i]``.

    Args:
        src: ``(E, D)`` source values (edge features).
        index: ``(E,)`` int target indices in ``[0, dim_size)``.
        dim_size: Output size along dim 0.

    Returns:
        ``(dim_size, D)`` aggregated output.
    """
    # Use backend-specific scatter to avoid O(E * dim_size) one_hot.
    backend = keras.backend.backend()
    if backend == "torch":
        import torch

        index_t = index.to(device=src.device)
        out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        idx = index_t.unsqueeze(-1).expand_as(src)
        return out.scatter_add_(0, idx.long(), src)
    elif backend == "jax":
        import jax.numpy as jnp

        return jnp.zeros((dim_size, src.shape[1]), dtype=src.dtype).at[index].add(src)
    else:
        # Fallback: one_hot approach (only safe for small graphs).
        one_hot = ops.one_hot(index, dim_size)
        return ops.matmul(ops.transpose(one_hot), src)


class GRUCell(keras.layers.Layer):
    """GRU cell matching PyTorch's ``nn.GRUCell`` behaviour.

    Keras 3 does not provide a standalone GRUCell suitable for
    non-sequential use, so we implement one with Dense layers +
    sigmoid/tanh gates, following the JAX implementation.
    """

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        # Match PyTorch nn.GRUCell: uniform(-1/sqrt(H), 1/sqrt(H)) for
        # both kernels and biases.
        k = 1.0 / math.sqrt(hidden_size)
        uniform_init = keras.initializers.RandomUniform(-k, k)
        # Input-to-hidden weights for r, z, n gates (stacked).
        self.W_ih = layers.Dense(
            3 * hidden_size,
            use_bias=True,
            kernel_initializer=uniform_init,
            bias_initializer=uniform_init,
        )
        # Hidden-to-hidden weights for r, z, n gates (stacked).
        self.W_hh = layers.Dense(
            3 * hidden_size,
            use_bias=True,
            kernel_initializer=uniform_init,
            bias_initializer=uniform_init,
        )

    def call(self, x, h):
        """Forward: x (N, input_size), h (N, hidden_size) -> h' (N, hidden_size)."""
        H = self.hidden_size
        gi = self.W_ih(x)  # (N, 3H)
        gh = self.W_hh(h)  # (N, 3H)

        i_r = gi[:, :H]
        i_z = gi[:, H : 2 * H]
        i_n = gi[:, 2 * H :]
        h_r = gh[:, :H]
        h_z = gh[:, H : 2 * H]
        h_n = gh[:, 2 * H :]

        r = ops.sigmoid(i_r + h_r)
        z = ops.sigmoid(i_z + h_z)
        n = ops.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h


class GatedGraphConv(keras.layers.Layer):
    """Gated Graph Neural Network (Li et al. 2016).

    Pure-Keras replacement for ``torch_geometric.nn.GatedGraphConv``.
    Shared weight ``W`` across all ``num_layers`` propagation steps,
    with a single ``GRUCell`` mixing messages into node states (matches
    GLORY's original configuration which uses the PyG default).

    Implementation:
        for t in range(num_layers):
            m_i = sum_{j in N(i)} W . h_j        # weighted neighbor sum
            h_i = GRU(m_i, h_i)                  # gated update
    """

    def __init__(
        self,
        out_channels: int,
        num_layers: int = 3,
        aggr: str = "add",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if aggr != "add":
            raise ValueError(
                f"Only aggr='add' supported; got {aggr!r}. "
                f"GLORY's reference uses add aggregation."
            )
        self.out_channels = out_channels
        self.num_layers = num_layers

    def build(self, input_shape):
        # Initialize each layer's weight slice independently so fan_in/fan_out
        # are computed on the 2D (out_channels, out_channels) shape, matching
        # PyTorch's per-slice xavier_uniform_ and the JAX port.
        # Use keras.ops.convert_to_numpy to handle both JAX and torch backends
        # (torch initializers may return CUDA tensors that raw np.array rejects).
        initializer = keras.initializers.GlorotUniform()
        slices = [
            keras.ops.convert_to_numpy(
                initializer(shape=(self.out_channels, self.out_channels))
            )
            for _ in range(self.num_layers)
        ]
        stacked = np.stack(slices, axis=0)
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_layers, self.out_channels, self.out_channels),
            initializer=lambda shape, dtype=None: stacked,
            trainable=True,
        )
        self.rnn = GRUCell(self.out_channels, self.out_channels, name="gru_cell")
        super().build(input_shape)

    def call(self, x, edge_index, real_mask=None):
        """Propagate features across a graph.

        Args:
            x: ``(N, F_in)`` node features.  Will be zero-padded to
                ``(N, out_channels)`` if ``F_in < out_channels``.
            edge_index: ``(2, E)`` int tensor of directed edges
                ``(src, dst)`` -- messages flow src -> dst.
            real_mask: optional ``(N, 1)`` bool mask — True for real
                nodes, False for padding.  When provided, padding node
                states are zeroed after each GRU step to prevent the
                GRU's learned biases from leaking non-zero values into
                padding nodes, which would be amplified ~718k× through
                self-loop scatter-add.

        Returns:
            ``(N, out_channels)`` updated node features.
        """
        if ops.shape(x)[1] < self.out_channels:
            pad_width = self.out_channels - ops.shape(x)[1]
            x = ops.pad(x, ((0, 0), (0, pad_width)))

        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = ops.shape(x)[0]

        for i in range(self.num_layers):
            m = ops.matmul(x, self.weight[i])  # (N, D)
            agg = _scatter_add(m[src], dst, num_nodes)  # (N, D)
            x = self.rnn(agg, x)  # (N, D)
            if real_mask is not None:
                x = ops.where(real_mask, x, 0)

        return x
