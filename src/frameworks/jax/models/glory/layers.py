"""GLORY-specific layers for JAX/Flax NNX: attention primitives + GatedGraphConv.

Parallels ``src.frameworks.pytorch.models.glory.layers`` one-for-one so
both framework implementations are structurally identical.  Graph scatter
operations use pure JAX primitives (``segment_sum``).

References:
- Yang et al., "Going Beyond Local: Global Graph-Enhanced Personalized
  News Recommendations", RecSys 2023.
- Li et al., "Gated Graph Sequence Neural Networks" (GGNN), ICLR 2016.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

# ======================================================================
# Attention primitives (mirror GLORY's ``src/models/base/layers.py``)
# ======================================================================


class ScaledDotProductAttention(nnx.Module):
    """Scaled dot-product attention used inside ``MultiHeadAttention``.

    Note: GLORY's reference uses a non-standard masking scheme — scores
    are exponentiated before masking, then re-normalised.  We replicate
    that behaviour exactly to match the paper's initialization and
    training dynamics.
    """

    def __init__(self, d_k: int):
        self.d_k = d_k

    def __call__(
        self,
        Q: jax.Array,
        K: jax.Array,
        V: jax.Array,
        attn_mask: jax.Array | None = None,
    ) -> jax.Array:
        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / math.sqrt(self.d_k)
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        scores = jnp.exp(scores)
        if attn_mask is not None:
            scores = scores * jnp.expand_dims(attn_mask, axis=-2)
        attn = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-8)
        return jnp.matmul(attn, V)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention matching GLORY's Q/K/V projection choice."""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        head_num: int,
        head_dim: int,
        residual: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.head_num = head_num
        self.head_dim = head_dim
        self.residual = residual
        out_dim = head_num * head_dim

        xavier = nnx.initializers.xavier_uniform()
        self.W_Q = nnx.Linear(
            key_size,
            out_dim,
            use_bias=True,
            kernel_init=xavier,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.W_K = nnx.Linear(
            query_size,
            out_dim,
            use_bias=False,
            kernel_init=xavier,
            rngs=rngs,
        )
        self.W_V = nnx.Linear(
            value_size,
            out_dim,
            use_bias=True,
            kernel_init=xavier,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.attn = ScaledDotProductAttention(head_dim)

    def __call__(
        self,
        Q: jax.Array,
        K: jax.Array | None = None,
        V: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.shape[0]
        H, D = self.head_num, self.head_dim
        if mask is not None:
            mask = jnp.broadcast_to(
                jnp.expand_dims(mask, axis=1), (batch_size, H, mask.shape[-1])
            )
        q = self.W_Q(Q).reshape(batch_size, -1, H, D).transpose(0, 2, 1, 3)
        k = self.W_K(K).reshape(batch_size, -1, H, D).transpose(0, 2, 1, 3)
        v = self.W_V(V).reshape(batch_size, -1, H, D).transpose(0, 2, 1, 3)
        ctx = self.attn(q, k, v, mask)
        out = ctx.transpose(0, 2, 1, 3).reshape(batch_size, -1, H * D)
        return out + Q if self.residual else out


class AttentionPooling(nnx.Module):
    """Additive attention pooling over a sequence of vectors.

    Follows GLORY's implementation exactly:
    ``alpha = softmax(v^T tanh(W x)) / (sum + eps)`` with multiplicative
    masking (consistent with ``ScaledDotProductAttention``).
    """

    def __init__(self, emb_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        xavier_tanh = nnx.initializers.variance_scaling(
            math.sqrt(5.0 / 3.0) ** 2,
            "fan_avg",
            "uniform",
        )
        self.att_fc1 = nnx.Linear(
            emb_size,
            hidden_size,
            use_bias=True,
            kernel_init=xavier_tanh,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.att_fc2 = nnx.Linear(
            hidden_size,
            1,
            use_bias=True,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, attn_mask: jax.Array | None = None) -> jax.Array:
        # x: (B, N, E)
        e = jnp.tanh(self.att_fc1(x))
        raw = self.att_fc2(e)  # (B, N, 1)
        alpha = jnp.exp(raw - jnp.max(raw, axis=1, keepdims=True))
        if attn_mask is not None:
            alpha = alpha * jnp.expand_dims(attn_mask, axis=2)
        alpha = alpha / (jnp.sum(alpha, axis=1, keepdims=True) + 1e-8)
        # (B, E, N) @ (B, N, 1) → (B, E, 1) → (B, E)
        return jnp.squeeze(jnp.matmul(jnp.swapaxes(x, 1, 2), alpha), axis=-1)


# ======================================================================
# GatedGraphConv (pure JAX, no torch_geometric)
# ======================================================================


def _scatter_add(
    src: jax.Array,
    index: jax.Array,
    dim_size: int,
) -> jax.Array:
    """Segment sum along dim 0: ``out[index[i]] += src[i]``.

    Args:
        src: ``(E, D)`` source values (edge features).
        index: ``(E,)`` int target indices in ``[0, dim_size)``.
        dim_size: Output size along dim 0.

    Returns:
        ``(dim_size, D)`` aggregated output.
    """
    return jnp.zeros((dim_size, src.shape[1]), dtype=src.dtype).at[index].add(src)


class GRUCell(nnx.Module):
    """GRU cell matching PyTorch's ``nn.GRUCell`` behaviour.

    JAX/Flax NNX does not provide a standalone GRUCell, so we implement
    one with the same gate equations and default initialisation as
    PyTorch (uniform(-1/sqrt(H), 1/sqrt(H))).
    """

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        k = 1.0 / math.sqrt(hidden_size)
        uniform_init = nnx.initializers.uniform(k)

        # Input-to-hidden weights for r, z, n gates (stacked).
        self.W_ih = nnx.Linear(
            input_size,
            3 * hidden_size,
            use_bias=True,
            kernel_init=uniform_init,
            bias_init=uniform_init,
            rngs=rngs,
        )
        # Hidden-to-hidden weights for r, z, n gates (stacked).
        self.W_hh = nnx.Linear(
            hidden_size,
            3 * hidden_size,
            use_bias=True,
            kernel_init=uniform_init,
            bias_init=uniform_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, h: jax.Array) -> jax.Array:
        """Forward: x (N, input_size), h (N, hidden_size) → h' (N, hidden_size)."""
        H = self.hidden_size
        gi = self.W_ih(x)  # (N, 3H)
        gh = self.W_hh(h)  # (N, 3H)

        i_r, i_z, i_n = gi[:, :H], gi[:, H : 2 * H], gi[:, 2 * H :]
        h_r, h_z, h_n = gh[:, :H], gh[:, H : 2 * H], gh[:, 2 * H :]

        r = jax.nn.sigmoid(i_r + h_r)
        z = jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h


class GatedGraphConv(nnx.Module):
    """Gated Graph Neural Network (Li et al. 2016).

    Pure-JAX replacement for ``torch_geometric.nn.GatedGraphConv``.
    Shared weight ``W`` across all ``num_layers`` propagation steps,
    with a single ``GRUCell`` mixing messages into node states (matches
    GLORY's original configuration which uses the PyG default).

    Implementation:
        for t in range(num_layers):
            m_i = Σ_{j ∈ N(i)} W · h_j        # weighted neighbor sum
            h_i = GRU(m_i, h_i)               # gated update
    """

    def __init__(
        self,
        out_channels: int,
        num_layers: int = 3,
        aggr: str = "add",
        *,
        rngs: nnx.Rngs,
    ):
        if aggr != "add":
            raise ValueError(
                f"Only aggr='add' supported; got {aggr!r}. "
                f"GLORY's reference uses add aggregation."
            )
        self.out_channels = out_channels
        self.num_layers = num_layers
        xavier = nnx.initializers.xavier_uniform()
        # Initialize each layer's weight matrix separately (no vmap —
        # NNX RNG state cannot be mutated inside a traced context).
        weight_slices = [
            xavier(rngs.params(), (out_channels, out_channels))
            for _ in range(num_layers)
        ]
        self.weight = nnx.Param(jnp.stack(weight_slices, axis=0))
        self.rnn = GRUCell(out_channels, out_channels, rngs=rngs)

    def __call__(self, x: jax.Array, edge_index: jax.Array) -> jax.Array:
        """Propagate features across a graph.

        Args:
            x: ``(N, F_in)`` node features.  Will be zero-padded to
                ``(N, out_channels)`` if ``F_in < out_channels``.
            edge_index: ``(2, E)`` int tensor of directed edges
                ``(src, dst)`` — messages flow src → dst.

        Returns:
            ``(N, out_channels)`` updated node features.
        """
        if x.shape[1] < self.out_channels:
            pad = jnp.zeros((x.shape[0], self.out_channels - x.shape[1]), dtype=x.dtype)
            x = jnp.concatenate([x, pad], axis=-1)

        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        for i in range(self.num_layers):
            m = x @ self.weight.value[i]  # (N, D)
            agg = _scatter_add(m[src], dst, num_nodes)  # (N, D)
            x = self.rnn(agg, x)  # (N, D)

        return x


# ======================================================================
# Entity encoders (for use_entity=True)
# ======================================================================


class EntityEncoder(nnx.Module):
    """Local entity encoder: emb → dropout → MHA → LN → drop → pool → LN → Dense → LeakyReLU.

    Matches the reference ``EntityEncoder`` — projects entity_dim to news_dim.
    """

    def __init__(
        self,
        entity_dim: int,
        news_dim: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        head_num = entity_dim // head_dim
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.msa = MultiHeadAttention(
            entity_dim,
            entity_dim,
            entity_dim,
            head_num,
            head_dim,
            rngs=rngs,
        )
        self.layernorm1 = nnx.LayerNorm(entity_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.attn_pool = AttentionPooling(
            entity_dim,
            attention_hidden_dim,
            rngs=rngs,
        )
        self.layernorm2 = nnx.LayerNorm(entity_dim, rngs=rngs)
        self.linear = nnx.Linear(entity_dim, news_dim, rngs=rngs)

    def __call__(
        self,
        entity_input: jax.Array,
        mask: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode entities.

        Args:
            entity_input: ``(B, N, E_ent, entity_dim)`` entity embeddings.
            mask: optional ``(B*N, E_ent)`` mask.

        Returns:
            ``(B, N, news_dim)`` entity representations.
        """
        det = not training
        B, N = entity_input.shape[:2]
        num_ent = entity_input.shape[2]
        x = entity_input.reshape(B * N, num_ent, -1)
        x = self.dropout1(x, deterministic=det)
        x = self.msa(x, x, x, mask)
        x = self.layernorm1(x)
        x = self.dropout2(x, deterministic=det)
        x = self.attn_pool(x, mask)
        x = self.layernorm2(x)
        x = jax.nn.leaky_relu(self.linear(x), negative_slope=0.2)
        return x.reshape(B, N, -1)


class GlobalEntityEncoder(nnx.Module):
    """Global entity encoder: emb → dropout → MHA → LN → drop → pool → LN.

    Same as EntityEncoder but WITHOUT the final Dense + LeakyReLU.
    Uses head_num/head_dim from config so output dim = head_num * head_dim = news_dim.
    """

    def __init__(
        self,
        entity_dim: int,
        head_num: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.news_dim = head_num * head_dim
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.msa = MultiHeadAttention(
            entity_dim,
            entity_dim,
            entity_dim,
            head_num,
            head_dim,
            rngs=rngs,
        )
        self.layernorm1 = nnx.LayerNorm(self.news_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.attn_pool = AttentionPooling(
            self.news_dim,
            attention_hidden_dim,
            rngs=rngs,
        )
        self.layernorm2 = nnx.LayerNorm(self.news_dim, rngs=rngs)

    def __call__(
        self,
        entity_input: jax.Array,
        mask: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode neighbor entities.

        Args:
            entity_input: ``(B, N, E_ent, entity_dim)`` entity embeddings.
            mask: optional ``(B*N, E_ent)`` mask.

        Returns:
            ``(B, N, news_dim)`` entity representations.
        """
        det = not training
        B, N = entity_input.shape[:2]
        num_ent = entity_input.shape[2]
        x = entity_input.reshape(B * N, num_ent, -1)
        x = self.dropout1(x, deterministic=det)
        x = self.msa(x, x, x, mask)
        x = self.layernorm1(x)
        x = self.dropout2(x, deterministic=det)
        x = self.attn_pool(x, mask)
        x = self.layernorm2(x)
        return x.reshape(B, N, self.news_dim)
