"""Custom Flax NNX layers for news recommendation models.

Provides AdditiveAttention (soft-alignment attention), PositionalEncoding,
MultiHeadAttentionBlock, GraphSAGELayer, GraphAttentionLayer, and masking
utilities as pure JAX functions.
"""

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx


class AdditiveAttention(nnx.Module):
    """Soft-alignment-based attention layer (Flax NNX).

    Computes a weighted sum of the input sequence where the weights are
    learned during training. This is the standard additive attention mechanism
    used across NRMS, NAML, and LSTUR models.

    Args:
        input_dim: The feature dimension of the input (last axis size).
        query_vec_dim: The hidden dimension of the attention mechanism.
        rngs: NNX random number generator state.
    """

    def __init__(self, input_dim: int, query_vec_dim: int = 200, *, rngs: nnx.Rngs):
        glorot_init = nnx.initializers.glorot_uniform()
        key_w = rngs.params()
        key_q = rngs.params()

        self.W = nnx.Param(glorot_init(key_w, (input_dim, query_vec_dim)))
        self.b = nnx.Param(jnp.zeros((query_vec_dim,)))
        self.q = nnx.Param(glorot_init(key_q, (query_vec_dim, 1)))

    def __call__(self, inputs: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Forward pass.

        Args:
            inputs: 3-D tensor of shape ``(batch, time_steps, features)``.
            mask: Optional boolean mask ``(batch, time_steps)``; ``True``
                for positions to attend to.

        Returns:
            2-D tensor ``(batch, features)`` -- weighted sum of the input
            sequence.
        """
        # Dense transformation + tanh
        attention_hidden = jnp.tanh(jnp.matmul(inputs, self.W.value) + self.b.value)

        # Attention scores
        attention_scores = jnp.matmul(attention_hidden, self.q.value)
        attention_scores = jnp.squeeze(attention_scores, axis=-1)  # (batch, time_steps)

        # Exp and optional masking (following original paper implementation)
        if mask is None:
            attention = jnp.exp(attention_scores)
        else:
            attention = jnp.exp(attention_scores) * mask.astype(inputs.dtype)

        # Normalise
        attention_weights = attention / (
            jnp.sum(attention, axis=-1, keepdims=True) + 1e-7
        )

        # Weighted sum
        attention_weights_expanded = jnp.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights_expanded

        return jnp.sum(weighted_input, axis=1)


# ---------------------------------------------------------------------------
# Pure JAX masking utilities
# ---------------------------------------------------------------------------

def compute_mask(inputs: jax.Array) -> jax.Array:
    """Compute a boolean mask where non-zero positions are ``True``.

    Args:
        inputs: Integer tensor (typically token ids).

    Returns:
        Boolean tensor of the same shape.
    """
    return jnp.not_equal(inputs, 0)


def overwrite_mask(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Zero out positions indicated by a mask.

    Args:
        values: Tensor whose values will be masked.
        mask: Boolean or float mask ``(batch, time_steps)``.

    Returns:
        Masked tensor with the same shape as ``values``.
    """
    return values * jnp.expand_dims(mask, axis=-1)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nnx.Module):
    """Sinusoidal positional encoding (Flax NNX).

    Adds deterministic sinusoidal position signals to word embeddings so that
    attention layers can distinguish token positions.

    Args:
        model_dim: Embedding / model dimension (last axis of input).
        dropout_rate: Dropout applied after adding positional encoding.
        max_len: Maximum sequence length to pre-compute encodings for.
        rngs: NNX random number generator state.
    """

    def __init__(
        self,
        model_dim: int,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
        *,
        rngs: nnx.Rngs,
    ):
        # Pre-compute sinusoidal table (non-trainable)
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, model_dim, 2, dtype=np.float32)
            * (-np.log(10000.0) / model_dim)
        )
        pe = np.zeros((1, max_len, model_dim), dtype=np.float32)
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        self.pe = jnp.asarray(pe)  # (1, max_len, model_dim)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        """Add positional encoding and apply dropout.

        Args:
            x: ``(batch, seq_len, model_dim)`` input embeddings.
            training: Enable dropout.

        Returns:
            Position-encoded tensor with the same shape.
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, deterministic=not training)


# ---------------------------------------------------------------------------
# Multi-Head Attention Block (MAB)
# ---------------------------------------------------------------------------

class MultiHeadAttentionBlock(nnx.Module):
    """Multi-head attention block with residual connections and layer norm.

    Implements MAB(X, Y) = LN(H + FF(H)), H = LN(X + MultiHead(X, Y, Y))
    used in the CROWN news encoder.

    Args:
        dim_in: Input feature dimension.
        dim_out: Output feature dimension.
        num_heads: Number of attention heads.
        use_layer_norm: Whether to apply layer normalisation.
        dropout_rate: Dropout rate for attention weights and FF output.
        rngs: NNX random number generator state.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm

        self.fc_q = nnx.Linear(dim_in, dim_out, rngs=rngs)
        self.fc_k = nnx.Linear(dim_in, dim_out, rngs=rngs)
        self.fc_v = nnx.Linear(dim_in, dim_out, rngs=rngs)
        self.fc_o = nnx.Linear(dim_out, dim_out, rngs=rngs)

        if use_layer_norm:
            self.ln0 = nnx.LayerNorm(dim_out, rngs=rngs)
            self.ln1 = nnx.LayerNorm(dim_out, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        """Forward pass (self-attention variant: Q = K = x).

        Args:
            x: ``(batch, seq_len, dim_in)`` input tensor.
            training: Enable dropout.

        Returns:
            ``(batch, seq_len, dim_out)`` tensor.
        """
        det = not training

        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        batch_size, seq_len, _ = Q.shape
        head_dim = self.dim_out // self.num_heads

        # Reshape to (B, H, S, D)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(
            jnp.float32(head_dim)
        )
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=det)

        attn_out = jnp.matmul(attn_weights, V)  # (B, H, S, D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim_out)

        O = self.fc_o(attn_out)
        O = self.dropout(O, deterministic=det)
        O = O + x  # residual
        if self.use_layer_norm:
            O = self.ln0(O)

        # Feed-forward + residual
        O_ff = self.fc_o(jax.nn.relu(O))
        O = O + O_ff
        if self.use_layer_norm:
            O = self.ln1(O)

        return O


# ---------------------------------------------------------------------------
# GraphSAGE layer
# ---------------------------------------------------------------------------

class GraphSAGELayer(nnx.Module):
    """GraphSAGE layer for the bipartite user-news graph in CROWN.

    Performs mutual message-passing between user and news nodes on a
    bipartite adjacency matrix.

    Args:
        user_dim: Input dimension of user features.
        news_dim: Input dimension of news features.
        units: Output dimension for both user and news embeddings.
        aggregator: Aggregation type (``'mean'``, ``'max'``, ``'sum'``).
        dropout_rate: Dropout rate.
        normalize: Whether to L2-normalise the output.
        rngs: NNX random number generator state.
    """

    def __init__(
        self,
        user_dim: int,
        news_dim: int,
        units: int,
        aggregator: str = "mean",
        dropout_rate: float = 0.0,
        normalize: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.units = units
        self.aggregator = aggregator
        self.normalize = normalize

        glorot = nnx.initializers.glorot_uniform()

        self.W_user_self = nnx.Param(glorot(rngs.params(), (user_dim, units)))
        self.W_user_neigh = nnx.Param(glorot(rngs.params(), (news_dim, units)))
        self.b_user = nnx.Param(jnp.zeros((units,)))

        self.W_news_self = nnx.Param(glorot(rngs.params(), (news_dim, units)))
        self.W_news_neigh = nnx.Param(glorot(rngs.params(), (user_dim, units)))
        self.b_news = nnx.Param(jnp.zeros((units,)))

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    # -- aggregation helpers ------------------------------------------------

    def _aggregate(
        self, adjacency_weights: jax.Array, neighbor_features: jax.Array
    ) -> jax.Array:
        if self.aggregator == "mean":
            neigh_sum = jnp.matmul(adjacency_weights, neighbor_features)
            degree = jnp.sum(adjacency_weights, axis=-1, keepdims=True)
            degree = jnp.maximum(degree, 1e-7)
            return neigh_sum / degree
        elif self.aggregator == "sum":
            return jnp.matmul(adjacency_weights, neighbor_features)
        elif self.aggregator == "max":
            # Expand dims for element-wise masking
            expanded_adj = jnp.expand_dims(adjacency_weights, axis=-1)
            expanded_neigh = jnp.expand_dims(neighbor_features, axis=1)
            masked = expanded_neigh * expanded_adj
            mask_bool = expanded_adj > 0
            masked = jnp.where(mask_bool, masked, -1e9)
            return jnp.max(masked, axis=2)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    # -- forward ------------------------------------------------------------

    def __call__(
        self,
        user_features: jax.Array,
        news_features: jax.Array,
        adjacency_matrix: jax.Array,
        *,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            user_features: ``(B, num_users, user_dim)``.
            news_features: ``(B, num_news, news_dim)``.
            adjacency_matrix: ``(B, num_users+num_news, num_users+num_news)``.
            training: Enable dropout.

        Returns:
            Tuple ``(updated_users, updated_news)`` each with last dim ``units``.
        """
        det = not training
        num_users = user_features.shape[1]

        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        news_to_user = adjacency_matrix[:, num_users:, :num_users]

        # Update users
        news_for_users = self._aggregate(user_to_news, news_features)
        updated_users = jax.nn.relu(
            jnp.matmul(user_features, self.W_user_self.value)
            + jnp.matmul(news_for_users, self.W_user_neigh.value)
            + self.b_user.value
        )
        updated_users = self.dropout(updated_users, deterministic=det)

        # Update news
        users_for_news = self._aggregate(news_to_user, user_features)
        updated_news = jax.nn.relu(
            jnp.matmul(news_features, self.W_news_self.value)
            + jnp.matmul(users_for_news, self.W_news_neigh.value)
            + self.b_news.value
        )
        updated_news = self.dropout(updated_news, deterministic=det)

        if self.normalize:
            u_norm = jnp.sqrt(jnp.sum(jnp.square(updated_users), axis=-1, keepdims=True))
            updated_users = updated_users / (u_norm + 1e-7)
            n_norm = jnp.sqrt(jnp.sum(jnp.square(updated_news), axis=-1, keepdims=True))
            updated_news = updated_news / (n_norm + 1e-7)

        return updated_users, updated_news


# ---------------------------------------------------------------------------
# Graph Attention (GAT) layer
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nnx.Module):
    """Graph Attention Network layer for CROWN's bipartite user-news graph.

    Args:
        user_dim: Input dimension of user features.
        news_dim: Input dimension of news features.
        units: Output dimension.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate.
        alpha: LeakyReLU negative slope.
        concat_heads: Whether to concatenate (True) or average (False) heads.
        use_bias: Whether to add bias after the final projection.
        rngs: NNX random number generator state.
    """

    def __init__(
        self,
        user_dim: int,
        news_dim: int,
        units: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        alpha: float = 0.2,
        concat_heads: bool = True,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.units = units
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat_heads = concat_heads
        self.use_bias = use_bias

        if concat_heads:
            assert units % num_heads == 0
            self.head_dim = units // num_heads
        else:
            self.head_dim = units

        glorot = nnx.initializers.glorot_uniform()

        self.W_user = nnx.Param(
            glorot(rngs.params(), (num_heads, user_dim, self.head_dim))
        )
        self.W_news = nnx.Param(
            glorot(rngs.params(), (num_heads, news_dim, self.head_dim))
        )
        self.a_user_news = nnx.Param(
            glorot(rngs.params(), (num_heads, self.head_dim * 2, 1))
        )
        self.a_news_user = nnx.Param(
            glorot(rngs.params(), (num_heads, self.head_dim * 2, 1))
        )
        if use_bias:
            out_dim = units if concat_heads else self.head_dim
            self.b_user = nnx.Param(jnp.zeros((out_dim,)))
            self.b_news = nnx.Param(jnp.zeros((out_dim,)))

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    # -- helpers ------------------------------------------------------------

    def _compute_attention(
        self,
        queries: jax.Array,
        keys: jax.Array,
        attn_vec: jax.Array,
        adj_mask: jax.Array,
        *,
        training: bool,
    ) -> jax.Array:
        """Compute GAT attention coefficients.

        Args:
            queries: ``(B, H, Nq, D)``
            keys: ``(B, H, Nk, D)``
            attn_vec: ``(H, 2D, 1)``
            adj_mask: ``(B, Nq, Nk)``

        Returns:
            ``(B, H, Nq, Nk)`` attention coefficients.
        """
        B = queries.shape[0]
        Nq = queries.shape[2]
        Nk = keys.shape[2]

        q_exp = jnp.broadcast_to(
            queries[:, :, :, None, :], (B, self.num_heads, Nq, Nk, self.head_dim)
        )
        k_exp = jnp.broadcast_to(
            keys[:, :, None, :, :], (B, self.num_heads, Nq, Nk, self.head_dim)
        )
        concat_qk = jnp.concatenate([q_exp, k_exp], axis=-1)  # (B, H, Nq, Nk, 2D)

        scores = jnp.squeeze(
            jnp.matmul(concat_qk, attn_vec[None, :, :, :]), axis=-1
        )  # (B, H, Nq, Nk)
        scores = jax.nn.leaky_relu(scores, negative_slope=self.alpha)

        adj_exp = adj_mask[:, None, :, :]  # (B, 1, Nq, Nk)
        scores = jnp.where(adj_exp > 0, scores, -1e9)
        coeffs = jax.nn.softmax(scores, axis=-1)
        coeffs = self.dropout(coeffs, deterministic=not training)
        return coeffs

    # -- forward ------------------------------------------------------------

    def __call__(
        self,
        user_features: jax.Array,
        news_features: jax.Array,
        adjacency_matrix: jax.Array,
        *,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            user_features: ``(B, Nu, user_dim)``.
            news_features: ``(B, Nn, news_dim)``.
            adjacency_matrix: ``(B, Nu+Nn, Nu+Nn)``.
            training: Enable dropout.

        Returns:
            ``(updated_users, updated_news)`` with last dim ``units``.
        """
        B = user_features.shape[0]
        Nu = user_features.shape[1]
        Nn = news_features.shape[1]

        user_to_news = adjacency_matrix[:, :Nu, Nu:]
        news_to_user = adjacency_matrix[:, Nu:, :Nu]

        # Project through per-head weights: einsum('bud,hdk->bhuk')
        user_t = jnp.einsum("bud,hdk->bhuk", user_features, self.W_user.value)
        news_t = jnp.einsum("bnd,hdk->bhnk", news_features, self.W_news.value)

        # User update
        u2n_attn = self._compute_attention(
            user_t, news_t, self.a_user_news.value, user_to_news, training=training
        )
        user_updates = jnp.einsum("bhun,bhnk->bhuk", u2n_attn, news_t)

        # News update
        n2u_attn = self._compute_attention(
            news_t, user_t, self.a_news_user.value, news_to_user, training=training
        )
        news_updates = jnp.einsum("bhnu,bhuk->bhnk", n2u_attn, user_t)

        if self.concat_heads:
            updated_users = user_updates.reshape(B, Nu, self.units)
            updated_news = news_updates.reshape(B, Nn, self.units)
        else:
            updated_users = jnp.mean(user_updates, axis=1)
            updated_news = jnp.mean(news_updates, axis=1)

        if self.use_bias:
            updated_users = updated_users + self.b_user.value
            updated_news = updated_news + self.b_news.value

        updated_users = jax.nn.relu(updated_users)
        updated_news = jax.nn.relu(updated_news)
        updated_users = self.dropout(updated_users, deterministic=not training)
        updated_news = self.dropout(updated_news, deterministic=not training)

        return updated_users, updated_news
