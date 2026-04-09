"""Graph and transformer layers for Flax NNX (CROWN model).

Layers used exclusively by the CROWN news recommendation model:
- :class:`PositionalEncoding` — sinusoidal position signals.
- :class:`MultiHeadAttentionBlock` — MAB with residual + layer norm.
- :class:`GraphSAGELayer` — bipartite message-passing.
- :class:`GraphAttentionLayer` — bipartite GAT.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class PositionalEncoding(nnx.Module):
    """Sinusoidal positional encoding (Flax NNX)."""

    def __init__(
        self,
        model_dim: int,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
        *,
        rngs: nnx.Rngs,
    ):
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, model_dim, 2, dtype=np.float32)
            * (-np.log(10000.0) / model_dim)
        )
        pe = np.zeros((1, max_len, model_dim), dtype=np.float32)
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        self.pe = jnp.asarray(pe)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, deterministic=not training)


class MultiHeadAttentionBlock(nnx.Module):
    """Multi-head attention block with residual connections and layer norm (CROWN)."""

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
        det = not training
        Q, K, V = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        batch_size, seq_len, _ = Q.shape
        head_dim = self.dim_out // self.num_heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        K = K.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        V = V.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(
            0, 2, 1, 3
        )
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(
            jnp.float32(head_dim)
        )
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=det)
        attn_out = jnp.matmul(attn_weights, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.dim_out
        )
        out = self.fc_o(attn_out)
        out = self.dropout(out, deterministic=det)
        out = out + x
        if self.use_layer_norm:
            out = self.ln0(out)
        ff = self.fc_o(jax.nn.relu(out))
        out = out + ff
        if self.use_layer_norm:
            out = self.ln1(out)
        return out


class GraphSAGELayer(nnx.Module):
    """GraphSAGE layer for the bipartite user-news graph in CROWN."""

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

    def _aggregate(
        self, adjacency_weights: jax.Array, neighbor_features: jax.Array
    ) -> jax.Array:
        if self.aggregator == "mean":
            neigh_sum = jnp.matmul(adjacency_weights, neighbor_features)
            degree = jnp.maximum(
                jnp.sum(adjacency_weights, axis=-1, keepdims=True), 1e-7
            )
            return neigh_sum / degree
        elif self.aggregator == "sum":
            return jnp.matmul(adjacency_weights, neighbor_features)
        elif self.aggregator == "max":
            expanded_adj = jnp.expand_dims(adjacency_weights, axis=-1)
            expanded_neigh = jnp.expand_dims(neighbor_features, axis=1)
            masked = jnp.where(expanded_adj > 0, expanded_neigh * expanded_adj, -1e9)
            return jnp.max(masked, axis=2)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def __call__(
        self,
        user_features: jax.Array,
        news_features: jax.Array,
        adjacency_matrix: jax.Array,
        *,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        det = not training
        num_users = user_features.shape[1]
        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        news_to_user = adjacency_matrix[:, num_users:, :num_users]
        news_for_users = self._aggregate(user_to_news, news_features)
        updated_users = jax.nn.relu(
            jnp.matmul(user_features, self.W_user_self.value)
            + jnp.matmul(news_for_users, self.W_user_neigh.value)
            + self.b_user.value
        )
        updated_users = self.dropout(updated_users, deterministic=det)
        users_for_news = self._aggregate(news_to_user, user_features)
        updated_news = jax.nn.relu(
            jnp.matmul(news_features, self.W_news_self.value)
            + jnp.matmul(users_for_news, self.W_news_neigh.value)
            + self.b_news.value
        )
        updated_news = self.dropout(updated_news, deterministic=det)
        if self.normalize:
            u_norm = jnp.sqrt(
                jnp.sum(jnp.square(updated_users), axis=-1, keepdims=True)
            )
            updated_users = updated_users / (u_norm + 1e-7)
            n_norm = jnp.sqrt(jnp.sum(jnp.square(updated_news), axis=-1, keepdims=True))
            updated_news = updated_news / (n_norm + 1e-7)
        return updated_users, updated_news


class GraphAttentionLayer(nnx.Module):
    """Graph Attention Network layer for CROWN's bipartite user-news graph."""

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
        self.head_dim = units // num_heads if concat_heads else units
        if concat_heads:
            assert units % num_heads == 0
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

    def _compute_attention(self, queries, keys, attn_vec, adj_mask, *, training):
        B, Nq, Nk = queries.shape[0], queries.shape[2], keys.shape[2]
        q_exp = jnp.broadcast_to(
            queries[:, :, :, None, :], (B, self.num_heads, Nq, Nk, self.head_dim)
        )
        k_exp = jnp.broadcast_to(
            keys[:, :, None, :, :], (B, self.num_heads, Nq, Nk, self.head_dim)
        )
        concat_qk = jnp.concatenate([q_exp, k_exp], axis=-1)
        scores = jnp.squeeze(jnp.matmul(concat_qk, attn_vec[None, :, :, :]), axis=-1)
        scores = jax.nn.leaky_relu(scores, negative_slope=self.alpha)
        scores = jnp.where(adj_mask[:, None, :, :] > 0, scores, -1e9)
        coeffs = jax.nn.softmax(scores, axis=-1)
        return self.dropout(coeffs, deterministic=not training)

    def __call__(
        self, user_features, news_features, adjacency_matrix, *, training=False
    ):
        B, Nu, Nn = (
            user_features.shape[0],
            user_features.shape[1],
            news_features.shape[1],
        )
        user_to_news = adjacency_matrix[:, :Nu, Nu:]
        news_to_user = adjacency_matrix[:, Nu:, :Nu]
        user_t = jnp.einsum("bud,hdk->bhuk", user_features, self.W_user.value)
        news_t = jnp.einsum("bnd,hdk->bhnk", news_features, self.W_news.value)
        u2n_attn = self._compute_attention(
            user_t, news_t, self.a_user_news.value, user_to_news, training=training
        )
        user_updates = jnp.einsum("bhun,bhnk->bhuk", u2n_attn, news_t)
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
