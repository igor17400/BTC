"""CROWN user encoder (Flax NNX) — encoder-agnostic, packed-input.

Mirror of the PyTorch user encoder. Accepts packed history
``(B, H, 3)`` = ``[news_idx | category | subcategory]`` per slot;
identical contract for training and eval.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import CROWNConfig

from .news_encoder import NewsEncoder


class UserQueryAttention(nnx.Module):
    """Additive attention with the GNN-updated user proxy as query."""

    def __init__(self, feature_dim: int, attention_dim: int, *, rngs: nnx.Rngs):
        self.W_key = nnx.Linear(feature_dim, attention_dim, use_bias=True, rngs=rngs)
        self.W_query = nnx.Linear(feature_dim, attention_dim, use_bias=False, rngs=rngs)

    def __call__(
        self, news: jax.Array, user_node: jax.Array, mask: jax.Array | None = None
    ) -> jax.Array:
        keys = jnp.tanh(self.W_key(news))  # (B, H, A)
        query = self.W_query(user_node)[:, None, :]  # (B, 1, A)
        scores = jnp.sum(keys * query, axis=-1)  # (B, H)
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1)[:, :, None]
        return jnp.sum(news * weights, axis=1)


class BipartiteGATLayer(nnx.Module):
    """GAT on a user<->news bipartite graph with self-loops."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        alpha: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.alpha = alpha

        self.W = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        glorot = nnx.initializers.glorot_uniform()
        self.a_src = nnx.Param(glorot(rngs.params(), (num_heads, self.head_dim)))
        self.a_dst = nnx.Param(glorot(rngs.params(), (num_heads, self.head_dim)))
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        user: jax.Array,
        news: jax.Array,
        news_mask: jax.Array,
        *,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        B, H, D = news.shape
        NH, HD = self.num_heads, self.head_dim
        neg_inf = jnp.finfo(news.dtype).min

        Wu = self.W(user).reshape(B, NH, HD)
        Wn = self.W(news).reshape(B, H, NH, HD)

        src_u = jnp.sum(Wu * self.a_src.value, axis=-1)
        dst_u = jnp.sum(Wu * self.a_dst.value, axis=-1)
        src_n = jnp.sum(Wn * self.a_src.value, axis=-1)
        dst_n = jnp.sum(Wn * self.a_dst.value, axis=-1)

        score_u_n = jax.nn.leaky_relu(
            src_u[:, None, :] + dst_n, negative_slope=self.alpha
        )
        score_u_u = jax.nn.leaky_relu(src_u + dst_u, negative_slope=self.alpha)
        score_u_n = jnp.where(news_mask[:, :, None], score_u_n, neg_inf)

        all_u = jnp.concatenate([score_u_n, score_u_u[:, None, :]], axis=1)
        attn_u = self.dropout(
            jax.nn.softmax(all_u, axis=1), deterministic=deterministic
        )
        u_from_n = jnp.sum(attn_u[:, :H, :, None] * Wn, axis=1)
        u_self = attn_u[:, H, :, None] * Wu
        user_new = jax.nn.elu(u_from_n + u_self).reshape(B, D)

        score_n_u = jax.nn.leaky_relu(
            src_n + dst_u[:, None, :], negative_slope=self.alpha
        )
        score_n_n = jax.nn.leaky_relu(src_n + dst_n, negative_slope=self.alpha)
        stacked = jnp.stack([score_n_u, score_n_n], axis=-2)
        attn_n = self.dropout(
            jax.nn.softmax(stacked, axis=-2), deterministic=deterministic
        )
        n_from_u = attn_n[:, :, 0, :, None] * Wu[:, None, :, :]
        n_self = attn_n[:, :, 1, :, None] * Wn
        news_new = jax.nn.elu(n_from_u + n_self).reshape(B, H, D)

        return user_new, news_new


class BipartiteSAGELayer(nnx.Module):
    """GraphSAGE on a user<->news bipartite graph (mean aggregator)."""

    def __init__(self, dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.W_self = nnx.Linear(dim, dim, rngs=rngs)
        self.W_neigh = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        user: jax.Array,
        news: jax.Array,
        news_mask: jax.Array,
        *,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        m = news_mask[:, :, None].astype(news.dtype)
        news_count = jnp.maximum(jnp.sum(m, axis=1), 1.0)
        news_mean = jnp.sum(news * m, axis=1) / news_count

        user_new = jax.nn.relu(self.W_self(user) + self.W_neigh(news_mean))
        news_new = jax.nn.relu(
            self.W_self(news) + self.W_neigh(user[:, None, :] * jnp.ones_like(news))
        )

        user_new = user_new / (jnp.linalg.norm(user_new, axis=-1, keepdims=True) + 1e-8)
        news_new = news_new / (jnp.linalg.norm(news_new, axis=-1, keepdims=True) + 1e-8)
        return (
            self.dropout(user_new, deterministic=deterministic),
            self.dropout(news_new, deterministic=deterministic),
        )


class UserEncoder(nnx.Module):
    """CROWN user encoder: bipartite GNN + paper eq. 9 user-query attention."""

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: NewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder
        news_emb_dim = news_encoder.news_embedding_dim

        self.user_node = nnx.Param(
            jax.random.uniform(rngs.params(), (news_emb_dim,), minval=-0.1, maxval=0.1)
        )

        if config.gnn_type == "gat":
            self.gnn_layers = nnx.List(
                [
                    BipartiteGATLayer(
                        dim=news_emb_dim,
                        num_heads=config.gat_num_heads,
                        dropout_rate=config.dropout_rate,
                        alpha=config.gat_alpha,
                        rngs=rngs,
                    )
                    for _ in range(config.graph_num_layers)
                ]
            )
        elif config.gnn_type == "graphsage":
            self.gnn_layers = nnx.List(
                [
                    BipartiteSAGELayer(
                        dim=news_emb_dim,
                        dropout_rate=config.dropout_rate,
                        rngs=rngs,
                    )
                    for _ in range(config.graph_num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown gnn_type: {config.gnn_type!r}")

        self.user_attention = UserQueryAttention(
            feature_dim=news_emb_dim,
            attention_dim=config.user_attention_dim,
            rngs=rngs,
        )

    def _encode_history_graph(
        self,
        history_packed: jax.Array,
        history_mask: jax.Array,
        *,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        B = history_packed.shape[0]
        det = not training

        news = self.news_encoder(
            history_packed, compute_aux_loss=False, training=training
        )

        user = jnp.broadcast_to(
            self.user_node.value[None, :], (B, self.user_node.value.shape[0])
        )
        for gnn in self.gnn_layers:
            user, news = gnn(user, news, history_mask, deterministic=det)
        return user, news

    def forward_with_candidates(
        self,
        history_packed: jax.Array,
        history_mask: jax.Array,
        candidate_repr: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        user_node, news = self._encode_history_graph(
            history_packed, history_mask, training=training
        )
        return self.user_attention(news, user_node, mask=history_mask)

    def __call__(
        self, packed_features: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Eval entry: packed history -> single user vector."""
        history_mask = self.news_encoder.valid_mask(packed_features)
        user_node, news = self._encode_history_graph(
            packed_features, history_mask, training=training
        )
        return self.user_attention(news, user_node, mask=history_mask)
