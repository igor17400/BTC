"""DIGAT dual interactive graph encoder (Flax NNX).

Mirror of the PyTorch ``GraphEncoder``. Maintains a news-graph channel
(SAG subgraph per candidate) and a user-graph channel (history + topic
nodes). The two channels interact across ``graph_depth`` layers: each
graph's attention incorporates the other channel's context vector.

This file also hosts the scatter utilities (``scatter_softmax``,
``scatter_sum``) used by the user-graph context, and the shared
``ScaledDotProductAttention`` used in both context extractors. Pure JAX —
no ``jax_scatter`` or similar.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import DIGATConfig

# Match PyTorch's ``nn.init.calculate_gain`` values so cross-framework
# runs produce statistically equivalent initializations.
_RELU_GAIN = math.sqrt(2.0)
_LEAKY_RELU_GAIN_02 = math.sqrt(2.0 / (1.0 + 0.2**2))


# ======================================================================
# Container utility
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


def scatter_softmax(src: jax.Array, index: jax.Array, num_groups: int) -> jax.Array:
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

    # Max per group (for numerical stability) via one-hot mask.
    group_range = jnp.arange(num_groups)[None, None, :]  # (1, 1, G)
    mask = idx[:, :, None] == group_range  # (B, N, G)
    neg_inf = jnp.finfo(src.dtype).min
    masked = jnp.where(mask, src[:, :, None], neg_inf)  # (B, N, G)
    group_max = jnp.max(masked, axis=1)  # (B, G)
    element_max = jnp.take_along_axis(group_max, idx, axis=1)  # (B, N)

    exp_src = jnp.exp(src - element_max)

    group_sum = jnp.sum(jnp.where(mask, exp_src[:, :, None], 0.0), axis=1)  # (B, G)
    element_sum = jnp.take_along_axis(group_sum, idx, axis=1)  # (B, N)

    return exp_src / (element_sum + 1e-10)


def scatter_sum(src: jax.Array, index: jax.Array, dim: int, dim_size: int) -> jax.Array:
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
    one_hot = jax.nn.one_hot(idx, dim_size, dtype=src.dtype)
    return jnp.einsum("bnk,bnd->bkd", one_hot, src)


# ======================================================================
# Shared attention primitive
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
        keys = self.K(features)  # (B, N, A)
        q = self.Q(query)[:, :, None]  # (B, A, 1)
        scores = jnp.squeeze(jnp.matmul(keys, q), axis=-1) / self.scale  # (B, N)
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        weights = jax.nn.softmax(scores, axis=1)[:, None, :]  # (B, 1, N)
        return jnp.squeeze(jnp.matmul(weights, features), axis=1)  # (B, F)


# ======================================================================
# Dual graph encoder
# ======================================================================


class GraphEncoder(nnx.Module):
    """Dual Interactive Graph Attention encoder."""

    def __init__(
        self,
        config: DIGATConfig,
        num_categories: int,
        *,
        rngs: nnx.Rngs,
    ):
        D = config.news_embedding_dim
        depth = config.graph_depth
        self.graph_depth = depth
        self.max_history = config.max_history_length
        self.D = D
        self.scale = math.sqrt(float(D))

        # topic_dropout — full rate, applied to topic embeddings after relu+residual.
        # attn_dropout  — full rate, applied to attention weight matrices.
        # input_dropout — half rate, applied to node embeddings before graph update
        #                 layers and to the gate linear output in news context;
        #                 mirrors reference ``dropout__ = Dropout(rate/2)``.
        self.topic_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.attn_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.input_dropout = nnx.Dropout(rate=config.dropout_rate / 2, rngs=rngs)

        # PyTorch: ``nn.init.xavier_uniform_(w, gain=G)`` →
        # ``variance_scaling(G**2, "fan_avg", "uniform")``.
        xavier = nnx.initializers.xavier_uniform()
        xavier_relu = nnx.initializers.variance_scaling(
            _RELU_GAIN**2,
            "fan_avg",
            "uniform",
        )
        xavier_leaky = nnx.initializers.variance_scaling(
            _LEAKY_RELU_GAIN_02**2,
            "fan_avg",
            "uniform",
        )

        # --- News graph context ---
        self.news_ctx_attn = ScaledDotProductAttention(D, D, D, rngs=rngs)
        self.news_ctx_gate = nnx.Linear(
            D * 2,
            D,
            kernel_init=xavier,
            rngs=rngs,
        )

        # --- User graph context (topic-level scatter + attention) ---
        self.user_news_K = nnx.Linear(
            D,
            D,
            use_bias=False,
            kernel_init=xavier,
            rngs=rngs,
        )
        self.user_news_Q = nnx.Linear(
            D,
            D,
            use_bias=True,
            kernel_init=xavier,
            rngs=rngs,
        )
        self.topic_affine = nnx.Linear(
            D,
            D,
            kernel_init=xavier_relu,
            rngs=rngs,
        )
        self.user_ctx_attn = ScaledDotProductAttention(D, D, D, rngs=rngs)

        # --- Per-depth news graph update layers ---
        self.n_W = ModuleStack(
            [nnx.Linear(D, D, kernel_init=xavier, rngs=rngs) for _ in range(depth)]
        )
        self.n_ffn1 = ModuleStack(
            [
                nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
                for _ in range(depth)
            ]
        )
        self.n_ffn2 = ModuleStack(
            [
                nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
                for _ in range(depth)
            ]
        )
        self.n_ffn3 = ModuleStack(
            [nnx.Linear(D, D, kernel_init=xavier_relu, rngs=rngs) for _ in range(depth)]
        )
        self.n_a = ModuleStack(
            [
                nnx.Linear(D, 1, use_bias=False, kernel_init=xavier_leaky, rngs=rngs)
                for _ in range(depth)
            ]
        )

        # --- Per-depth user graph update layers ---
        self.u_W = ModuleStack(
            [nnx.Linear(D, D, kernel_init=xavier, rngs=rngs) for _ in range(depth)]
        )
        self.u_ffn1 = ModuleStack(
            [
                nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
                for _ in range(depth)
            ]
        )
        self.u_ffn2 = ModuleStack(
            [
                nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
                for _ in range(depth)
            ]
        )
        self.u_ffn3 = ModuleStack(
            [nnx.Linear(D, D, kernel_init=xavier_relu, rngs=rngs) for _ in range(depth)]
        )
        self.u_a = ModuleStack(
            [
                nnx.Linear(D, 1, use_bias=False, kernel_init=xavier_leaky, rngs=rngs)
                for _ in range(depth)
            ]
        )

        # Learnable topic node embeddings — initialized here in NNX
        # (unlike PyTorch where this is assigned by ``DIGAT.__init__``).
        # NNX requires explicit declaration of data attributes up-front.
        self.topic_node_emb = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (num_categories, D),
                minval=-0.1,
                maxval=0.1,
            )
        )

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def _news_graph_context(
        self,
        emb: jax.Array,
        mask: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        local = emb[:, 0, :]
        glob = self.news_ctx_attn(emb, local, mask=mask)
        gate_in = jnp.concatenate([local, glob], axis=-1)
        gate = jax.nn.sigmoid(
            self.input_dropout(self.news_ctx_gate(gate_in), deterministic=deterministic)
        )
        return gate * local + (1 - gate) * glob

    def _user_graph_context(
        self,
        emb: jax.Array,
        cat_mask: jax.Array,
        cat_indices: jax.Array,
        news_ctx: jax.Array,
        num_categories: int,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        hist = emb[:, : self.max_history, :]  # (B, H, D)

        K = self.user_news_K(hist)
        # news_ctx is referred to as "news graph context — c_n" in the paper
        Q = jnp.expand_dims(self.user_news_Q(news_ctx), axis=2)  # (B, D, 1)
        scores = jnp.squeeze(jnp.matmul(K, Q), axis=2) / self.scale  # (B, H)

        alpha = scatter_softmax(scores, cat_indices, num_categories)  # (B, H)
        weighted = jnp.expand_dims(alpha, axis=-1) * hist  # (B, H, D)
        topic_emb = scatter_sum(weighted, cat_indices, dim=1, dim_size=num_categories)
        topic_emb = self.topic_dropout(
            jax.nn.relu(self.topic_affine(topic_emb)) + topic_emb,
            deterministic=deterministic,
        )

        return self.user_ctx_attn(topic_emb, news_ctx, mask=cat_mask)

    # ------------------------------------------------------------------
    # Graph update (one layer)
    # ------------------------------------------------------------------

    def _update_graph(
        self,
        idx: int,
        emb: jax.Array,
        adj: jax.Array,
        cross_ctx: jax.Array,
        W_list: ModuleStack,
        ffn1_list: ModuleStack,
        ffn2_list: ModuleStack,
        ffn3_list: ModuleStack,
        a_list: ModuleStack,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Single cross-interactive graph attention update."""
        batch_size, _, _ = emb.shape
        emb = self.input_dropout(emb, deterministic=deterministic)
        h = W_list[idx](emb)

        K1 = jnp.expand_dims(ffn1_list[idx](emb), axis=1)  # (B, 1, N, D)
        K2 = jnp.expand_dims(ffn2_list[idx](emb), axis=2)  # (B, N, 1, D)
        K3 = ffn3_list[idx](cross_ctx).reshape(batch_size, 1, 1, self.D)

        scores = jnp.squeeze(
            a_list[idx](jax.nn.relu(K1 + K2 + K3)),
            axis=-1,
        )  # (B, N, N)
        scores = jax.nn.leaky_relu(scores, negative_slope=0.2)
        scores = jnp.where(adj == 0, -1e9, scores)

        alpha = self.attn_dropout(
            jax.nn.softmax(scores, axis=2),
            deterministic=deterministic,
        )

        return jax.nn.relu(jnp.matmul(alpha, h)) + emb

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        news_emb: jax.Array,
        news_graph: jax.Array,
        news_mask: jax.Array,
        user_news_emb: jax.Array,
        user_graph: jax.Array,
        user_cat_mask: jax.Array,
        user_cat_indices: jax.Array,
        num_categories: int,
        *,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Dual graph interaction forward.

        Args:
            news_emb: ``(B, G_n, D)`` SAG node embeddings per candidate.
            news_graph: ``(B, G_n, G_n)`` SAG adjacency.
            news_mask: ``(B, G_n)`` valid SAG nodes.
            user_news_emb: ``(B, H, D)`` history news embeddings.
            user_graph: ``(B, G_u, G_u)`` user graph adjacency.
            user_cat_mask: ``(B, C)`` active categories.
            user_cat_indices: ``(B, H)`` topic per history item.
            num_categories: ``C`` number of topic nodes.

        Returns:
            ``(news_ctx, user_ctx)`` — each ``(B, D)``.
        """
        det = not training
        batch_size = news_emb.shape[0]
        topic = self.topic_node_emb.value  # (C, D)
        topic_nodes = jnp.broadcast_to(
            topic[None, :, :], (batch_size, topic.shape[0], topic.shape[1])
        )
        user_emb = jnp.concatenate(
            [user_news_emb, self.input_dropout(topic_nodes, deterministic=det)],
            axis=1,
        )

        news_ctx = self._news_graph_context(news_emb, news_mask, deterministic=det)
        user_ctx = self._user_graph_context(
            user_emb,
            user_cat_mask,
            user_cat_indices,
            news_ctx,
            num_categories,
            deterministic=det,
        )

        for i in range(self.graph_depth):
            news_emb = self._update_graph(
                i,
                news_emb,
                news_graph,
                user_ctx,
                self.n_W,
                self.n_ffn1,
                self.n_ffn2,
                self.n_ffn3,
                self.n_a,
                deterministic=det,
            )
            user_emb = self._update_graph(
                i,
                user_emb,
                user_graph,
                news_ctx,
                self.u_W,
                self.u_ffn1,
                self.u_ffn2,
                self.u_ffn3,
                self.u_a,
                deterministic=det,
            )
            news_ctx = news_ctx + self._news_graph_context(
                news_emb,
                news_mask,
                deterministic=det,
            )
            user_ctx = user_ctx + self._user_graph_context(
                user_emb,
                user_cat_mask,
                user_cat_indices,
                news_ctx,
                num_categories,
                deterministic=det,
            )

        return news_ctx, user_ctx
