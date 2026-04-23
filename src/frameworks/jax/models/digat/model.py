"""DIGAT (EMNLP 2022 Findings) — JAX/Flax NNX implementation.

Dual Interactive Graph Attention Networks for news recommendation.
Two interacting graph channels (news-SAG + user-topic) co-evolve across
``graph_depth`` layers.  The SAG (Semantic Augmented Graph) is precomputed
offline; the user-topic graph is built per behavior from category
assignments.

Structurally identical to the PyTorch version in
:mod:`src.frameworks.pytorch.models.digat.model` — the two ports share
identical module layout, method names, and forward-pass shapes so
evaluation results are reproducible across frameworks.

Reference: Mao et al., "DIGAT: Modeling News Recommendation with
Dual-Graph Interaction", EMNLP 2022 Findings.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import DIGATConfig

from ..base import BaseModel
from .layers import (
    AdditiveAttention,
    ModuleStack,
    MultiHeadSelfAttention,
    ScaledDotProductAttention,
    scatter_softmax,
    scatter_sum,
)


# Match PyTorch's ``nn.init.calculate_gain`` values so cross-framework
# runs produce statistically equivalent initializations.
_RELU_GAIN = math.sqrt(2.0)
_LEAKY_RELU_GAIN_02 = math.sqrt(2.0 / (1.0 + 0.2 ** 2))


# ======================================================================
# News encoder
# ======================================================================


class DIGATNewsEncoder(nnx.Module):
    """MSA-based news encoder: embedding → MHA → additive attention."""

    def __init__(
        self,
        config: DIGATConfig,
        word_embedding: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.word_embedding = word_embedding
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.msa = MultiHeadSelfAttention(
            config.embedding_size, config.msa_head_num, config.msa_head_dim, rngs=rngs,
        )
        self.news_embedding_dim = config.news_embedding_dim
        self.attention = AdditiveAttention(
            self.news_embedding_dim, config.attention_dim, rngs=rngs,
        )

    def __call__(
        self,
        title_text: jax.Array,
        title_mask: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode a batch of news with SAG neighbor structure.

        Args:
            title_text: ``(batch_size, num_news, title_len)`` token ids.
            title_mask: ``(batch_size, num_news, title_len)`` float mask
                (1 = valid token).

        Returns:
            ``(batch_size, num_news, news_embedding_dim)`` news reprs.
        """
        det = not training
        batch_size, num_news, title_len = title_text.shape
        flat_text = title_text.reshape(batch_size * num_news, title_len)
        flat_mask = (
            title_mask.reshape(batch_size * num_news, title_len)
            if title_mask is not None
            else None
        )

        w = self.dropout(self.word_embedding(flat_text), deterministic=det)
        h = jax.nn.relu(self.msa(w))
        news_repr = self.attention(h, mask=flat_mask)
        return news_repr.reshape(batch_size, num_news, self.news_embedding_dim)


# ======================================================================
# Dual graph encoder
# ======================================================================


class DIGATGraphEncoder(nnx.Module):
    """Dual Interactive Graph Attention encoder.

    Maintains a news-graph channel (SAG subgraph per candidate) and a
    user-graph channel (history + topic nodes).  The two channels
    interact across ``graph_depth`` layers: each graph's attention
    incorporates the other channel's context vector.

    Note: ``topic_node_emb`` is assigned post-``__init__`` by the top
    ``DIGAT`` class once ``num_categories`` is known (matches the
    PyTorch layout).
    """

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

        # topic_dropout  — full rate, applied to topic embeddings after relu+residual
        # attn_dropout   — full rate, applied to attention weight matrices
        # input_dropout  — half rate, applied to node embeddings before graph update layers
        #                  and to the gate linear output in news context; mirrors reference
        #                  dropout__ = Dropout(rate/2) from the original DIGAT code.
        self.topic_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.attn_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.input_dropout = nnx.Dropout(rate=config.dropout_rate / 2, rngs=rngs)

        # PyTorch: nn.init.xavier_uniform_(w, gain=G) → variance_scaling(G**2, "fan_avg", "uniform")
        xavier = nnx.initializers.xavier_uniform()
        xavier_relu = nnx.initializers.variance_scaling(
            _RELU_GAIN ** 2, "fan_avg", "uniform",
        )
        xavier_leaky = nnx.initializers.variance_scaling(
            _LEAKY_RELU_GAIN_02 ** 2, "fan_avg", "uniform",
        )

        # --- News graph context ---
        self.news_ctx_attn = ScaledDotProductAttention(D, D, D, rngs=rngs)
        self.news_ctx_gate = nnx.Linear(
            D * 2, D, kernel_init=xavier, rngs=rngs,
        )

        # --- User graph context (topic-level scatter + attention) ---
        self.user_news_K = nnx.Linear(
            D, D, use_bias=False, kernel_init=xavier, rngs=rngs,
        )
        self.user_news_Q = nnx.Linear(
            D, D, use_bias=True, kernel_init=xavier, rngs=rngs,
        )
        self.topic_affine = nnx.Linear(
            D, D, kernel_init=xavier_relu, rngs=rngs,
        )
        self.user_ctx_attn = ScaledDotProductAttention(D, D, D, rngs=rngs)

        # --- Per-depth news graph update layers ---
        self.n_W = ModuleStack([
            nnx.Linear(D, D, kernel_init=xavier, rngs=rngs) for _ in range(depth)
        ])
        self.n_ffn1 = ModuleStack([
            nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
            for _ in range(depth)
        ])
        self.n_ffn2 = ModuleStack([
            nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
            for _ in range(depth)
        ])
        self.n_ffn3 = ModuleStack([
            nnx.Linear(D, D, kernel_init=xavier_relu, rngs=rngs) for _ in range(depth)
        ])
        self.n_a = ModuleStack([
            nnx.Linear(D, 1, use_bias=False, kernel_init=xavier_leaky, rngs=rngs)
            for _ in range(depth)
        ])

        # --- Per-depth user graph update layers ---
        self.u_W = ModuleStack([
            nnx.Linear(D, D, kernel_init=xavier, rngs=rngs) for _ in range(depth)
        ])
        self.u_ffn1 = ModuleStack([
            nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
            for _ in range(depth)
        ])
        self.u_ffn2 = ModuleStack([
            nnx.Linear(D, D, use_bias=False, kernel_init=xavier_relu, rngs=rngs)
            for _ in range(depth)
        ])
        self.u_ffn3 = ModuleStack([
            nnx.Linear(D, D, kernel_init=xavier_relu, rngs=rngs) for _ in range(depth)
        ])
        self.u_a = ModuleStack([
            nnx.Linear(D, 1, use_bias=False, kernel_init=xavier_leaky, rngs=rngs)
            for _ in range(depth)
        ])

        # Learnable topic node embeddings — initialized here in NNX
        # (unlike PyTorch where this is assigned by ``DIGAT.__init__``).
        # Needs ``num_categories`` at construction time because NNX
        # requires explicit declaration of data attributes up-front.
        self.topic_node_emb = nnx.Param(
            jax.random.uniform(
                rngs.params(), (num_categories, D), minval=-0.1, maxval=0.1,
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
        """Gated local/global context from a news SAG subgraph.

        Args:
            emb: ``(B, G, D)`` node embeddings for the SAG subgraph.
            mask: ``(B, G)`` valid-node mask.

        Returns:
            ``(B, D)`` context vector.
        """
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
        """Two-level user context: topic-level scatter → user-level attention.

        Args:
            emb: ``(B, H+C, D)`` history news + topic node embeddings.
            cat_mask: ``(B, C)`` which categories are active.
            cat_indices: ``(B, H)`` topic assignment per history news.
            news_ctx: ``(B, D)`` current news context.
            num_categories: ``C`` number of category/topic nodes.

        Returns:
            ``(B, D)`` user context vector.
        """
        hist = emb[:, : self.max_history, :]  # (B, H, D)

        K = self.user_news_K(hist)
        Q = jnp.expand_dims(self.user_news_Q(news_ctx), axis=2)          # (B, D, 1)
        scores = jnp.squeeze(jnp.matmul(K, Q), axis=2) / self.scale      # (B, H)

        alpha = scatter_softmax(scores, cat_indices, num_categories)     # (B, H)
        weighted = jnp.expand_dims(alpha, axis=-1) * hist                # (B, H, D)
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
        batch_size, num_nodes, _ = emb.shape
        emb = self.input_dropout(emb, deterministic=deterministic)
        h = W_list[idx](emb)
        K1 = jnp.expand_dims(ffn1_list[idx](emb), axis=1)   # (B, 1, N, D)
        K2 = jnp.expand_dims(ffn2_list[idx](emb), axis=2)   # (B, N, 1, D)
        K3 = ffn3_list[idx](cross_ctx).reshape(batch_size, 1, 1, self.D)
        scores = jnp.squeeze(
            a_list[idx](jax.nn.relu(K1 + K2 + K3)), axis=-1,
        )                                                    # (B, N, N)
        scores = jax.nn.leaky_relu(scores, negative_slope=0.2)
        scores = jnp.where(adj == 0, -1e9, scores)
        alpha = self.attn_dropout(
            jax.nn.softmax(scores, axis=2), deterministic=deterministic,
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
        topic = self.topic_node_emb.value                                  # (C, D)
        topic_nodes = jnp.broadcast_to(topic[None, :, :], (batch_size, topic.shape[0], topic.shape[1]))
        user_emb = jnp.concatenate(
            [user_news_emb, self.input_dropout(topic_nodes, deterministic=det)],
            axis=1,
        )

        news_ctx = self._news_graph_context(news_emb, news_mask, deterministic=det)
        user_ctx = self._user_graph_context(
            user_emb, user_cat_mask, user_cat_indices, news_ctx,
            num_categories, deterministic=det,
        )

        for i in range(self.graph_depth):
            news_emb = self._update_graph(
                i, news_emb, news_graph, user_ctx,
                self.n_W, self.n_ffn1, self.n_ffn2, self.n_ffn3, self.n_a,
                deterministic=det,
            )
            user_emb = self._update_graph(
                i, user_emb, user_graph, news_ctx,
                self.u_W, self.u_ffn1, self.u_ffn2, self.u_ffn3, self.u_a,
                deterministic=det,
            )
            news_ctx = news_ctx + self._news_graph_context(
                news_emb, news_mask, deterministic=det,
            )
            user_ctx = user_ctx + self._user_graph_context(
                user_emb, user_cat_mask, user_cat_indices, news_ctx,
                num_categories, deterministic=det,
            )

        return news_ctx, user_ctx


# ======================================================================
# Full DIGAT model
# ======================================================================


class DIGAT(BaseModel):
    """DIGAT: Dual Interactive Graph Attention Networks.

    Training: encode candidates (with SAG) + history → dual graph
    interaction → dot-product scoring.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: DIGATConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if config is None:
            config = DIGATConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        self.num_categories = num_categories + 1  # +1 for padding category

        vocab_size = int(processed_news["vocab_size"])
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        self.word_embedding = nnx.Embed(
            num_embeddings=vocab_size, features=config.embedding_size, rngs=rngs,
        )
        self.word_embedding.embedding.value = jnp.asarray(embeddings_matrix)

        self.news_encoder = DIGATNewsEncoder(config, self.word_embedding, rngs=rngs)

        self.graph_encoder = DIGATGraphEncoder(
            config, self.num_categories, rngs=rngs,
        )

        # No user encoder in the evaluation adapter sense — DIGAT's
        # evaluation goes through ``score_digat_impression`` directly.
        # The BaseModel contract fields are set to satisfy attribute
        # access; the standard evaluator is not used for DIGAT.
        self.user_encoder = None

        self.news_graph_size = config.news_graph_size
        self.max_history = config.max_history_length
        self.D = config.news_embedding_dim

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Training forward pass.

        Expected input keys:
            hist_tokens: (batch_size, hist_len, title_len)
            hist_mask: (batch_size, hist_len, title_len)
            user_graph: (batch_size, user_graph_size, user_graph_size)
            user_category_mask: (batch_size, num_categories)
            user_category_indices: (batch_size, hist_len)
            cand_tokens: (batch_size, num_cands, sag_size, title_len)
            cand_mask: (batch_size, num_cands, sag_size, title_len)
            cand_graph: (batch_size, num_cands, sag_size, sag_size)
            cand_graph_mask: (batch_size, num_cands, sag_size)

        Returns:
            ``(batch_size, num_cands)`` raw logits.
        """
        batch_size, num_cands = inputs["cand_graph"].shape[:2]
        sag_size = self.news_graph_size
        user_graph_size = self.max_history + self.num_categories
        batch_cands = batch_size * num_cands

        # Flatten candidates: (batch_size, num_cands, sag_size, title_len) → (batch_cands, sag_size, title_len)
        cand_tokens = inputs["cand_tokens"].reshape(batch_cands, sag_size, -1)
        cand_mask = (
            inputs["cand_mask"].reshape(batch_cands, sag_size, -1)
            if "cand_mask" in inputs
            else None
        )
        cand_graph = inputs["cand_graph"].reshape(batch_cands, sag_size, sag_size)
        cand_graph_mask = inputs["cand_graph_mask"].reshape(batch_cands, sag_size)

        # Expand user data: (batch_size, ...) → (batch_cands, ...)
        user_graph = jnp.broadcast_to(
            inputs["user_graph"][:, None],
            (batch_size, num_cands, user_graph_size, user_graph_size),
        ).reshape(batch_cands, user_graph_size, user_graph_size)
        user_cat_mask = jnp.broadcast_to(
            inputs["user_category_mask"][:, None],
            (batch_size, num_cands, self.num_categories),
        ).reshape(batch_cands, self.num_categories)
        user_cat_indices = jnp.broadcast_to(
            inputs["user_category_indices"][:, None],
            (batch_size, num_cands, self.max_history),
        ).reshape(batch_cands, self.max_history)

        # Encode candidate news (each with SAG neighbors)
        cand_emb = self.news_encoder(cand_tokens, cand_mask, training=training)

        # Encode user history
        user_news_emb = self.news_encoder(
            inputs["hist_tokens"], inputs.get("hist_mask"), training=training,
        )
        user_news_emb = jnp.broadcast_to(
            user_news_emb[:, None],
            (batch_size, num_cands, self.max_history, self.D),
        ).reshape(batch_cands, self.max_history, self.D)

        # Dual graph interaction
        news_ctx, user_ctx = self.graph_encoder(
            cand_emb, cand_graph, cand_graph_mask,
            user_news_emb, user_graph, user_cat_mask, user_cat_indices,
            self.num_categories, training=training,
        )

        # Dot-product scoring
        news_ctx = news_ctx.reshape(batch_size, num_cands, self.D)
        user_ctx = user_ctx.reshape(batch_size, num_cands, self.D)
        return jnp.sum(news_ctx * user_ctx, axis=-1)
