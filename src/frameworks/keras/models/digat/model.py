"""DIGAT (EMNLP 2022 Findings) — Keras 3 implementation.

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

import keras
from keras import layers, ops

from src.core.models.configs import DIGATConfig

from ..base import BaseModel
from .layers import (
    DIGATAdditiveAttention,
    MultiHeadSelfAttention,
    ScaledDotProductAttention,
    scatter_softmax,
    scatter_sum,
)


# ======================================================================
# News encoder
# ======================================================================


class DIGATNewsEncoder(keras.Model):
    """MSA-based news encoder: embedding -> MHA -> additive attention."""

    def __init__(
        self,
        config: DIGATConfig,
        word_embedding: layers.Embedding,
        name: str = "digat_news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.word_embedding = word_embedding
        self.dropout = layers.Dropout(config.dropout_rate, seed=config.seed, name="emb_dropout")
        self.msa = MultiHeadSelfAttention(
            config.embedding_size,
            config.msa_head_num,
            config.msa_head_dim,
            name="msa",
        )
        self.news_embedding_dim = config.news_embedding_dim
        self.attention = DIGATAdditiveAttention(
            self.news_embedding_dim,
            config.attention_dim,
            name="news_additive_attention",
        )

    def call(self, title_text, title_mask=None, training=None):
        """Encode a batch of news with SAG neighbor structure.

        Args:
            title_text: ``(batch_size, num_news, title_len)`` token ids.
            title_mask: ``(batch_size, num_news, title_len)`` float mask
                (1 = valid token).

        Returns:
            ``(batch_size, num_news, news_embedding_dim)`` news reprs.
        """
        batch_size = ops.shape(title_text)[0]
        num_news = ops.shape(title_text)[1]
        title_len = ops.shape(title_text)[2]

        flat_text = ops.reshape(title_text, (batch_size * num_news, title_len))
        flat_mask = (
            ops.reshape(title_mask, (batch_size * num_news, title_len))
            if title_mask is not None
            else None
        )

        w = self.dropout(self.word_embedding(flat_text), training=training)
        h = ops.relu(self.msa(w))
        news_repr = self.attention(h, mask=flat_mask)
        return ops.reshape(news_repr, (batch_size, num_news, self.news_embedding_dim))


# ======================================================================
# Dual graph encoder
# ======================================================================


class DIGATGraphEncoder(keras.Model):
    """Dual Interactive Graph Attention encoder.

    Maintains a news-graph channel (SAG subgraph per candidate) and a
    user-graph channel (history + topic nodes).  The two channels
    interact across ``graph_depth`` layers: each graph's attention
    incorporates the other channel's context vector.
    """

    def __init__(self, config: DIGATConfig, num_categories: int, name: str = "digat_graph_encoder"):
        super().__init__(name=name)
        D = config.news_embedding_dim
        depth = config.graph_depth
        self.graph_depth = depth
        self.max_history = config.max_history_length
        self.D = D
        self.scale = math.sqrt(float(D))

        # topic_dropout  -- full rate
        # attn_dropout   -- full rate
        # input_dropout  -- half rate (mirrors reference dropout__)
        self.topic_dropout = layers.Dropout(config.dropout_rate, seed=config.seed, name="topic_dropout")
        self.attn_dropout = layers.Dropout(config.dropout_rate, seed=config.seed, name="attn_dropout")
        self.input_dropout = layers.Dropout(config.dropout_rate / 2, seed=config.seed, name="input_dropout")

        # --- News graph context ---
        self.news_ctx_attn = ScaledDotProductAttention(D, D, D, name="news_ctx_attn")
        self.news_ctx_gate = layers.Dense(D, use_bias=True, name="news_ctx_gate")

        # --- User graph context (topic-level scatter + attention) ---
        self.user_news_K = layers.Dense(D, use_bias=False, name="user_news_K")
        self.user_news_Q = layers.Dense(D, use_bias=True, name="user_news_Q")
        self.topic_affine = layers.Dense(D, use_bias=True, name="topic_affine")
        self.user_ctx_attn = ScaledDotProductAttention(D, D, D, name="user_ctx_attn")

        # --- Per-depth news graph update layers ---
        self.n_W = [layers.Dense(D, use_bias=True, name=f"n_W_{i}") for i in range(depth)]
        self.n_ffn1 = [layers.Dense(D, use_bias=False, name=f"n_ffn1_{i}") for i in range(depth)]
        self.n_ffn2 = [layers.Dense(D, use_bias=False, name=f"n_ffn2_{i}") for i in range(depth)]
        self.n_ffn3 = [layers.Dense(D, use_bias=True, name=f"n_ffn3_{i}") for i in range(depth)]
        self.n_a = [layers.Dense(1, use_bias=False, name=f"n_a_{i}") for i in range(depth)]

        # --- Per-depth user graph update layers ---
        self.u_W = [layers.Dense(D, use_bias=True, name=f"u_W_{i}") for i in range(depth)]
        self.u_ffn1 = [layers.Dense(D, use_bias=False, name=f"u_ffn1_{i}") for i in range(depth)]
        self.u_ffn2 = [layers.Dense(D, use_bias=False, name=f"u_ffn2_{i}") for i in range(depth)]
        self.u_ffn3 = [layers.Dense(D, use_bias=True, name=f"u_ffn3_{i}") for i in range(depth)]
        self.u_a = [layers.Dense(1, use_bias=False, name=f"u_a_{i}") for i in range(depth)]

        # Learnable topic node embeddings — added as a weight variable.
        # Initialized in build() once shape is known, but we store the count.
        self._num_categories = num_categories
        self._D = D

    def build(self, input_shape=None):
        self.topic_node_emb = self.add_weight(
            name="topic_node_emb",
            shape=(self._num_categories, self._D),
            initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )
        super().build(input_shape)

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def _news_graph_context(self, emb, mask, training=None):
        """Gated local/global context from a news SAG subgraph.

        Args:
            emb: (B, G, D) node embeddings for the SAG subgraph.
            mask: (B, G) valid-node mask.

        Returns:
            (B, D) context vector.
        """
        local = emb[:, 0, :]
        glob = self.news_ctx_attn(emb, local, mask=mask)
        gate_in = ops.concatenate([local, glob], axis=-1)
        gate = ops.sigmoid(self.input_dropout(self.news_ctx_gate(gate_in), training=training))
        return gate * local + (1 - gate) * glob

    def _user_graph_context(self, emb, cat_mask, cat_indices, news_ctx, num_categories, training=None):
        """Two-level user context: topic-level scatter -> user-level attention.

        Args:
            emb: (B, H+C, D) history news + topic node embeddings.
            cat_mask: (B, C) which categories are active.
            cat_indices: (B, H) topic assignment per history news.
            news_ctx: (B, D) current news context.
            num_categories: C number of category/topic nodes.

        Returns:
            (B, D) user context vector.
        """
        hist = emb[:, : self.max_history, :]  # (B, H, D)

        K = self.user_news_K(hist)
        Q = ops.expand_dims(self.user_news_Q(news_ctx), axis=2)  # (B, D, 1)
        scores = ops.squeeze(ops.matmul(K, Q), axis=2) / self.scale  # (B, H)

        alpha = scatter_softmax(scores, cat_indices, num_categories)  # (B, H)
        weighted = ops.expand_dims(alpha, axis=-1) * hist  # (B, H, D)
        topic_emb = scatter_sum(weighted, cat_indices, dim=1, dim_size=num_categories)
        topic_emb = self.topic_dropout(
            ops.relu(self.topic_affine(topic_emb)) + topic_emb,
            training=training,
        )

        return self.user_ctx_attn(topic_emb, news_ctx, mask=cat_mask)

    # ------------------------------------------------------------------
    # Graph update (one layer)
    # ------------------------------------------------------------------

    def _update_graph(
        self,
        idx,
        emb,
        adj,
        cross_ctx,
        W_list,
        ffn1_list,
        ffn2_list,
        ffn3_list,
        a_list,
        training=None,
    ):
        """Single cross-interactive graph attention update."""
        batch_size = ops.shape(emb)[0]
        emb = self.input_dropout(emb, training=training)
        h = W_list[idx](emb)

        K1 = ops.expand_dims(ffn1_list[idx](emb), axis=1)  # (B, 1, N, D)
        K2 = ops.expand_dims(ffn2_list[idx](emb), axis=2)  # (B, N, 1, D)
        K3 = ops.reshape(ffn3_list[idx](cross_ctx), (batch_size, 1, 1, self.D))

        scores = ops.squeeze(
            a_list[idx](ops.relu(K1 + K2 + K3)),
            axis=-1,
        )  # (B, N, N)
        scores = ops.leaky_relu(scores, negative_slope=0.2)
        scores = ops.where(ops.equal(adj, 0), -1e9, scores)

        alpha = self.attn_dropout(
            ops.softmax(scores, axis=2),
            training=training,
        )

        return ops.relu(ops.matmul(alpha, h)) + emb

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def call(self, inputs, training=None):
        """Dual graph interaction forward.

        Args:
            inputs: tuple of
                (news_emb, news_graph, news_mask, user_news_emb,
                 user_graph, user_cat_mask, user_cat_indices)

                news_emb: (B, G_n, D) SAG node embeddings per candidate.
                news_graph: (B, G_n, G_n) SAG adjacency.
                news_mask: (B, G_n) valid SAG nodes.
                user_news_emb: (B, H, D) history news embeddings.
                user_graph: (B, G_u, G_u) user graph adjacency.
                user_cat_mask: (B, C) active categories.
                user_cat_indices: (B, H) topic per history item.

        Returns:
            (news_ctx, user_ctx) -- each (B, D).
        """
        (news_emb, news_graph, news_mask, user_news_emb,
         user_graph, user_cat_mask, user_cat_indices) = inputs
        num_categories = self._num_categories

        batch_size = ops.shape(news_emb)[0]
        # topic_node_emb: (C, D) -> broadcast to (B, C, D)
        topic_nodes = ops.broadcast_to(
            ops.expand_dims(self.topic_node_emb, axis=0),
            (batch_size, num_categories, self.D),
        )
        user_emb = ops.concatenate(
            [user_news_emb, self.input_dropout(topic_nodes, training=training)],
            axis=1,
        )

        news_ctx = self._news_graph_context(news_emb, news_mask, training=training)
        user_ctx = self._user_graph_context(
            user_emb, user_cat_mask, user_cat_indices, news_ctx, num_categories,
            training=training,
        )

        for i in range(self.graph_depth):
            news_emb = self._update_graph(
                i, news_emb, news_graph, user_ctx,
                self.n_W, self.n_ffn1, self.n_ffn2, self.n_ffn3, self.n_a,
                training=training,
            )
            user_emb = self._update_graph(
                i, user_emb, user_graph, news_ctx,
                self.u_W, self.u_ffn1, self.u_ffn2, self.u_ffn3, self.u_a,
                training=training,
            )
            news_ctx = news_ctx + self._news_graph_context(
                news_emb, news_mask, training=training,
            )
            user_ctx = user_ctx + self._user_graph_context(
                user_emb, user_cat_mask, user_cat_indices, news_ctx, num_categories,
                training=training,
            )

        return news_ctx, user_ctx


# ======================================================================
# Full DIGAT model
# ======================================================================


class DIGAT(BaseModel):
    """DIGAT: Dual Interactive Graph Attention Networks.

    Training: encode candidates (with SAG) + history -> dual graph
    interaction -> dot-product scoring.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: DIGATConfig | None = None,
        name: str = "digat",
        **config_overrides,
    ):
        super().__init__(name=name)

        if config is None:
            config = DIGATConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news

        # BaseModel contract
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        self.num_categories = num_categories + 1  # +1 for padding category

        vocab_size = int(processed_news["vocab_size"])

        self.word_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(
                processed_news["embeddings"]
            ),
            trainable=True,
            name="word_embedding",
        )

        self.news_encoder = DIGATNewsEncoder(config, self.word_embedding)

        self.graph_encoder = DIGATGraphEncoder(config, self.num_categories)
        # Force build so topic_node_emb weight is created
        self.graph_encoder.build(None)

        # No user encoder for DIGAT — evaluation goes through custom evaluator
        self.user_encoder = None

        self.news_graph_size = config.news_graph_size
        self.max_history = config.max_history_length
        self.D = config.news_embedding_dim

    def call(self, inputs, training=None):
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
            (batch_size, num_cands) raw logits.
        """
        batch_size = ops.shape(inputs["cand_graph"])[0]
        num_cands = ops.shape(inputs["cand_graph"])[1]
        sag_size = self.news_graph_size
        user_graph_size = self.max_history + self.num_categories
        batch_cands = batch_size * num_cands

        # Flatten candidates: (B, C, G, T) -> (B*C, G, T)
        cand_tokens = ops.reshape(inputs["cand_tokens"], (batch_cands, sag_size, -1))
        cand_mask = (
            ops.reshape(inputs["cand_mask"], (batch_cands, sag_size, -1))
            if "cand_mask" in inputs
            else None
        )
        cand_graph = ops.reshape(inputs["cand_graph"], (batch_cands, sag_size, sag_size))
        cand_graph_mask = ops.reshape(inputs["cand_graph_mask"], (batch_cands, sag_size))

        # Expand user data: (B, ...) -> (B*C, ...)
        user_graph = ops.reshape(
            ops.broadcast_to(
                ops.expand_dims(inputs["user_graph"], axis=1),
                (batch_size, num_cands, user_graph_size, user_graph_size),
            ),
            (batch_cands, user_graph_size, user_graph_size),
        )
        user_cat_mask = ops.reshape(
            ops.broadcast_to(
                ops.expand_dims(inputs["user_category_mask"], axis=1),
                (batch_size, num_cands, self.num_categories),
            ),
            (batch_cands, self.num_categories),
        )
        user_cat_indices = ops.reshape(
            ops.broadcast_to(
                ops.expand_dims(inputs["user_category_indices"], axis=1),
                (batch_size, num_cands, self.max_history),
            ),
            (batch_cands, self.max_history),
        )

        # Encode candidate news (each with SAG neighbors)
        cand_emb = self.news_encoder(cand_tokens, cand_mask, training=training)

        # Encode user history
        user_news_emb = self.news_encoder(
            inputs["hist_tokens"],
            inputs.get("hist_mask"),
            training=training,
        )
        user_news_emb = ops.reshape(
            ops.broadcast_to(
                ops.expand_dims(user_news_emb, axis=1),
                (batch_size, num_cands, self.max_history, self.D),
            ),
            (batch_cands, self.max_history, self.D),
        )

        # Dual graph interaction
        news_ctx, user_ctx = self.graph_encoder(
            (cand_emb, cand_graph, cand_graph_mask,
             user_news_emb, user_graph, user_cat_mask, user_cat_indices),
            training=training,
        )

        # Dot-product scoring
        news_ctx = ops.reshape(news_ctx, (batch_size, num_cands, self.D))
        user_ctx = ops.reshape(user_ctx, (batch_size, num_cands, self.D))
        return ops.sum(news_ctx * user_ctx, axis=-1)  # (B, C)
