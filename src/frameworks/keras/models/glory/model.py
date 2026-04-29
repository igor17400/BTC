"""GLORY (RecSys 2023) -- Keras 3 implementation.

Global Graph-Enhanced Personalized News Recommendations.  Encodes each
candidate / history news with both:

- a **local** text encoder (MHA on title tokens -> additive pooling), and
- a **global** GNN over a pre-built news graph built from user click
  trajectories.

The two views are fused per clicked news, pooled into a user vector via
MHA + attention, and compared with candidate embeddings via dot product.

Structurally identical to the PyTorch version in
:mod:`src.frameworks.pytorch.models.glory.model` -- the ports share
identical module layout, method names, and forward-pass shapes so
evaluation results are reproducible across frameworks.

Reference: Yang et al., "Going Beyond Local: Global Graph-Enhanced
Personalized News Recommendations", RecSys 2023.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
from keras import layers, ops

from src.core.models.configs import GLORYConfig
from src.frameworks.keras.models.base import BaseModel

from .layers import (
    AttentionPooling,
    DotProduct,
    GatedGraphConv,
    MultiHeadAttention,
)


# ======================================================================
# News encoder (local -- no graph)
# ======================================================================


class GLORYNewsEncoder(keras.Model):
    """Local news encoder: word emb -> dropout -> MHA -> LN -> drop -> pool -> LN.

    Consumes a (*, T + E + 1 + 1 + 1) feature tensor where the columns
    are [title tokens (T), entity ids (E), category, subcategory,
    news_index].  Only the title tokens are used here.
    """

    def __init__(
        self,
        config: GLORYConfig,
        word_embedding: layers.Embedding,
        name: str = "glory_news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.word_embedding = word_embedding
        self.news_dim = config.head_num * config.head_dim
        self.title_size = config.title_size
        self.entity_size = config.entity_size

        self.dropout1 = layers.Dropout(config.dropout_rate, name="emb_dropout")
        self.msa = MultiHeadAttention(
            config.word_emb_dim,
            config.word_emb_dim,
            config.word_emb_dim,
            config.head_num,
            config.head_dim,
            name="msa",
        )
        self.layernorm1 = layers.LayerNormalization(name="ln1")
        self.dropout2 = layers.Dropout(config.dropout_rate, name="attn_dropout")
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, name="attn_pool",
        )
        self.layernorm2 = layers.LayerNormalization(name="ln2")

    def call(self, news_input, mask=None, training=None):
        """Encode a batch of news.

        Args:
            news_input: ``(B, N, feature_dim)`` int tensor.
            mask: optional ``(B*N, title_size)`` token mask.
            training: whether in training mode.

        Returns:
            ``(B, N, news_dim)`` news representations.
        """
        B = ops.shape(news_input)[0]
        N = ops.shape(news_input)[1]

        title_tokens = news_input[..., : self.title_size]
        title_tokens = ops.cast(title_tokens, "int32")
        flat_title = ops.reshape(title_tokens, (B * N, self.title_size))

        word_emb = self.dropout1(
            self.word_embedding(flat_title), training=training,
        )  # (B*N, T, E)

        attn_out = self.msa(word_emb, word_emb, word_emb, mask)  # (B*N, T, D)
        attn_out = self.layernorm1(attn_out)
        attn_out = self.dropout2(attn_out, training=training)

        pooled = self.attn_pool(attn_out, mask)  # (B*N, D)
        pooled = self.layernorm2(pooled)

        return ops.reshape(pooled, (B, N, self.news_dim))


# ======================================================================
# Click / User / Candidate encoders
# ======================================================================


class GLORYClickEncoder(keras.layers.Layer):
    """Fuse per-clicked-news (title_emb, graph_emb) via attention pooling."""

    def __init__(self, config: GLORYConfig, **kwargs):
        super().__init__(**kwargs)
        self.news_dim = config.head_num * config.head_dim
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, name="click_attn_pool",
        )

    def call(self, title_emb, graph_emb):
        B = ops.shape(title_emb)[0]
        N = ops.shape(title_emb)[1]

        stacked = ops.stack([title_emb, graph_emb], axis=-2)  # (B, N, 2, D)
        stacked = ops.reshape(stacked, (B * N, 2, self.news_dim))

        fused = self.attn_pool(stacked)  # (B*N, D)
        return ops.reshape(fused, (B, N, self.news_dim))


class GLORYUserEncoder(keras.layers.Layer):
    """Pool a sequence of clicked-news embeddings into a user vector."""

    def __init__(self, config: GLORYConfig, **kwargs):
        super().__init__(**kwargs)
        self.news_dim = config.head_num * config.head_dim
        self.msa = MultiHeadAttention(
            self.news_dim,
            self.news_dim,
            self.news_dim,
            config.head_num,
            config.head_dim,
            name="user_msa",
        )
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, name="user_attn_pool",
        )

    def call(self, clicked_news, mask=None):
        h = self.msa(clicked_news, clicked_news, clicked_news, mask)
        return self.attn_pool(h, mask)


class GLORYCandidateEncoder(keras.layers.Layer):
    """Candidate encoder -- Linear + LeakyReLU (no entity path in v1)."""

    def __init__(self, config: GLORYConfig, **kwargs):
        super().__init__(**kwargs)
        self.news_dim = config.head_num * config.head_dim
        self.linear = layers.Dense(self.news_dim, name="cand_linear")

    def call(self, cand_emb):
        return ops.leaky_relu(self.linear(cand_emb), negative_slope=0.2)


# ======================================================================
# Full GLORY model
# ======================================================================


class GLORY(BaseModel):
    """GLORY: local news encoder + global GNN + user fusion + dot scoring."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: GLORYConfig | None = None,
        name: str = "glory",
        **config_overrides,
    ):
        super().__init__(name=name)

        if config is None:
            config = GLORYConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        self.news_dim = config.head_num * config.head_dim
        self.his_size = config.max_history_length

        vocab_size = int(processed_news["vocab_size"])
        embeddings_matrix = np.asarray(processed_news["embeddings"])

        self.word_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.word_emb_dim,
            embeddings_initializer=keras.initializers.Constant(embeddings_matrix),
            trainable=True,
            name="word_embedding",
        )

        self.local_news_encoder = GLORYNewsEncoder(
            config, self.word_embedding,
        )
        self.global_news_encoder = GatedGraphConv(
            self.news_dim,
            num_layers=config.gnn_num_layers,
            aggr="add",
            name="gated_graph_conv",
        )
        self.click_encoder = GLORYClickEncoder(config, name="click_encoder")
        self._user_encoder = GLORYUserEncoder(config, name="user_encoder")
        self.candidate_encoder = GLORYCandidateEncoder(config, name="candidate_encoder")
        self.click_predictor = DotProduct(name="dot_product")

        # BaseModel contract -- news_encoder / user_encoder names satisfy
        # the shared evaluator for standard models.  GLORY's eval path is
        # custom (pre-computes global embeddings once per epoch), so these
        # exist for introspection.
        self.news_encoder = self.local_news_encoder
        self.user_encoder = self._user_encoder

    def call(self, inputs, training=None):
        """Training forward pass.

        Expected input keys:
            subgraph_x: ``(total_nodes, feature_dim)`` concatenated
                node features across all subgraphs in the batch.
            subgraph_edge_index: ``(2, total_edges)`` concatenated edges
                with per-sample offsets already applied.
            mapping_idx: ``(B, his_size)`` int -- for each user's
                history slot, the node index in ``subgraph_x``
                (``-1`` = padding).
            cand_tokens: ``(B, C, feature_dim)`` candidate features.

        Returns:
            ``(B, C)`` raw logits.
        """
        subgraph_x = inputs["subgraph_x"]
        edge_index = inputs["subgraph_edge_index"]
        mapping_idx = inputs["mapping_idx"]
        cand_tokens = inputs["cand_tokens"]

        # Valid-history mask (1 where mapping_idx != -1).
        valid = ops.not_equal(mapping_idx, -1)
        mapping = ops.where(valid, mapping_idx, 0)  # safe index

        # Encode every subgraph node once with the local encoder.
        flat = ops.expand_dims(subgraph_x, axis=0)  # (1, N_total, feat)
        x_encoded = ops.squeeze(
            self.local_news_encoder(flat, training=training), axis=0,
        )  # (N_total, D)

        # Build padding mask for GNN.  The JAX collate pads to fixed
        # tensor shapes for JIT; padded edges are self-loops on the last
        # node.  Without masking inside the GNN, the GRU's learned biases
        # produce non-zero padding-node states that get amplified ~718k×
        # through self-loop scatter-add, causing NaN after a few epochs.
        real_mask = None
        if "num_real_nodes" in inputs:
            node_idx = ops.arange(ops.shape(x_encoded)[0])
            real_mask = ops.expand_dims(
                node_idx < inputs["num_real_nodes"], axis=-1,
            )  # (N_total, 1)
            x_encoded = ops.where(real_mask, x_encoded, 0)

        # GNN over the full (batched) subgraph.
        graph_emb = self.global_news_encoder(
            x_encoded, edge_index, real_mask=real_mask,
        )  # (N_total, D)

        # Gather history embeddings from both views.
        valid_mask = ops.expand_dims(valid, axis=-1)  # (B, H, 1)
        clicked_title = ops.where(
            valid_mask, ops.take(x_encoded, mapping, axis=0), 0,
        )  # (B, H, D)
        clicked_graph = ops.where(
            valid_mask, ops.take(graph_emb, mapping, axis=0), 0,
        )  # (B, H, D)

        # Fuse -> pool into user vector.
        fused = self.click_encoder(clicked_title, clicked_graph)  # (B, H, D)
        user_emb = self._user_encoder(
            fused, ops.cast(valid, "float32"),
        )  # (B, D)

        # Candidates: encode locally then project.
        cand_local = self.local_news_encoder(
            cand_tokens, training=training,
        )  # (B, C, D)
        cand_final = self.candidate_encoder(cand_local)  # (B, C, D)

        # Dot-product scoring.
        return self.click_predictor(cand_final, user_emb)  # (B, C)
