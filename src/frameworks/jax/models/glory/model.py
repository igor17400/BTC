"""GLORY (RecSys 2023) — JAX/Flax NNX implementation.

Global Graph-Enhanced Personalized News Recommendations.  Encodes each
candidate / history news with both:

- a **local** text encoder (MHA on title tokens → additive pooling), and
- a **global** GNN over a pre-built news graph built from user click
  trajectories.

The two views are fused per clicked news, pooled into a user vector via
MHA + attention, and compared with candidate embeddings via dot product.

Structurally identical to the PyTorch version in
:mod:`src.frameworks.pytorch.models.glory.model` — the two ports share
identical module layout, method names, and forward-pass shapes so
evaluation results are reproducible across frameworks.

Reference: Yang et al., "Going Beyond Local: Global Graph-Enhanced
Personalized News Recommendations", RecSys 2023.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import GLORYConfig

from ..base import BaseModel
from .layers import (
    AttentionPooling,
    GatedGraphConv,
    MultiHeadAttention,
)


# ======================================================================
# News encoder (local — no graph)
# ======================================================================


class GLORYNewsEncoder(nnx.Module):
    """Local news encoder: word emb → dropout → MHA → LN → drop → pool → LN.

    Consumes a (*, T + E + 1 + 1 + 1) feature tensor where the columns
    are [title tokens (T), entity ids (E), category, subcategory,
    news_index].  Only the title tokens are used here.
    """

    def __init__(
        self,
        config: GLORYConfig,
        word_embedding: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.word_embedding = word_embedding
        self.news_dim = config.head_num * config.head_dim
        self.title_size = config.title_size
        self.entity_size = config.entity_size

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.msa = MultiHeadAttention(
            config.word_emb_dim,
            config.word_emb_dim,
            config.word_emb_dim,
            config.head_num,
            config.head_dim,
            rngs=rngs,
        )
        self.layernorm1 = nnx.LayerNorm(self.news_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, rngs=rngs,
        )
        self.layernorm2 = nnx.LayerNorm(self.news_dim, rngs=rngs)

    def __call__(
        self,
        news_input: jax.Array,
        mask: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode a batch of news.

        Args:
            news_input: ``(B, N, feature_dim)`` int tensor — feature_dim
                = ``title_size + entity_size + 3``.
            mask: optional ``(B*N, title_size)`` token mask.

        Returns:
            ``(B, N, news_dim)`` news representations.
        """
        det = not training
        B, N = news_input.shape[:2]
        title_tokens = news_input[..., : self.title_size].astype(jnp.int32)
        flat_title = title_tokens.reshape(B * N, self.title_size)

        word_emb = self.dropout1(
            self.word_embedding(flat_title), deterministic=det,
        )  # (B*N, T, E)

        attn_out = self.msa(word_emb, word_emb, word_emb, mask)  # (B*N, T, D)
        attn_out = self.layernorm1(attn_out)
        attn_out = self.dropout2(attn_out, deterministic=det)

        pooled = self.attn_pool(attn_out, mask)  # (B*N, D)
        pooled = self.layernorm2(pooled)

        return pooled.reshape(B, N, self.news_dim)


# ======================================================================
# Click / User / Candidate encoders
# ======================================================================


class GLORYClickEncoder(nnx.Module):
    """Fuse per-clicked-news (title_emb, graph_emb) via attention pooling."""

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, rngs=rngs,
        )

    def __call__(
        self,
        title_emb: jax.Array,   # (B, N, D)
        graph_emb: jax.Array,   # (B, N, D)
    ) -> jax.Array:
        B, N = title_emb.shape[:2]
        stacked = jnp.stack([title_emb, graph_emb], axis=-2)  # (B, N, 2, D)
        stacked = stacked.reshape(B * N, 2, self.news_dim)
        fused = self.attn_pool(stacked)  # (B*N, D)
        return fused.reshape(B, N, self.news_dim)


class GLORYUserEncoder(nnx.Module):
    """Pool a sequence of clicked-news embeddings into a user vector."""

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.msa = MultiHeadAttention(
            self.news_dim,
            self.news_dim,
            self.news_dim,
            config.head_num,
            config.head_dim,
            rngs=rngs,
        )
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, rngs=rngs,
        )

    def __call__(
        self,
        clicked_news: jax.Array,  # (B, H, D)
        mask: jax.Array | None = None,
    ) -> jax.Array:
        h = self.msa(clicked_news, clicked_news, clicked_news, mask)
        return self.attn_pool(h, mask)


class GLORYCandidateEncoder(nnx.Module):
    """Candidate encoder — Linear + LeakyReLU (no entity path in v1)."""

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.linear = nnx.Linear(self.news_dim, self.news_dim, rngs=rngs)

    def __call__(self, cand_emb: jax.Array) -> jax.Array:
        return jax.nn.leaky_relu(self.linear(cand_emb), negative_slope=0.2)


# ======================================================================
# Full GLORY model
# ======================================================================


class GLORY(BaseModel):
    """GLORY: local news encoder + global GNN + user fusion + dot scoring."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: GLORYConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if config is None:
            config = GLORYConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        self.news_dim = config.head_num * config.head_dim
        self.his_size = config.max_history_length

        vocab_size = int(processed_news["vocab_size"])
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        self.word_embedding = nnx.Embed(
            num_embeddings=vocab_size, features=config.word_emb_dim, rngs=rngs,
        )
        self.word_embedding.embedding.value = jnp.asarray(embeddings_matrix)

        self.local_news_encoder = GLORYNewsEncoder(
            config, self.word_embedding, rngs=rngs,
        )
        self.global_news_encoder = GatedGraphConv(
            self.news_dim,
            num_layers=config.gnn_num_layers,
            aggr="add",
            rngs=rngs,
        )
        self.click_encoder = GLORYClickEncoder(config, rngs=rngs)
        self.user_encoder = GLORYUserEncoder(config, rngs=rngs)
        self.candidate_encoder = GLORYCandidateEncoder(config, rngs=rngs)

        # BaseModel contract — GLORY's eval path is custom.
        self.news_encoder = self.local_news_encoder

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = True,
    ) -> jax.Array:
        """Training forward pass.

        Expected input keys:
            subgraph_x: ``(total_nodes, feature_dim)`` concatenated
                node features across all subgraphs in the batch.
            subgraph_edge_index: ``(2, total_edges)`` concatenated edges
                with per-sample offsets already applied.
            mapping_idx: ``(B, his_size)`` int — for each user's
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
        valid = mapping_idx != -1
        mapping = jnp.where(valid, mapping_idx, 0)  # safe index

        # Encode every subgraph node once with the local encoder.
        flat = jnp.expand_dims(subgraph_x, axis=0)  # (1, N_total, feat)
        x_encoded = self.local_news_encoder(
            flat, training=training,
        ).squeeze(axis=0)  # (N_total, D)

        # GNN over the full (batched) subgraph.
        graph_emb = self.global_news_encoder(x_encoded, edge_index)  # (N_total, D)

        # Gather history embeddings from both views.
        valid_mask = jnp.expand_dims(valid, axis=-1)  # (B, H, 1)
        clicked_title = jnp.where(valid_mask, x_encoded[mapping], 0)   # (B, H, D)
        clicked_graph = jnp.where(valid_mask, graph_emb[mapping], 0)   # (B, H, D)

        # Fuse → pool into user vector.
        fused = self.click_encoder(clicked_title, clicked_graph)  # (B, H, D)
        user_emb = self.user_encoder(fused, valid.astype(jnp.float32))  # (B, D)

        # Candidates: encode locally then project.
        cand_local = self.local_news_encoder(cand_tokens, training=training)  # (B, C, D)
        cand_final = self.candidate_encoder(cand_local)  # (B, C, D)

        # Dot-product scoring.
        return jnp.sum(cand_final * jnp.expand_dims(user_emb, axis=1), axis=-1)  # (B, C)
