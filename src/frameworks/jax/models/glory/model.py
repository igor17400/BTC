"""GLORY (RecSys 2023) — JAX/Flax NNX implementation.

Global Graph-Enhanced Personalized News Recommendations.  Encodes each
candidate / history news with both:

- a **local** text encoder (MHA on title tokens → additive pooling), and
- a **global** GNN over a pre-built news graph built from user click
  trajectories.
- (optional) **entity** encoders over knowledge-graph entity embeddings.

The two (or three) views are fused per clicked news, pooled into a user
vector via MHA + attention, and compared with candidate embeddings via
dot product.

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
    EntityEncoder,
    GatedGraphConv,
    GlobalEntityEncoder,
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
        det = not training
        B, N = news_input.shape[:2]
        title_tokens = news_input[..., : self.title_size].astype(jnp.int32)
        flat_title = title_tokens.reshape(B * N, self.title_size)

        word_emb = self.dropout1(
            self.word_embedding(flat_title), deterministic=det,
        )

        attn_out = self.msa(word_emb, word_emb, word_emb, mask)
        attn_out = self.layernorm1(attn_out)
        attn_out = self.dropout2(attn_out, deterministic=det)

        pooled = self.attn_pool(attn_out, mask)
        pooled = self.layernorm2(pooled)

        return pooled.reshape(B, N, self.news_dim)


# ======================================================================
# Click / User / Candidate encoders
# ======================================================================


class GLORYClickEncoder(nnx.Module):
    """Fuse per-clicked-news views via attention pooling.

    Stacks 2 views (title, graph) or 3 views (title, graph, entity)
    depending on whether entity_emb is provided.
    """

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.attn_pool = AttentionPooling(
            self.news_dim, config.attention_hidden_dim, rngs=rngs,
        )

    def __call__(
        self,
        title_emb: jax.Array,   # (B, N, D)
        graph_emb: jax.Array,   # (B, N, D)
        entity_emb: jax.Array | None = None,  # (B, N, D)
    ) -> jax.Array:
        B, N = title_emb.shape[:2]
        if entity_emb is not None:
            stacked = jnp.stack(
                [title_emb, graph_emb, entity_emb], axis=-2,
            )  # (B, N, 3, D)
            num_views = 3
        else:
            stacked = jnp.stack(
                [title_emb, graph_emb], axis=-2,
            )  # (B, N, 2, D)
            num_views = 2
        stacked = stacked.reshape(B * N, num_views, self.news_dim)
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
    """Candidate encoder.

    Without entities: Linear + LeakyReLU on title embeddings.
    With entities: stack [title, origin_entity, neighbor_entity] →
    AttentionPooling → Linear + LeakyReLU.
    """

    def __init__(self, config: GLORYConfig, *, rngs: nnx.Rngs):
        self.news_dim = config.head_num * config.head_dim
        self.use_entity = config.use_entity
        if self.use_entity:
            self.attn_pool = AttentionPooling(
                self.news_dim, config.attention_hidden_dim, rngs=rngs,
            )
        self.linear = nnx.Linear(self.news_dim, self.news_dim, rngs=rngs)

    def __call__(
        self,
        cand_emb: jax.Array,
        origin_entity_emb: jax.Array | None = None,
        neighbor_entity_emb: jax.Array | None = None,
    ) -> jax.Array:
        if self.use_entity and origin_entity_emb is not None and neighbor_entity_emb is not None:
            B, C = cand_emb.shape[:2]
            stacked = jnp.stack(
                [cand_emb, origin_entity_emb, neighbor_entity_emb], axis=-2,
            )  # (B, C, 3, D)
            stacked = stacked.reshape(B * C, 3, self.news_dim)
            pooled = self.attn_pool(stacked)  # (B*C, D)
            cand_emb = pooled.reshape(B, C, self.news_dim)
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
        self.use_entity = config.use_entity

        self.news_dim = config.head_num * config.head_dim
        self.his_size = config.max_history_length
        self.title_size = config.title_size
        self.entity_size = config.entity_size

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

        # Entity path (optional).
        if self.use_entity:
            entity_emb = processed_news.get("entity_embeddings")
            if entity_emb is not None:
                entity_emb = np.asarray(entity_emb, dtype=np.float32)
                entity_vocab = entity_emb.shape[0]
                self.entity_emb_dim = entity_emb.shape[1]
            else:
                entity_vocab = 1
                self.entity_emb_dim = config.entity_emb_dim
                entity_emb = np.zeros(
                    (entity_vocab, self.entity_emb_dim), dtype=np.float32,
                )
            self.entity_embedding = nnx.Embed(
                num_embeddings=entity_vocab,
                features=self.entity_emb_dim,
                rngs=rngs,
            )
            self.entity_embedding.embedding.value = jnp.asarray(entity_emb)

            self.local_entity_encoder = EntityEncoder(
                entity_dim=self.entity_emb_dim,
                news_dim=self.news_dim,
                head_dim=config.head_dim,
                attention_hidden_dim=config.attention_hidden_dim,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
            self.global_entity_encoder = GlobalEntityEncoder(
                entity_dim=self.entity_emb_dim,
                head_num=config.head_num,
                head_dim=config.head_dim,
                attention_hidden_dim=config.attention_hidden_dim,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )

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
            candidate_entity: (optional) ``(B, C, entity_size +
                entity_size * entity_neighbors)`` entity + neighbor IDs.
            entity_mask: (optional) ``(B, C, entity_size *
                entity_neighbors)`` mask for valid neighbor entities.

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

        # Zero out padding nodes before GNN (JAX fixed-shape padding).
        if "num_real_nodes" in inputs:
            node_idx = jnp.arange(x_encoded.shape[0])
            real_mask = jnp.expand_dims(
                node_idx < inputs["num_real_nodes"], axis=-1,
            )
            x_encoded = jnp.where(real_mask, x_encoded, 0)

        # GNN over the full (batched) subgraph.
        graph_emb = self.global_news_encoder(x_encoded, edge_index)  # (N_total, D)

        # Gather history embeddings from both views.
        valid_mask = jnp.expand_dims(valid, axis=-1)  # (B, H, 1)
        clicked_title = jnp.where(valid_mask, x_encoded[mapping], 0)   # (B, H, D)
        clicked_graph = jnp.where(valid_mask, graph_emb[mapping], 0)   # (B, H, D)

        # Clicked entity encoding (optional).
        clicked_entity_emb = None
        if self.use_entity:
            # Extract entity IDs for clicked news from subgraph features.
            clicked_entity_ids = subgraph_x[
                mapping, self.title_size : self.title_size + self.entity_size,
            ].astype(jnp.int32)  # (B, H, entity_size)
            clicked_entity_ids = jnp.where(valid_mask, clicked_entity_ids, 0)
            entity_embedded = self.entity_embedding(clicked_entity_ids)  # (B, H, E, entity_dim)
            clicked_entity_emb = self.local_entity_encoder(
                entity_embedded, training=training,
            )  # (B, H, D)

        # Fuse → pool into user vector.
        fused = self.click_encoder(
            clicked_title, clicked_graph, clicked_entity_emb,
        )  # (B, H, D)
        user_emb = self.user_encoder(fused, valid.astype(jnp.float32))  # (B, D)

        # Candidates: encode locally then project.
        cand_local = self.local_news_encoder(cand_tokens, training=training)  # (B, C, D)

        # Candidate entity encoding (optional).
        cand_origin_emb = None
        cand_neighbor_emb = None
        if self.use_entity and "candidate_entity" in inputs:
            cand_entity = inputs["candidate_entity"].astype(jnp.int32)
            entity_mask = inputs.get("entity_mask")

            E = self.entity_size
            origin_ids = cand_entity[..., :E]  # (B, C, E)
            neighbor_ids = cand_entity[..., E:]  # (B, C, E*neighbors)

            origin_embedded = self.entity_embedding(origin_ids)  # (B, C, E, dim)
            cand_origin_emb = self.local_entity_encoder(
                origin_embedded, training=training,
            )  # (B, C, D)

            B, C = neighbor_ids.shape[:2]
            n_neighbors = neighbor_ids.shape[-1]
            neighbor_embedded = self.entity_embedding(
                neighbor_ids,
            )  # (B, C, E*neighbors, dim)

            # Reshape mask for GlobalEntityEncoder.
            ent_mask = None
            if entity_mask is not None:
                ent_mask = entity_mask.reshape(B * C, n_neighbors)

            cand_neighbor_emb = self.global_entity_encoder(
                neighbor_embedded, ent_mask, training=training,
            )  # (B, C, D)

        cand_final = self.candidate_encoder(
            cand_local, cand_origin_emb, cand_neighbor_emb,
        )  # (B, C, D)

        # Dot-product scoring.
        return jnp.sum(cand_final * jnp.expand_dims(user_emb, axis=1), axis=-1)  # (B, C)
