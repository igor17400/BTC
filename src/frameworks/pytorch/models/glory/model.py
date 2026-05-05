"""GLORY (RecSys 2023) — PyTorch implementation.

Global Graph-Enhanced Personalized News Recommendations.  Encodes each
candidate / history news with both:

- a **local** text encoder (MHA on title tokens → additive pooling), and
- a **global** GNN over a pre-built news graph built from user click
  trajectories.

The two views are fused per clicked news, pooled into a user vector via
MHA + attention, and compared with candidate embeddings via dot product.

No ``torch_geometric`` dependency — all graph ops live in
:mod:`.layers`.  This keeps the model parallel to
:mod:`src.frameworks.pytorch.models.digat` and makes the future JAX /
Keras ports direct translations.

Reference: Yang et al., "Going Beyond Local: Global Graph-Enhanced
Personalized News Recommendations", RecSys 2023.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.core.models.configs import GLORYConfig

from ..base import BaseModel
from .layers import (
    AttentionPooling,
    DotProduct,
    EntityEncoder,
    GatedGraphConv,
    GlobalEntityEncoder,
    MultiHeadAttention,
)

# ======================================================================
# News encoder (local — no graph)
# ======================================================================


class GLORYNewsEncoder(nn.Module):
    """Local news encoder: word emb → dropout → MHA → LN → drop → pool → LN.

    Consumes a (*, T + E + 1 + 1 + 1) feature tensor where the columns
    are [title tokens (T), entity ids (E), category, subcategory,
    news_index].  Only the title tokens are used here — entity /
    category features are consumed downstream.
    """

    def __init__(
        self,
        config: GLORYConfig,
        word_embedding: nn.Embedding,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.news_dim = config.head_num * config.head_dim
        self.title_size = config.title_size
        self.entity_size = config.entity_size

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.msa = MultiHeadAttention(
            config.word_emb_dim,
            config.word_emb_dim,
            config.word_emb_dim,
            config.head_num,
            config.head_dim,
        )
        self.layernorm1 = nn.LayerNorm(self.news_dim)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.attn_pool = AttentionPooling(self.news_dim, config.attention_hidden_dim)
        self.layernorm2 = nn.LayerNorm(self.news_dim)

    def forward(
        self,
        news_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of news.

        Args:
            news_input: ``(B, N, feature_dim)`` int tensor — feature_dim
                = ``title_size + entity_size + 3``.
            mask: optional ``(B*N, title_size)`` token mask.

        Returns:
            ``(B, N, news_dim)`` news representations.
        """
        B, N = news_input.shape[:2]
        # Columns: [title | entity | cat | sub | news_idx]
        title_tokens = news_input[..., : self.title_size]
        flat_title = title_tokens.reshape(B * N, self.title_size).long()

        word_emb = self.dropout1(self.word_embedding(flat_title))  # (B*N, T, E)

        attn_out = self.msa(word_emb, word_emb, word_emb, mask)  # (B*N, T, D)
        attn_out = self.layernorm1(attn_out)
        attn_out = self.dropout2(attn_out)

        pooled = self.attn_pool(attn_out, mask)  # (B*N, D)
        pooled = self.layernorm2(pooled)

        return pooled.view(B, N, self.news_dim)


# ======================================================================
# Click / User / Candidate encoders
# ======================================================================


class GLORYClickEncoder(nn.Module):
    """Fuse per-clicked-news views via attention pooling.

    Stacks 2 views (title, graph) or 3 views (title, graph, entity)
    depending on whether ``entity_emb`` is provided.
    """

    def __init__(self, config: GLORYConfig):
        super().__init__()
        self.news_dim = config.head_num * config.head_dim
        self.attn_pool = AttentionPooling(self.news_dim, config.attention_hidden_dim)

    def forward(
        self,
        title_emb: torch.Tensor,  # (B, N, D)
        graph_emb: torch.Tensor,  # (B, N, D)
        entity_emb: torch.Tensor | None = None,  # (B, N, D)
    ) -> torch.Tensor:
        B, N = title_emb.shape[:2]
        if entity_emb is not None:
            stacked = torch.stack(
                [title_emb, graph_emb, entity_emb],
                dim=-2,
            )  # (B, N, 3, D)
            num_views = 3
        else:
            stacked = torch.stack([title_emb, graph_emb], dim=-2)  # (B, N, 2, D)
            num_views = 2
        stacked = stacked.view(B * N, num_views, self.news_dim)
        fused = self.attn_pool(stacked)  # (B*N, D)
        return fused.view(B, N, self.news_dim)


class GLORYUserEncoder(nn.Module):
    """Pool a sequence of clicked-news embeddings into a user vector."""

    def __init__(self, config: GLORYConfig):
        super().__init__()
        self.news_dim = config.head_num * config.head_dim
        self.msa = MultiHeadAttention(
            self.news_dim,
            self.news_dim,
            self.news_dim,
            config.head_num,
            config.head_dim,
        )
        self.attn_pool = AttentionPooling(self.news_dim, config.attention_hidden_dim)

    def forward(
        self,
        clicked_news: torch.Tensor,  # (B, H, D)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.msa(clicked_news, clicked_news, clicked_news, mask)
        return self.attn_pool(h, mask)


class GLORYCandidateEncoder(nn.Module):
    """Candidate encoder.

    Without entities: Linear + LeakyReLU on title embeddings.
    With entities: stack ``[title, origin_entity, neighbor_entity]`` →
    AttentionPooling → Linear + LeakyReLU.
    """

    def __init__(self, config: GLORYConfig):
        super().__init__()
        self.news_dim = config.head_num * config.head_dim
        self.use_entity = config.use_entity
        if self.use_entity:
            self.attn_pool = AttentionPooling(
                self.news_dim, config.attention_hidden_dim
            )
        self.linear = nn.Linear(self.news_dim, self.news_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        cand_emb: torch.Tensor,
        origin_entity_emb: torch.Tensor | None = None,
        neighbor_entity_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.use_entity
            and origin_entity_emb is not None
            and neighbor_entity_emb is not None
        ):
            B, C = cand_emb.shape[:2]
            stacked = torch.stack(
                [cand_emb, origin_entity_emb, neighbor_entity_emb],
                dim=-2,
            )  # (B, C, 3, D)
            stacked = stacked.view(B * C, 3, self.news_dim)
            pooled = self.attn_pool(stacked)  # (B*C, D)
            cand_emb = pooled.view(B, C, self.news_dim)
        return self.act(self.linear(cand_emb))


# ======================================================================
# Full GLORY model
# ======================================================================


class GLORY(BaseModel):
    """GLORY: local news encoder + global GNN + user fusion + dot scoring."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: GLORYConfig | None = None,
        **kwargs,
    ):
        super().__init__()
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

        self.word_embedding = nn.Embedding(
            vocab_size, config.word_emb_dim, padding_idx=0
        )
        self.word_embedding.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        self.local_news_encoder = GLORYNewsEncoder(config, self.word_embedding)
        # ``use_torchgeo=True`` swaps in PyG's CUDA-optimized GatedGraphConv
        # (matches the reference GLORY repo). Default keeps the
        # framework-agnostic pure-PyTorch implementation.
        if getattr(config, "use_torchgeo", False):
            from torch_geometric.nn import GatedGraphConv as _PyGGatedGraphConv
            self.global_news_encoder = _PyGGatedGraphConv(
                out_channels=self.news_dim,
                num_layers=config.gnn_num_layers,
                aggr="add",
            )
        else:
            self.global_news_encoder = GatedGraphConv(
                self.news_dim,
                num_layers=config.gnn_num_layers,
                aggr="add",
            )
        self.click_encoder = GLORYClickEncoder(config)
        self.user_encoder = GLORYUserEncoder(config)
        self.candidate_encoder = GLORYCandidateEncoder(config)
        self.click_predictor = DotProduct()

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
                    (entity_vocab, self.entity_emb_dim),
                    dtype=np.float32,
                )
            self.entity_embedding = nn.Embedding(entity_vocab, self.entity_emb_dim)
            self.entity_embedding.weight = nn.Parameter(
                torch.tensor(entity_emb, dtype=torch.float32)
            )

            self.local_entity_encoder = EntityEncoder(
                entity_dim=self.entity_emb_dim,
                news_dim=self.news_dim,
                head_dim=config.head_dim,
                attention_hidden_dim=config.attention_hidden_dim,
                dropout_rate=config.dropout_rate,
            )
            self.global_entity_encoder = GlobalEntityEncoder(
                entity_dim=self.entity_emb_dim,
                head_num=config.head_num,
                head_dim=config.head_dim,
                attention_hidden_dim=config.attention_hidden_dim,
                dropout_rate=config.dropout_rate,
            )

        # news_encoder / user_encoder names satisfy the BaseModel contract
        # used by the shared evaluator for standard models.  GLORY's eval
        # path is custom (pre-computes global embeddings once per epoch),
        # so these attributes exist only for introspection.
        self.news_encoder = self.local_news_encoder
        self.user_encoder_public = self.user_encoder

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Training forward.

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
        mapping = mapping_idx.masked_fill(~valid, 0)  # safe index

        # Encode every subgraph node once with the local encoder.
        flat = subgraph_x.unsqueeze(0)  # (1, N_total, feat)
        x_encoded = self.local_news_encoder(flat).squeeze(0)  # (N_total, D)

        # GNN over the full (batched) subgraph.
        graph_emb = self.global_news_encoder(x_encoded, edge_index)  # (N_total, D)

        # Gather history embeddings from both views.
        valid_mask = valid.unsqueeze(-1)  # (B, H, 1)
        clicked_title = x_encoded[mapping].masked_fill(~valid_mask, 0)  # (B, H, D)
        clicked_graph = graph_emb[mapping].masked_fill(~valid_mask, 0)  # (B, H, D)

        # Clicked entity encoding (optional).
        clicked_entity_emb: torch.Tensor | None = None
        if self.use_entity:
            # Extract entity IDs for clicked news from subgraph features.
            clicked_entity_ids = subgraph_x[
                mapping,
                self.title_size : self.title_size + self.entity_size,
            ].long()  # (B, H, entity_size)
            clicked_entity_ids = clicked_entity_ids.masked_fill(~valid_mask, 0)
            entity_embedded = self.entity_embedding(
                clicked_entity_ids
            )  # (B, H, E, dim)
            clicked_entity_emb = self.local_entity_encoder(
                entity_embedded,
            )  # (B, H, D)

        # Fuse → pool into user vector.
        fused = self.click_encoder(
            clicked_title,
            clicked_graph,
            clicked_entity_emb,
        )  # (B, H, D)
        user_emb = self.user_encoder(fused, valid.float())  # (B, D)

        # Candidates: encode locally then project.
        cand_local = self.local_news_encoder(cand_tokens)  # (B, C, D)

        # Candidate entity encoding (optional).
        cand_origin_emb: torch.Tensor | None = None
        cand_neighbor_emb: torch.Tensor | None = None
        if self.use_entity and "candidate_entity" in inputs:
            cand_entity = inputs["candidate_entity"].long()
            entity_mask = inputs.get("entity_mask")

            E = self.entity_size
            origin_ids = cand_entity[..., :E]  # (B, C, E)
            neighbor_ids = cand_entity[..., E:]  # (B, C, E*neighbors)

            origin_embedded = self.entity_embedding(origin_ids)  # (B, C, E, dim)
            cand_origin_emb = self.local_entity_encoder(
                origin_embedded,
            )  # (B, C, D)

            B, C = neighbor_ids.shape[:2]
            n_neighbors = neighbor_ids.shape[-1]
            neighbor_embedded = self.entity_embedding(
                neighbor_ids,
            )  # (B, C, E*neighbors, dim)

            ent_mask = None
            if entity_mask is not None:
                ent_mask = entity_mask.float().reshape(B * C, n_neighbors)

            cand_neighbor_emb = self.global_entity_encoder(
                neighbor_embedded,
                ent_mask,
            )  # (B, C, D)

        cand_final = self.candidate_encoder(
            cand_local,
            cand_origin_emb,
            cand_neighbor_emb,
        )  # (B, C, D)

        # Dot-product scoring.
        return self.click_predictor(cand_final, user_emb)  # (B, C)
