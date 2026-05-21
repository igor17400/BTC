"""DIGAT (EMNLP 2022 Findings) — PyTorch implementation.

Dual Interactive Graph Attention Networks for news recommendation.
Two interacting graph channels (news-SAG + user-topic) co-evolve across
``graph_depth`` layers.  The SAG (Semantic Augmented Graph) is precomputed
offline; the user-topic graph is built per behavior from category
assignments.

No ``torch_scatter`` dependency — scatter operations are implemented
with standard PyTorch ops (see :mod:`.graph_encoder`).

Reference: Mao et al., "DIGAT: Modeling News Recommendation with
Dual-Graph Interaction", EMNLP 2022 Findings.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import DIGATConfig

from ...layers import PLMTokenLookup
from ..base import BaseModel
from .graph_encoder import GraphEncoder
from .news_encoder import NewsEncoder


class DIGAT(BaseModel):
    """DIGAT: Dual Interactive Graph Attention Networks.

    Training: encode candidates (with SAG) + history → dual graph
    interaction → dot-product scoring.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: DIGATConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = DIGATConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        self.num_categories = num_categories + 1  # +1 for padding category
        encoder_type = getattr(getattr(config, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        if encoder_type == "glove":
            vocab_size = int(processed_news["vocab_size"])
            embeddings_matrix = processed_news["embeddings"]
            self.word_embedding: nn.Module = nn.Embedding(
                vocab_size, config.embedding_size
            )
            self.word_embedding.weight = nn.Parameter(
                torch.tensor(embeddings_matrix, dtype=torch.float32)
            )
        else:
            if "plm_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"DIGAT with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id']."
                )
            plm_dim = int(processed_news["plm_dim"])
            self.word_embedding = PLMTokenLookup(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                plm_dim=plm_dim,
                output_dim=config.embedding_size,
            )

        self.news_encoder = NewsEncoder(
            config, self.word_embedding, encoder_type=encoder_type
        )

        self.graph_encoder = GraphEncoder(config)
        self.graph_encoder.topic_node_emb = nn.Parameter(
            torch.zeros(self.num_categories, config.news_embedding_dim)
        )
        nn.init.uniform_(self.graph_encoder.topic_node_emb, -0.1, 0.1)

        self.news_graph_size = config.news_graph_size
        self.max_history = config.max_history_length
        self.D = config.news_embedding_dim

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Training forward pass.

        Expected input keys:
            hist_tokens: ``(batch_size, hist_len, title_len)``
            hist_mask: ``(batch_size, hist_len, title_len)``
            user_graph: ``(batch_size, user_graph_size, user_graph_size)``
            user_category_mask: ``(batch_size, num_categories)``
            user_category_indices: ``(batch_size, hist_len)``
            cand_tokens: ``(batch_size, num_cands, sag_size, title_len)``
            cand_mask: ``(batch_size, num_cands, sag_size, title_len)``
            cand_graph: ``(batch_size, num_cands, sag_size, sag_size)``
            cand_graph_mask: ``(batch_size, num_cands, sag_size)``

        Returns:
            ``(batch_size, num_cands)`` raw logits.
        """
        batch_size, num_cands = inputs["cand_graph"].shape[:2]
        sag_size = self.news_graph_size
        user_graph_size = self.max_history + self.num_categories
        batch_cands = batch_size * num_cands

        # Flatten candidates: shape depends on encoder.
        # GloVe: ``(B, C, sag_size, T)`` token ids → ``(B*C, sag_size, T)``.
        # PLM:   ``(B, C, sag_size)`` parsed news_idx → ``(B*C, sag_size)``.
        if self.encoder_type == "glove":
            cand_tokens = inputs["cand_tokens"].view(batch_cands, sag_size, -1)
            cand_mask = (
                inputs["cand_mask"].view(batch_cands, sag_size, -1)
                if "cand_mask" in inputs
                else None
            )
        else:
            cand_tokens = inputs["cand_tokens"].view(batch_cands, sag_size)
            cand_mask = None
        cand_graph = inputs["cand_graph"].view(batch_cands, sag_size, sag_size)
        cand_graph_mask = inputs["cand_graph_mask"].view(batch_cands, sag_size)

        # Expand user data: (batch_size, ...) → (batch_cands, ...)
        user_graph = (
            inputs["user_graph"]
            .unsqueeze(1)
            .expand(-1, num_cands, -1, -1)
            .reshape(batch_cands, user_graph_size, user_graph_size)
        )
        user_cat_mask = (
            inputs["user_category_mask"]
            .unsqueeze(1)
            .expand(-1, num_cands, -1)
            .reshape(batch_cands, self.num_categories)
        )
        user_cat_indices = (
            inputs["user_category_indices"]
            .unsqueeze(1)
            .expand(-1, num_cands, -1)
            .reshape(batch_cands, self.max_history)
        )

        # Encode candidate news (each with SAG neighbors)
        cand_emb = self.news_encoder(
            cand_tokens, cand_mask
        )  # (batch_cands, sag_size, feat_dim)

        # Encode user history
        user_news_emb = self.news_encoder(
            inputs["hist_tokens"],
            inputs.get("hist_mask"),
        )  # (batch_size, hist_len, feat_dim)
        user_news_emb = (
            user_news_emb.unsqueeze(1)
            .expand(-1, num_cands, -1, -1)
            .reshape(batch_cands, self.max_history, self.D)
        )

        # Dual graph interaction
        news_ctx, user_ctx = self.graph_encoder(
            cand_emb,
            cand_graph,
            cand_graph_mask,
            user_news_emb,
            user_graph,
            user_cat_mask,
            user_cat_indices,
            self.num_categories,
        )

        # Dot-product scoring
        news_ctx = news_ctx.view(batch_size, num_cands, self.D)
        user_ctx = user_ctx.view(batch_size, num_cands, self.D)
        return (news_ctx * user_ctx).sum(dim=-1)
