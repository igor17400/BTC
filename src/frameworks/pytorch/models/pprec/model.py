"""PP-Rec top-level model (PyTorch).

Reference: Qi et al., "PP-Rec: News Recommendation with Personalized
User Interest and Time-aware News Popularity", ACL 2021.

Composes the relevance branch (:class:`NewsEncoder` + :class:`UserEncoder`)
with the popularity branch (:class:`PopularityPredictor` driven by a
separate *bias* news encoder) and an :class:`ActivityGater` that balances
the two. Each news encoder receives its own :class:`TextEncoder` instance
so the relevance and bias paths have independent text weights.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.models.configs import PPRecConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .popularity import ActivityGater, PopularityPredictor
from .user_encoder import UserEncoder


class PPRec(BaseModel):
    """PP-Rec PyTorch implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: PPRecConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = PPRecConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        encoder_cfg = getattr(config, "encoder", None)

        # Two TextEncoder instances — relevance and bias news encoders
        # need independent text weights.
        rel_text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
        )
        bias_text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
        )

        entity_emb, category_emb = _build_aux_embeddings(processed_news, config)

        self.news_encoder = NewsEncoder(
            config, rel_text_encoder, entity_emb, category_emb
        )
        self.bias_news_encoder = NewsEncoder(
            config, bias_text_encoder, entity_emb, category_emb
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)
        self.popularity_predictor = PopularityPredictor(config)
        self.activity_gater = (
            ActivityGater(config) if config.use_activity_gate else None
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass for training. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluations.pp_rec`).
        """
        return self.score_training_batch(inputs)

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_ctr = inputs.get("cand_ctr")
        cand_recency = inputs.get("cand_recency")

        user_vec = self.user_encoder(hist_features, hist_ctr, training=True)

        rel_cand_vecs = self.news_encoder(cand_features, training=True)  # (B, C, D)
        bias_cand_vecs = self.bias_news_encoder(cand_features, training=True)
        rel_scores = torch.sum(rel_cand_vecs * user_vec.unsqueeze(1), dim=-1)

        B, C = cand_features.shape[:2]
        bias_flat = bias_cand_vecs.reshape(B * C, self.config.news_dim)
        recency_flat = cand_recency.reshape(B * C) if cand_recency is not None else None
        ctr_flat = cand_ctr.reshape(B * C) if cand_ctr is not None else None
        pop_scores = self.popularity_predictor(
            bias_flat, recency_indices=recency_flat, ctr_values=ctr_flat
        ).reshape(B, C)

        if self.activity_gater is not None:
            eta = self.activity_gater(user_vec).unsqueeze(-1)
            scores = eta * rel_scores + (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores
        return scores


def _build_aux_embeddings(
    processed_news: dict[str, Any], config: PPRecConfig
) -> tuple[torch.nn.Embedding | None, torch.nn.Embedding | None]:
    """Build entity + category embedding layers (or ``None`` when disabled)."""
    import torch.nn as nn

    entity_emb: nn.Embedding | None = None
    if config.use_entity and "entity_embeddings" in processed_news:
        entity_emb = nn.Embedding(
            processed_news["entity_vocab_size"], config.entity_embedding_dim
        )
        entity_emb.weight = nn.Parameter(
            torch.tensor(processed_news["entity_embeddings"], dtype=torch.float32)
        )

    category_emb: nn.Embedding | None = None
    if "num_categories" in processed_news:
        category_emb = nn.Embedding(
            int(processed_news["num_categories"]) + 1, config.category_embedding_dim
        )

    return entity_emb, category_emb
