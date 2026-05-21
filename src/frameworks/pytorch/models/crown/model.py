"""CROWN top-level model (PyTorch).

Reference: Ko et al., "CROWN: A Novel Approach to Comprehending Users'
Preferences for Accurate Personalized News Recommendation", WWW 2025.

Composes :class:`NewsEncoder` (Transformer + k-intent disentanglement)
with :class:`UserEncoder` (bipartite user<->news GNN + paper eq. 9
attention). Each text view (title, abstract) gets its own
:class:`TextEncoder` instance via :func:`build_text_encoder`; no
encoder-type branches inside the model body.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import CROWNConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class CROWN(BaseModel):
    """CROWN: Category-aware intent disentanglement + GNN user encoder."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CROWNConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = CROWNConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        num_subcategories = int(processed_news.get("num_subcategories", 100))
        encoder_cfg = getattr(config, "encoder", None)

        # Two TextEncoders (title + abstract).
        title_text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
        )
        abstract_text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="abstract",
        )

        # Category / subcategory embeddings (paper: uniform init).
        self.category_embedding = nn.Embedding(
            num_categories + 1, config.category_embedding_dim
        )
        self.subcategory_embedding = nn.Embedding(
            num_subcategories + 1, config.subcategory_embedding_dim
        )
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subcategory_embedding.weight, -0.1, 0.1)

        self.news_encoder = NewsEncoder(
            config,
            title_text_encoder,
            abstract_text_encoder,
            self.category_embedding,
            self.subcategory_embedding,
        )
        # Resize the category predictor now that we know num_categories.
        self.news_encoder.category_predictor = nn.Linear(
            config.intent_embedding_dim, num_categories + 1
        )

        self.user_encoder = UserEncoder(config, self.news_encoder)
        self.alpha = config.alpha

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Training forward pass. Returns raw logits ``(B, C)``.

        Inputs (training-time keys from the runner):
            hist_features   (B, H)   news_idx
            hist_category   (B, H)   category_id
            hist_subcategory (B, H)  subcategory_id
            cand_features   (B, C)   news_idx
            cand_category   (B, C)
            cand_subcategory (B, C)
        """
        hist_packed = _pack(inputs, "hist")
        cand_packed = _pack(inputs, "cand")

        # Encode candidates first so the aux loss is computed against them.
        cand_full = self.news_encoder(cand_packed, compute_aux_loss=training)
        # Snapshot — history encoding below resets self.auxiliary_loss.
        self._aux_loss_snapshot = self.news_encoder.auxiliary_loss

        history_mask = self.news_encoder.valid_mask(hist_packed)

        user_repr = self.user_encoder.forward_with_candidates(
            history_packed=hist_packed,
            history_mask=history_mask,
            candidate_repr=cand_full,
        )

        return (user_repr.unsqueeze(1) * cand_full).sum(dim=-1)

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Return weighted auxiliary loss from the news encoder."""
        return self.alpha * self._aux_loss_snapshot


def _pack(inputs: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """Pack [news_idx | category | subcategory] columns into ``(*, 3)``."""
    news_idx = inputs[f"{prefix}_features"]
    category = inputs[f"{prefix}_category"]
    subcategory = inputs[f"{prefix}_subcategory"]
    return torch.stack([news_idx, category, subcategory], dim=-1)
