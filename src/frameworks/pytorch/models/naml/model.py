"""NAML top-level model (PyTorch).

Reference: Wu et al., "Neural News Recommendation with Attentive Multi-View
Learning", IJCAI 2019.

Composes :class:`NewsEncoder` and :class:`UserEncoder` with injected
:class:`TextEncoder` instances (one per text view). Single code path
regardless of encoder type (GloVe or frozen-cached PLM).
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.models.configs import NAMLConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class NAML(BaseModel):
    """NAML — multi-view news + additive-attention user encoder."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NAMLConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = NAMLConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        encoder_cfg = getattr(config, "encoder", None)
        title_text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
        )
        # Abstract view is optional: when process_abstract is False the data
        # pipeline produces no abstract cache, so skip the encoder entirely
        # (the news encoder then runs on title + category + subcategory).
        abstract_text_encoder = (
            build_text_encoder(
                framework="pytorch",
                encoder_cfg=encoder_cfg,
                processed_news=processed_news,
                kind="abstract",
            )
            if config.process_abstract
            else None
        )

        self.news_encoder = NewsEncoder(
            config,
            title_text_encoder,
            abstract_text_encoder,
            num_categories=int(processed_news["num_categories"]),
            num_subcategories=int(processed_news["num_subcategories"]),
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Score a training batch. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator.
        """
        return self.score_training_batch(inputs)

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score history × candidates.

        Args:
            inputs: dict with at least::

                hist_features  (B, H)         news_idx
                hist_category  (B, H)         category_id
                hist_subcategory (B, H)       subcategory_id
                cand_features  (B, C)         news_idx
                cand_category  (B, C)         category_id
                cand_subcategory (B, C)       subcategory_id

        Returns:
            ``(B, C)`` raw logit scores.
        """
        hist_packed = _pack(inputs, "hist")
        cand_packed = _pack(inputs, "cand")

        user_repr = self.user_encoder(hist_packed, training=True)  # (B, D)
        cand_repr = self.news_encoder(cand_packed, training=True)  # (B, C, D)
        return torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)


def _pack(inputs: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """Pack [news_idx | category | subcategory] columns from the inputs dict."""
    news_idx = inputs[f"{prefix}_features"]
    category = inputs[f"{prefix}_category"]
    subcategory = inputs[f"{prefix}_subcategory"]
    return torch.stack([news_idx, category, subcategory], dim=-1)
