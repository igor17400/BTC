"""CAUM top-level model (PyTorch).

Reference: Qi et al., "News Recommendation with Candidate-aware User
Modeling", SIGIR 2022.

Composes :class:`NewsEncoder`, :class:`UserEncoder` (fallback mean-pool)
and :class:`InterModel` (candidate-aware scoring) with an injected
:class:`TextEncoder` (GloVe or PLM). Single code path regardless of
encoder type.

During training, ``score_training_batch`` encodes clicked + candidate
news once, then loops over each impression slot to score it via the
candidate-aware ``inter_model``. During evaluation, a custom evaluator
(see :mod:`src.core.models.evaluations.custom.caum`) precomputes news
vectors and runs ``inter_model`` per candidate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.core.models.configs import CAUMConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .inter_model import InterModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class CAUM(BaseModel):
    """CAUM — candidate-aware news recommendation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CAUMConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = CAUMConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=getattr(config, "encoder", None),
            processed_news=processed_news,
            kind="title",
        )

        num_entities = _resolve_num_entities(processed_news)
        num_categories = int(processed_news.get("num_categories", 0))

        self.news_encoder = NewsEncoder(
            config,
            text_encoder,
            num_entities=num_entities,
            num_categories=num_categories,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)
        self.inter_model = InterModel(config)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Score a training batch. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.inter_model``
        directly via the CAUM custom evaluator.
        """
        return self.score_training_batch(inputs, training)

    def score_training_batch(
        self, inputs: dict[str, torch.Tensor], training: bool = True
    ) -> torch.Tensor:
        """Score history × candidates with the candidate-aware inter_model.

        Args:
            inputs: dict with at least::

                hist_tokens  (B, H, k)   packed [news_idx | entities | category]
                cand_tokens  (B, C, k)   same column layout

        Returns:
            ``(B, C)`` raw logit scores.
        """
        hist_packed = inputs["hist_tokens"]
        cand_packed = inputs["cand_tokens"]

        clicked_vecs = self.news_encoder(
            hist_packed, training=training
        )  # (B, H, news_dim)
        cand_vecs = self.news_encoder(
            cand_packed, training=training
        )  # (B, C, news_dim)

        scores = []
        for i in range(self.config.max_impressions_length):
            cand_i = cand_vecs[:, i, :]
            score_i = self.inter_model(cand_i, clicked_vecs, training=training)
            scores.append(score_i)

        return torch.stack(scores, dim=1)  # (B, C)


def _resolve_num_entities(processed_news: dict[str, Any]) -> int:
    """Recover the entity vocab size from processed_news.

    Tries ``num_entities`` first; falls back to ``max(entity_indices) + 1``;
    defaults to 1 (single padding row) when neither is available — which
    keeps construction safe even on datasets without entities.
    """
    n = processed_news.get("num_entities")
    if n is not None:
        return int(n)
    entity_indices = processed_news.get("entity_indices")
    if entity_indices is not None:
        return int(np.max(entity_indices)) + 1
    return 1
