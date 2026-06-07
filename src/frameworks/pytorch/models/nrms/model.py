"""NRMS top-level model (PyTorch).

Reference: Wu et al., "Neural News Recommendation with Multi-Head
Self-Attention", EMNLP 2019.

Composes :class:`NewsEncoder` and :class:`UserEncoder` with an injected
:class:`TextEncoder` (GloVe or PLM, picked by the runner-provided
``encoder`` config). Single code path regardless of encoder type.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.models.configs import NRMSConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class NRMS(BaseModel):
    """NRMS: news-encoder + user-encoder + dot-product scoring.

    The text-pipeline choice (GloVe or PLM) is fully abstracted by the
    injected :class:`TextEncoder` — the model body is a single forward.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NRMSConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = NRMSConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=getattr(config, "encoder", None),
            processed_news=processed_news,
            kind="title",
        )

        self.news_encoder = NewsEncoder(config, text_encoder)
        self.user_encoder = UserEncoder(config, self.news_encoder)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Score a training batch. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluations.default`).
        """
        return self.score_training_batch(inputs)

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score with softmax-over-candidates loss.

        Args:
            inputs: ``{"hist_features": (B, H), "cand_features": (B, C)}``
                — both are parsed news-id int tensors.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        history = inputs["hist_features"]
        candidates = inputs["cand_features"]

        user_repr = self.user_encoder(history, training=True)  # (B, E)
        cand_repr = self.news_encoder(candidates, training=True)  # (B, C, E)
        return torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
