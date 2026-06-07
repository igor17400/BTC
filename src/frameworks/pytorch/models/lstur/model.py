"""LSTUR top-level model (PyTorch).

Reference: An et al., "Neural News Recommendation with Long- and
Short-term User Representations", ACL 2019.

Composes :class:`NewsEncoder` and :class:`UserEncoder` with an injected
:class:`TextEncoder` (GloVe or PLM, picked by the runner-provided
``encoder`` config). Single code path regardless of encoder type.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.models.configs import LSTURConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class LSTUR(BaseModel):
    """LSTUR — CNN news encoder + GRU user encoder + long-term user embedding."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        num_users: int,
        config: LSTURConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = LSTURConfig(**config_overrides)
        self.config = config
        self.num_users = num_users
        self.process_user_id = config.process_user_id

        text_encoder = build_text_encoder(
            framework="pytorch",
            encoder_cfg=getattr(config, "encoder", None),
            processed_news=processed_news,
            kind="title",
        )

        self.news_encoder = NewsEncoder(
            config,
            text_encoder,
            num_categories=int(processed_news.get("num_categories", 0)),
            num_subcategories=int(processed_news.get("num_subcategories", 0)),
        )
        self.user_encoder = UserEncoder(config, self.news_encoder, num_users)

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
            inputs: dict containing at least::

                hist_features   (B, H)        news_idx
                cand_features   (B, C)        news_idx
                user_ids        (B,)          int

            plus optional structured side-channels packed by the runner::

                hist_category, hist_subcategory   (B, H)
                cand_category, cand_subcategory   (B, C)

        Returns:
            ``(B, C)`` raw logit scores.
        """
        history_packed = _pack(inputs, "hist", self.config)
        candidate_packed = _pack(inputs, "cand", self.config)
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))

        user_repr = self.user_encoder(
            [history_packed, user_ids], training=True
        )  # (B, D)
        cand_repr = self.news_encoder(candidate_packed, training=True)  # (B, C, D)
        return torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)


def _pack(
    inputs: dict[str, torch.Tensor], prefix: str, config: LSTURConfig
) -> torch.Tensor:
    """Pack ``[news_idx | category | subcategory]`` columns from the inputs dict.

    Returns plain ``news_idx`` when both category flags are disabled —
    matching the eval dataloader's single-view shape so the news encoder
    receives the same layout at train and eval time.
    """
    news_idx = inputs[f"{prefix}_features"]
    if not config.use_category and not config.use_subcategory:
        return news_idx

    parts: list[torch.Tensor] = [news_idx.unsqueeze(-1)]
    if config.use_category:
        parts.append(inputs[f"{prefix}_category"].unsqueeze(-1))
    if config.use_subcategory:
        parts.append(inputs[f"{prefix}_subcategory"].unsqueeze(-1))
    return torch.cat(parts, dim=-1)
