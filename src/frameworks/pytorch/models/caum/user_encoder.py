"""CAUM user encoder (PyTorch) — fallback mean-pool over history.

CAUM's *real* user modelling is candidate-aware and lives in
:mod:`.inter_model`. This module exists so the model satisfies the
:class:`BaseModel` contract (``self.user_encoder`` must be an
``nn.Module``), and is used by the per-impression eval pipeline when a
plain user vector is needed.

It mean-pools the per-history-slot news vectors with a validity mask
derived from the packed news_idx column. The shared evaluator does not
actually call this for CAUM (CAUM has a custom evaluator at
:mod:`src.core.models.evaluations.custom.caum`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import CAUMConfig

from .news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    """Mean-pool user encoder for the BaseModel contract."""

    def __init__(self, config: CAUMConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """``inputs``: ``(B, H, k)`` packed → ``(B, news_dim)``."""
        news_embeds = self.news_encoder(inputs, training=training)  # (B, H, D)
        mask = self.news_encoder.valid_mask(inputs).float()  # (B, H)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (news_embeds * mask.unsqueeze(-1)).sum(dim=1) / count
