"""PyTorch token-level frozen-PLM news encoder.

Companion to :class:`PLMNewsEncoder` (sentence-level v1). The
token-level encoder keeps the full ``(T, plm_dim)`` sequence per news
and applies a configurable :class:`Pooler` at training time —
matching IP2's structure but with the PLM weights still frozen.

Use cases:
    - ``pooler=mean``  → equivalent to sentence-level (option 2a)
    - ``pooler=attention`` → IP2-style learnable attention pool (option 2b)
    - ``pooler=cls`` / ``pooler=gate`` → alternative frozen / learnable poolers
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .plm_pooler import Pooler


class PLMTokenNewsEncoder(nn.Module):
    """Frozen-PLM token-level news encoder + pluggable pooler + projection.

    Honors the shared encoder contract:
        - ``__call__(features, training=...) -> (*leading, news_dim)``
        - ``valid_mask(features) -> bool of shape (*leading,)``

    Args:
        plm_token_embeddings_by_id: ``(max_parsed_id + 1, T, plm_dim)``
            float32 — token-level frozen PLM outputs, indexed by parsed
            news id. Row 0 is the padding zero block.
        plm_attention_mask_by_id: ``(max_parsed_id + 1, T)`` int8 (0/1)
            attention mask, indexed by parsed news id.
        news_dim: Output dimension after the trainable projection.
        pooler_type: One of ``"mean"``, ``"cls"``, ``"attention"``,
            ``"gate"``, ``"avg_first_last"``. See :class:`Pooler`.
        attention_query_dim, num_heads, head_dim, dropout_rate:
            Pooler-specific hyperparameters.
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
        news_dim: int,
        *,
        pooler_type: str = "attention",
        attention_query_dim: int = 200,
        num_heads: int = 6,
        head_dim: int = 128,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        max_id, T, plm_dim = plm_token_embeddings_by_id.shape
        self.plm_dim = int(plm_dim)
        self.news_dim = int(news_dim)
        self.max_length = int(T)

        # Frozen 3D lookup table: (max_id+1, T, plm_dim). Stored as a
        # buffer (not a Parameter) so the optimizer never tries to update
        # it. Lives on whichever device the rest of the model is moved to.
        self.register_buffer(
            "token_embeddings",
            torch.from_numpy(plm_token_embeddings_by_id).float(),
            persistent=False,
        )
        self.register_buffer(
            "attention_mask",
            torch.from_numpy(plm_attention_mask_by_id).to(torch.int8),
            persistent=False,
        )

        self.pooler = Pooler(
            plm_dim=plm_dim,
            pooler_type=pooler_type,
            attention_query_dim=attention_query_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
        )
        self.projection = nn.Linear(plm_dim, news_dim)

    @staticmethod
    def valid_mask(features: torch.Tensor) -> torch.Tensor:
        """A PLM news slot is valid when its parsed id is non-zero."""
        return features != 0

    def forward(
        self,
        features: torch.Tensor,
        training: bool = True,  # accepted for API parity
    ) -> torch.Tensor:
        """Look up token-level frozen vectors, pool, project.

        Args:
            features: int tensor of any shape ``(...,)`` of parsed news ids.

        Returns:
            ``(..., news_dim)`` float tensor.
        """
        leading_shape = features.shape
        ids = features.long().reshape(-1)  # (N*,)

        # Lookup: (N*, T, plm_dim) and (N*, T)
        tokens = self.token_embeddings.index_select(0, ids)
        mask = self.attention_mask.index_select(0, ids)

        # Pool over tokens → (N*, plm_dim)
        pooled = self.pooler(tokens, mask)

        # Project to news_dim → (N*, news_dim) → reshape to (..., news_dim)
        out = self.projection(pooled)
        return out.reshape(*leading_shape, self.news_dim)
