"""PyTorch frozen-PLM news encoder.

Drop-in replacement for the GloVe-based news encoder. Takes per-news
**parsed integer ids** (the suffix of ``"N123"``) and returns
``(..., news_dim)`` vectors by looking up the cached PLM embedding and
applying a small trainable projection.

This is the v1 frozen variant: the PLM has already been run offline and
its outputs are cached as a dense ``(max_parsed_id + 1, plm_dim)`` table
via :func:`src.core.data.encoders.plm.attach_plm_embeddings`. Training-
time forward is just a lookup + Linear — fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PLMNewsEncoder(nn.Module):
    """Frozen-PLM news encoder.

    Args:
        plm_embeddings_by_id: ``(max_parsed_id + 1, plm_dim)`` float32
            lookup table indexed by parsed-integer news id. Row 0 is
            the padding zero vector. Produced by
            :func:`src.core.data.encoders.plm.attach_plm_embeddings`.
        news_dim: Output dimension after the trainable projection.
            Typically matches the downstream model's ``embedding_size``.
    """

    def __init__(self, plm_embeddings_by_id: np.ndarray, news_dim: int):
        super().__init__()
        plm_dim = int(plm_embeddings_by_id.shape[1])
        self.plm_dim = plm_dim
        self.news_dim = news_dim

        # Frozen embedding lookup. ``padding_idx=0`` zeros the row 0
        # gradient (redundant since we freeze, but documents intent).
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(plm_embeddings_by_id).float(),
            freeze=True,
            padding_idx=0,
        )
        # Only trainable params in this encoder.
        self.projection = nn.Linear(plm_dim, news_dim)

    @staticmethod
    def valid_mask(features: torch.Tensor) -> torch.Tensor:
        """A PLM news slot is valid when its parsed id is non-zero
        (``0`` is reserved for padding by :func:`attach_plm_embeddings`).
        """
        return features != 0

    def forward(
        self,
        features: torch.Tensor,
        training: bool = True,  # accepted for API parity with other encoders
    ) -> torch.Tensor:
        """Look up frozen PLM vectors and project.

        Args:
            features: int tensor of any shape ``(..., )``. Values must
                be valid parsed-int news ids; ``0`` returns the zero
                padding row.

        Returns:
            ``(..., news_dim)`` float tensor.
        """
        return self.projection(self.embedding(features.long()))
