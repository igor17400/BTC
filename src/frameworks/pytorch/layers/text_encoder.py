"""Unified text-encoder interface for news models (PyTorch).

Every news model receives a :class:`TextEncoder` already configured for
either GloVe word embeddings or a frozen-cached PLM, and calls it the
same way::

    tokens, mask = text_encoder(news_idx)   # (..., T, D), (..., T)
    D = text_encoder.output_dim             # = embedding_size

This hides the GloVe/PLM choice from the model. The model body becomes
a single encoder-agnostic code path; swapping the encoder is a yaml
change, not a code change.

The leaf abstraction here is matched in
:mod:`src.frameworks.jax.layers.text_encoder`. The factory that picks
the right implementation from a Hydra ``encoder`` block lives in
:mod:`src.core.models.text_encoder`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Protocol class for the model-facing text encoder contract.

    Subclasses MUST:
      - Set ``self.output_dim: int`` and ``self.max_length: int``.
      - Implement ``forward(news_idx) -> (tokens, mask)`` where
        ``tokens`` is ``(..., max_length, output_dim)`` float and
        ``mask`` is ``(..., max_length)`` int8 (1 = valid token).
    """

    output_dim: int
    max_length: int

    def forward(  # pragma: no cover - protocol
        self, news_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GloveTextEncoder(TextEncoder):
    """GloVe word lookup keyed by news_idx.

    Holds the precomputed per-news word-id matrix
    (``processed_news['tokens']`` for titles or
    ``processed_news['abstract_tokens']`` for abstracts) as a frozen
    buffer, plus a trainable ``nn.Embedding(vocab_size, embedding_dim)``
    initialised from pretrained GloVe vectors.

    The padding row of the word-id matrix is expected to be all zeros
    so that ``mask`` correctly marks padded positions as 0.

    Args:
        word_id_matrix: ``(num_news, T)`` int — per-news word IDs.
        vocab_size: Word vocabulary size.
        embedding_dim: Output channel dim (= ``output_dim``).
        pretrained_embeddings: Optional ``(vocab_size, embedding_dim)``
            float matrix to initialise the embedding weights.
    """

    def __init__(
        self,
        word_id_matrix: np.ndarray,
        *,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings: np.ndarray | None = None,
    ):
        super().__init__()
        word_id_matrix = np.asarray(word_id_matrix)
        if word_id_matrix.ndim != 2:
            raise ValueError(
                f"word_id_matrix must be 2D (num_news, T); got shape "
                f"{word_id_matrix.shape}"
            )
        _, T = word_id_matrix.shape
        self.output_dim = int(embedding_dim)
        self.max_length = int(T)

        # Frozen per-news word-id table — moved with the model but
        # ignored by the optimizer (buffer, not Parameter).
        self.register_buffer(
            "word_ids",
            torch.from_numpy(word_id_matrix).long(),
            persistent=False,
        )

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            pretrained = np.asarray(pretrained_embeddings)
            if pretrained.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    f"pretrained_embeddings shape {pretrained.shape} != "
                    f"(vocab_size={vocab_size}, embedding_dim={embedding_dim})"
                )
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained, dtype=torch.float32)
            )

    @staticmethod
    def valid_mask(news_idx: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed id is non-zero."""
        return news_idx != 0

    def forward(self, news_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = news_idx.shape
        flat_idx = news_idx.long().reshape(-1)
        ids = self.word_ids.index_select(0, flat_idx)  # (N*, T)
        tokens = self.embedding(ids)  # (N*, T, D)
        mask = (ids != 0).to(torch.int8)  # (N*, T)
        return (
            tokens.reshape(*leading_shape, self.max_length, self.output_dim),
            mask.reshape(*leading_shape, self.max_length),
        )


class PLMTextEncoder(TextEncoder):
    """Frozen-cached PLM token lookup keyed by news_idx.

    Returns tokens at the **native** PLM hidden size (e.g. 768 for
    BERT-base). The model's per-news pipeline (MHSA, additive pool, ...)
    runs at this dim and is responsible for projecting down to its
    target ``news_dim`` at the end of pool when needed. This preserves
    the full BERT signal through the per-news attention layers — the
    IP2-style design the reference NRMS+PLM literature uses.

    Args:
        plm_token_embeddings_by_id: ``(max_id+1, T, plm_dim)`` float32.
        plm_attention_mask_by_id: ``(max_id+1, T)`` int8 (0/1).
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
    ):
        super().__init__()
        cache = np.asarray(plm_token_embeddings_by_id)
        if cache.ndim != 3:
            raise ValueError(
                f"plm_token_embeddings_by_id must be 3D (num_news, T, plm_dim); "
                f"got shape {cache.shape}"
            )
        _, T, plm_dim = cache.shape
        self.plm_dim = int(plm_dim)
        self.output_dim = int(plm_dim)
        self.max_length = int(T)

        self.register_buffer(
            "token_embeddings",
            torch.from_numpy(cache).float(),
            persistent=False,
        )
        self.register_buffer(
            "attention_mask",
            torch.from_numpy(np.asarray(plm_attention_mask_by_id)).to(torch.int8),
            persistent=False,
        )

    @staticmethod
    def valid_mask(news_idx: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed id is non-zero."""
        return news_idx != 0

    def forward(self, news_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = news_idx.shape
        ids = news_idx.long().reshape(-1)
        tokens = self.token_embeddings.index_select(0, ids)
        mask = self.attention_mask.index_select(0, ids)
        return (
            tokens.reshape(*leading_shape, self.max_length, self.output_dim),
            mask.reshape(*leading_shape, self.max_length),
        )
