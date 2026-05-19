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

from .attention_layers import AdditiveAttention
from .layer_utils import get_activation
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


class PLMTokenLookup(nn.Module):
    """Frozen token-level PLM lookup with optional projection.

    Returns BOTH the per-token features and the attention mask for the
    sequence — slots in as a replacement for the GloVe ``nn.Embedding``
    layer in models that need token-level features for downstream
    MHSA / cross-attention / Transformer (PP-Rec, TCCM, CAUM, CROWN,
    DIGAT, MINER, ...).

    The model is responsible for handling the mask wherever it
    previously derived a padding mask from ``token_ids == 0``.

    Args:
        plm_token_embeddings_by_id: ``(max_parsed_id + 1, T, plm_dim)``
            float32 — token-level frozen PLM outputs.
        plm_attention_mask_by_id: ``(max_parsed_id + 1, T)`` int8 (0/1).
        plm_dim: PLM hidden size (last axis of the cache).
        output_dim: Output channels. ``None`` → same as ``plm_dim`` (no
            projection); otherwise a trainable ``Linear(plm_dim, output_dim)``
            projects each token vector.
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
        *,
        plm_dim: int,
        output_dim: int | None = None,
    ):
        super().__init__()
        max_id, T, in_dim = plm_token_embeddings_by_id.shape
        if in_dim != plm_dim:
            raise ValueError(
                f"plm_dim mismatch: cache has {in_dim}, config says {plm_dim}"
            )
        self.plm_dim = int(plm_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(plm_dim)
        self.max_length = int(T)

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
        if output_dim is not None and output_dim != plm_dim:
            self.projection: nn.Module = nn.Linear(plm_dim, output_dim)
        else:
            self.projection = nn.Identity()

    @staticmethod
    def valid_mask(features: torch.Tensor) -> torch.Tensor:
        """A PLM news slot is valid when its parsed id is non-zero."""
        return features != 0

    def forward(
        self,
        news_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up cached token features + mask.

        Args:
            news_idx: int tensor of any shape ``(...,)`` of parsed news ids.

        Returns:
            features: ``(..., T, output_dim)`` float tensor.
            mask:     ``(..., T)`` int8 (0/1) attention mask.
        """
        leading_shape = news_idx.shape
        ids = news_idx.long().reshape(-1)
        tokens = self.token_embeddings.index_select(0, ids)
        mask = self.attention_mask.index_select(0, ids)
        out = self.projection(tokens)
        return (
            out.reshape(*leading_shape, self.max_length, self.output_dim),
            mask.reshape(*leading_shape, self.max_length),
        )


class PLMTokenCNNEncoder(nn.Module):
    """NAML-style news encoder over token-level PLM features.

    Equivalent to NAML's original ``TitleEncoder`` / ``AbstractEncoder``
    but with cached BERT-token features ``(num_news, T, plm_dim)``
    replacing the GloVe-Embedding output. The Conv1d + AdditiveAttention
    pipeline downstream is unchanged from NAML's design, so the model
    still gets per-token signal and learns its own n-gram / attention
    pool.

    Frozen lookup buffers are stored via ``register_buffer`` (not
    ``Parameter``) so they're moved with the model but ignored by the
    optimizer.

    Args:
        plm_token_embeddings_by_id: ``(max_parsed_id + 1, T, plm_dim)``
            float32 — token-level frozen PLM outputs, indexed by parsed
            news id. Row 0 is the padding zero block.
        plm_attention_mask_by_id: ``(max_parsed_id + 1, T)`` int8 (0/1)
            attention mask, indexed by parsed news id.
        news_dim: Output channel dimension (= NAML's ``cnn_filter_num``).
        plm_dim: Per-token PLM hidden size (input channels to Conv1d).
        cnn_kernel_size: Conv1d kernel size (default 3, NAML default).
        dropout_rate: Pre- and post-Conv dropout (mirrors NAML).
        word_attention_query_dim: AdditiveAttention query-vector dim.
        activation: Non-linearity after Conv1d (NAML default "relu").
    """

    def __init__(
        self,
        plm_token_embeddings_by_id: np.ndarray,
        plm_attention_mask_by_id: np.ndarray,
        *,
        news_dim: int,
        plm_dim: int,
        cnn_kernel_size: int = 3,
        dropout_rate: float = 0.2,
        word_attention_query_dim: int = 200,
        activation: str = "relu",
    ):
        super().__init__()
        max_id, T, in_dim = plm_token_embeddings_by_id.shape
        if in_dim != plm_dim:
            raise ValueError(
                f"plm_dim mismatch: cache has {in_dim}, config says {plm_dim}"
            )
        self.plm_dim = int(plm_dim)
        self.news_dim = int(news_dim)
        self.max_length = int(T)

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

        # NAML's view pipeline, but with PLM-dim inputs.
        self.dropout1 = nn.Dropout(dropout_rate)
        self.cnn = nn.Conv1d(
            in_channels=plm_dim,
            out_channels=news_dim,
            kernel_size=cnn_kernel_size,
            padding="same",
        )
        self.activation = get_activation(activation)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=news_dim,
            query_vec_dim=word_attention_query_dim,
        )

    @staticmethod
    def valid_mask(features: torch.Tensor) -> torch.Tensor:
        """A PLM news slot is valid when its parsed id is non-zero."""
        return features != 0

    def forward(
        self,
        features: torch.Tensor,
        training: bool = True,  # accepted for API parity
    ) -> torch.Tensor:
        """Look up token-level frozen vectors and run NAML's CNN+attn.

        Args:
            features: int tensor of any shape ``(...,)`` of parsed news ids.

        Returns:
            ``(..., news_dim)`` float tensor.
        """
        leading_shape = features.shape
        ids = features.long().reshape(-1)  # (N*,)

        # Lookup: (N*, T, plm_dim) and (N*, T)
        tokens = self.token_embeddings.index_select(0, ids)
        mask = self.attention_mask.index_select(0, ids).bool()

        y = self.dropout1(tokens)
        y = y.transpose(1, 2)  # (N*, plm_dim, T)
        y = self.activation(self.cnn(y))  # (N*, news_dim, T)
        y = y.transpose(1, 2)  # (N*, T, news_dim)
        y = self.dropout2(y)

        out = self.additive_attention(y, mask=mask)  # (N*, news_dim)
        return out.reshape(*leading_shape, self.news_dim)
