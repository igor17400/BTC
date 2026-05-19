"""LSTUR news encoder (PyTorch) — encoder-agnostic.

CNN-over-tokens title pipeline + optional category/subcategory embeddings.
Mirrors NAML's title view but emits a single news vector by concatenating
the title CNN pool with the structured-view embeddings (no view-level
attention).

Pipeline (single code path for GloVe and PLM):
    text_encoder(news_idx) -> (tokens, mask)          # tokens @ text_dim
        -> Dropout -> Conv1d(text_dim -> cnn_filter_num)
        -> activation -> Dropout
        -> AdditiveAttention with mask                # pool @ cnn_filter_num
    cat_embedding(category_id)                        # @ cat_dim   (optional)
    subcat_embedding(subcategory_id)                  # @ subcat_dim (optional)
    -> concat -> news vector @ output_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import LSTURConfig

from ...layers import AdditiveAttention, TextEncoder, get_activation


class _CategoryEmbedding(nn.Module):
    """LSTUR-style structured-view embedding (no projection)."""

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)

    def forward(self, class_id: torch.Tensor) -> torch.Tensor:
        """``class_id``: ``(N*,)`` ints → ``(N*, embedding_dim)`` vector."""
        return self.embedding(class_id)


class NewsEncoder(nn.Module):
    """LSTUR news encoder. Outputs ``(*leading, output_dim)`` where

        output_dim = cnn_filter_num
                   + category_embedding_dim    (if use_category)
                   + subcategory_embedding_dim (if use_subcategory)

    Args:
        config: LSTUR hyperparameters.
        text_encoder: :class:`TextEncoder` for the title cache (GloVe or PLM).
        num_categories: Category vocab size (excludes padding index 0).
            Ignored when ``config.use_category`` is False.
        num_subcategories: Subcategory vocab size (excludes padding index 0).
            Ignored when ``config.use_subcategory`` is False.

    Input contract (mirrors NAML for the packed case):
        - ``(..., 1+use_cat+use_subcat)`` packed int tensor with column order
          ``[news_idx | category_id | subcategory_id]``. Trailing columns
          drop when their flag is False.
        - ``(...,)`` plain int tensor when neither flag is set.
    """

    def __init__(
        self,
        config: LSTURConfig,
        text_encoder: TextEncoder,
        num_categories: int = 0,
        num_subcategories: int = 0,
    ):
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder
        self.activation = get_activation(config.cnn_activation)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.cnn = nn.Conv1d(
            in_channels=text_encoder.output_dim,
            out_channels=config.cnn_filter_num,
            kernel_size=config.cnn_kernel_size,
            padding="same",
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.attention_hidden_dim,
        )

        self.category_embedding: nn.Module | None = None
        if config.use_category:
            self.category_embedding = _CategoryEmbedding(
                num_categories, config.category_embedding_dim
            )
        self.subcategory_embedding: nn.Module | None = None
        if config.use_subcategory:
            self.subcategory_embedding = _CategoryEmbedding(
                num_subcategories, config.subcategory_embedding_dim
            )

        self._n_cols = 1 + int(config.use_category) + int(config.use_subcategory)

    @property
    def output_dim(self) -> int:
        """Final per-news vector dim — used to size the user encoder GRU."""
        dim = self.config.cnn_filter_num
        if self.category_embedding is not None:
            dim += self.config.category_embedding_dim
        if self.subcategory_embedding is not None:
            dim += self.config.subcategory_embedding_dim
        return dim

    @staticmethod
    def valid_mask(features: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero.

        Accepts both packed ``(..., k)`` and plain ``(...,)`` int tensors.
        """
        if features.dim() == 0 or features.shape[-1] == 0:
            return features != 0
        # Heuristic: when the last axis is small (<= 4) and the tensor has
        # at least 2 dims, treat it as a packed column axis. Otherwise the
        # input is a plain news_idx tensor.
        if features.dim() >= 2 and features.shape[-1] <= 4:
            return features[..., 0] != 0
        return features != 0

    def _split_columns(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return ``(news_idx, category_id, subcategory_id)`` from inputs.

        When ``self._n_cols == 1`` the input may be a plain int tensor
        (no trailing column axis) or a packed ``(..., 1)`` tensor — both
        forms unwrap to ``news_idx`` with no category/subcategory tensors.
        """
        if self._n_cols == 1:
            news_idx = (
                inputs if inputs.shape[-1] != 1 or inputs.dim() == 1 else inputs[..., 0]
            )
            return news_idx, None, None

        news_idx = inputs[..., 0]
        col = 1
        category_id: torch.Tensor | None = None
        subcategory_id: torch.Tensor | None = None
        if self.category_embedding is not None:
            category_id = inputs[..., col]
            col += 1
        if self.subcategory_embedding is not None:
            subcategory_id = inputs[..., col]
        return news_idx, category_id, subcategory_id

    def _encode_title(self, news_idx: torch.Tensor) -> torch.Tensor:
        """Encode title tokens through TextEncoder + CNN + additive pool.

        Args:
            news_idx: ``(N*,)`` int tensor of parsed news ids.

        Returns:
            ``(N*, cnn_filter_num)`` title vectors.
        """
        tokens, mask = self.text_encoder(news_idx)  # (N*, T, text_dim), (N*, T)
        y = self.dropout1(tokens)

        # Conv1d expects (B, C, T); transpose, apply, transpose back.
        y = y.transpose(1, 2)
        y = self.activation(self.cnn(y))
        y = y.transpose(1, 2)
        y = self.dropout2(y)

        valid = mask != 0
        return self.additive_attention(y, mask=valid)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode a batch of news articles.

        Args:
            inputs: Packed ``(..., k)`` int with column order
                ``[news_idx | category_id | subcategory_id]`` (trailing
                columns drop when their flag is False), or plain
                ``(...,)`` int when no structured views are enabled.
            training: Controls dropout.

        Returns:
            ``(..., output_dim)`` news vectors.
        """
        news_idx, category_id, subcategory_id = self._split_columns(inputs)
        leading_shape = news_idx.shape

        flat_news_idx = news_idx.reshape(-1)
        title_vec = self._encode_title(flat_news_idx)  # (N*, cnn_filter_num)

        parts: list[torch.Tensor] = [title_vec]
        if self.category_embedding is not None:
            assert category_id is not None
            parts.append(self.category_embedding(category_id.reshape(-1)))
        if self.subcategory_embedding is not None:
            assert subcategory_id is not None
            parts.append(self.subcategory_embedding(subcategory_id.reshape(-1)))

        news_vec = torch.cat(parts, dim=-1) if len(parts) > 1 else title_vec
        return news_vec.reshape(*leading_shape, self.output_dim)
