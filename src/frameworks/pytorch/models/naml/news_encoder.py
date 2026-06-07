"""NAML news encoder (PyTorch) — encoder-agnostic, multi-view.

Four parallel view encoders feed a view-level :class:`AdditiveAttention`:

  - Title view:        text_encoder(news_idx) -> (tokens, mask)
                           -> Dropout -> Conv1d(text_dim -> cnn_filter_num)
                           -> activation -> Dropout
                           -> AdditiveAttention with mask
  - Abstract view:     same pipeline, separate text_encoder + CNN.
  - Category view:     Embedding -> Linear(cat_dim -> cnn_filter_num) -> activation
  - Subcategory view:  Embedding -> Linear(subcat_dim -> cnn_filter_num) -> activation

Each view output lives at ``cnn_filter_num``; the four are stacked and
pooled by view attention into a single news vector.

The encoder accepts a packed integer feature tensor with columns
``[news_idx | category | subcategory]`` so the same forward serves
both training-time and eval-time call sites. The model is unaware of
whether GloVe or PLM is in use — both flow through the same
:class:`TextEncoder` interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import NAMLConfig

from ...layers import AdditiveAttention, TextEncoder, get_activation


class _TextViewEncoder(nn.Module):
    """Title / abstract pipeline: text_encoder -> Conv1d -> additive pool."""

    def __init__(
        self,
        config: NAMLConfig,
        text_encoder: TextEncoder,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.activation = get_activation(config.activation)

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
            query_vec_dim=config.word_attention_query_dim,
        )

    def forward(self, news_idx: torch.Tensor, training: bool = True) -> torch.Tensor:
        """news_idx: ``(N*,)`` ints → ``(N*, cnn_filter_num)`` vector."""
        tokens, mask = self.text_encoder(news_idx)  # (N*, T, text_dim), (N*, T)
        y = self.dropout1(tokens)

        # Conv1d expects (B, C, T); transpose, apply, transpose back.
        y = y.transpose(1, 2)
        y = self.activation(self.cnn(y))
        y = y.transpose(1, 2)
        y = self.dropout2(y)

        valid = mask != 0
        return self.additive_attention(y, mask=valid)


class _StructuredViewEncoder(nn.Module):
    """Category / subcategory pipeline: Embedding -> Linear -> activation."""

    def __init__(self, config: NAMLConfig, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, config.cnn_filter_num)
        self.activation = get_activation(config.activation)

    def forward(self, class_id: torch.Tensor, training: bool = True) -> torch.Tensor:
        """class_id: ``(N*,)`` ints → ``(N*, cnn_filter_num)`` vector."""
        embedded = self.embedding(class_id)
        return self.activation(self.projection(embedded))


class NewsEncoder(nn.Module):
    """Multi-view news encoder. Outputs ``(*leading, cnn_filter_num)``.

    Args:
        config: NAML hyperparameters.
        title_text_encoder: :class:`TextEncoder` for the title cache.
        abstract_text_encoder: :class:`TextEncoder` for the abstract cache,
            or ``None`` to drop the abstract view (``process_abstract=False``).
        num_categories: Size of category vocab (excludes padding index 0).
        num_subcategories: Size of subcategory vocab (excludes padding index 0).
    """

    def __init__(
        self,
        config: NAMLConfig,
        title_text_encoder: TextEncoder,
        abstract_text_encoder: TextEncoder | None,
        num_categories: int,
        num_subcategories: int,
    ):
        super().__init__()
        self.config = config
        self.news_dim = config.cnn_filter_num

        self.title_view = _TextViewEncoder(config, title_text_encoder)
        # Optional abstract view — absent when process_abstract=False, in which
        # case the view-attention pools over title + category + subcategory only.
        self.abstract_view = (
            _TextViewEncoder(config, abstract_text_encoder)
            if abstract_text_encoder is not None
            else None
        )
        self.category_view = _StructuredViewEncoder(
            config, num_categories, config.category_embedding_dim
        )
        self.subcategory_view = _StructuredViewEncoder(
            config, num_subcategories, config.subcategory_embedding_dim
        )
        self.view_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.view_attention_query_dim,
        )

    @staticmethod
    def valid_mask(packed_features: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero."""
        return packed_features[..., 0] != 0

    def forward(
        self, packed_features: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode a batch of news articles.

        Args:
            packed_features: ``(*leading, 3)`` int tensor with column order
                ``[news_idx | category_id | subcategory_id]``.
            training: Controls dropout.

        Returns:
            ``(*leading, cnn_filter_num)`` news vectors.
        """
        leading_shape = packed_features.shape[:-1]
        flat = packed_features.reshape(-1, 3)
        news_idx = flat[:, 0]
        category_id = flat[:, 1]
        subcategory_id = flat[:, 2]

        title_vec = self.title_view(news_idx, training=training)
        category_vec = self.category_view(category_id, training=training)
        subcategory_vec = self.subcategory_view(subcategory_id, training=training)

        # Stack views: (N*, num_views, cnn_filter_num). Abstract is included
        # only when present (process_abstract=True), giving 4 views; otherwise
        # 3. View-attention pools over whatever views are stacked.
        view_vecs = [title_vec]
        if self.abstract_view is not None:
            view_vecs.append(self.abstract_view(news_idx, training=training))
        view_vecs += [category_vec, subcategory_vec]

        views = torch.stack(view_vecs, dim=1)
        pooled = self.view_attention(views)  # (N*, cnn_filter_num)
        return pooled.reshape(*leading_shape, self.news_dim)
