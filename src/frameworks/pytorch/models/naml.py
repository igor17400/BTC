"""NAML (Neural News Recommendation with Attentive Multi-View Learning) -- PyTorch."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import NAMLConfig

from ..layers import AdditiveAttention
from .base import BaseModel


def _get_activation(name: str):
    """Resolve a string activation name to a torch.nn.functional callable.

    Centralised so all sub-encoders honour ``NAMLConfig.activation`` instead
    of hardcoding ``F.relu``.
    """
    fn = getattr(F, name, None)
    if fn is None:
        raise ValueError(f"Unknown activation '{name}' for torch.nn.functional")
    return fn


# ------------------------------------------------------------------
# Sub-encoders
# ------------------------------------------------------------------


class TitleEncoder(nn.Module):
    """Title encoder: Embedding -> Dropout -> Conv1d -> Dropout -> AdditiveAttention."""

    def __init__(self, config: NAMLConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.activation = _get_activation(config.activation)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        # PyTorch Conv1d: (batch, channels, length)
        self.cnn = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.cnn_filter_num,
            kernel_size=config.cnn_kernel_size,
            padding="same",
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.word_attention_query_dim,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode a title.

        Args:
            inputs: (batch, title_len) int token ids.

        Returns:
            (batch, cnn_filter_num) title representation.
        """
        embedded = self.embedding_layer(inputs)  # (B, T, E)
        y = self.dropout1(embedded)

        # Conv1d expects (B, C, T), transpose and back
        y = y.transpose(1, 2)  # (B, E, T)
        y = self.activation(self.cnn(y))  # (B, F, T)
        y = y.transpose(1, 2)  # (B, T, F)
        y = self.dropout2(y)

        padding_mask = inputs != 0  # (B, T)
        return self.additive_attention(y, mask=padding_mask)


class AbstractEncoder(nn.Module):
    """Abstract encoder: same architecture as TitleEncoder."""

    def __init__(self, config: NAMLConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.activation = _get_activation(config.activation)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.cnn = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.cnn_filter_num,
            kernel_size=config.cnn_kernel_size,
            padding="same",
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.word_attention_query_dim,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded)

        y = y.transpose(1, 2)
        y = self.activation(self.cnn(y))
        y = y.transpose(1, 2)
        y = self.dropout2(y)

        padding_mask = inputs != 0
        return self.additive_attention(y, mask=padding_mask)


class CategoryEncoder(nn.Module):
    """Category encoder: Embedding -> Linear projection."""

    def __init__(self, config: NAMLConfig, num_categories: int):
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, config.category_embedding_dim)
        self.projection = nn.Linear(
            config.category_embedding_dim, config.cnn_filter_num
        )
        self.activation = _get_activation(config.activation)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode category ids.

        Args:
            inputs: (batch, 1) int category ids.

        Returns:
            (batch, cnn_filter_num) category representation.
        """
        embedded = self.embedding(inputs)  # (B, 1, cat_dim)
        projected = self.activation(self.projection(embedded))  # (B, 1, cnn_filter_num)
        return projected.squeeze(1)  # (B, cnn_filter_num)


class SubcategoryEncoder(nn.Module):
    """Subcategory encoder: Embedding -> Linear projection."""

    def __init__(self, config: NAMLConfig, num_subcategories: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_subcategories + 1, config.subcategory_embedding_dim
        )
        self.projection = nn.Linear(
            config.subcategory_embedding_dim, config.cnn_filter_num
        )
        self.activation = _get_activation(config.activation)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        embedded = self.embedding(inputs)
        projected = self.activation(self.projection(embedded))
        return projected.squeeze(1)


class NewsEncoder(nn.Module):
    """Multi-view news encoder combining title, abstract, category, subcategory."""

    def __init__(
        self,
        config: NAMLConfig,
        title_encoder: TitleEncoder,
        abstract_encoder: AbstractEncoder,
        category_encoder: CategoryEncoder,
        subcategory_encoder: SubcategoryEncoder,
    ):
        super().__init__()
        self.config = config
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        # View-level attention (4 views, each cnn_filter_num dim)
        self.view_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.view_attention_query_dim,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode a news article from its concatenated features.

        Args:
            inputs: (batch, title_len + abstract_len + 2) where the last two
                    positions are category_id and subcategory_id.

        Returns:
            (batch, cnn_filter_num) news representation.
        """
        cfg = self.config
        title_tokens = inputs[:, : cfg.max_title_length]
        abstract_tokens = inputs[
            :, cfg.max_title_length : cfg.max_title_length + cfg.max_abstract_length
        ]
        category_id = inputs[
            :,
            cfg.max_title_length + cfg.max_abstract_length : cfg.max_title_length
            + cfg.max_abstract_length
            + 1,
        ]
        subcategory_id = inputs[:, cfg.max_title_length + cfg.max_abstract_length + 1 :]

        title_vec = self.title_encoder(title_tokens, training=training)
        abstract_vec = self.abstract_encoder(abstract_tokens, training=training)
        category_vec = self.category_encoder(category_id, training=training)
        subcategory_vec = self.subcategory_encoder(subcategory_id, training=training)

        # Stack views: (batch, 4, cnn_filter_num)
        views = torch.stack(
            [title_vec, abstract_vec, category_vec, subcategory_vec], dim=1
        )
        return self.view_attention(views)


class UserEncoder(nn.Module):
    """User encoder: TimeDistributed(NewsEncoder) -> AdditiveAttention."""

    def __init__(self, config: NAMLConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.user_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.user_attention_query_dim,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode user history.

        Args:
            inputs: (batch, history_len, feature_size) concatenated features.

        Returns:
            (batch, cnn_filter_num) user representation.
        """
        batch_size, hist_len, feat_size = inputs.shape

        # TimeDistributed
        flat = inputs.reshape(batch_size * hist_len, feat_size)
        news_vecs = self.news_encoder(flat, training=training)
        news_vecs = news_vecs.reshape(batch_size, hist_len, -1)

        # Mask: valid if title portion has any nonzero token
        title_tokens = inputs[:, :, : self.config.max_title_length]
        history_mask = title_tokens.any(dim=-1)  # (B, H)

        return self.user_attention(news_vecs, mask=history_mask)


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------


class NAML(BaseModel):
    """Neural News Recommendation with Attentive Multi-View Learning."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NAMLConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = NAMLConfig(**kwargs)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        # Shared word embedding
        vocab_size = processed_news["vocab_size"]
        embeddings_matrix = processed_news["embeddings"]
        self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # View encoders
        self.title_encoder = TitleEncoder(config, self.embedding_layer)
        self.abstract_encoder = AbstractEncoder(config, self.embedding_layer)
        self.category_encoder = CategoryEncoder(
            config, processed_news["num_categories"]
        )
        self.subcategory_encoder = SubcategoryEncoder(
            config, processed_news["num_subcategories"]
        )

        # Composite encoders
        self.news_encoder = NewsEncoder(
            config,
            self.title_encoder,
            self.abstract_encoder,
            self.category_encoder,
            self.subcategory_encoder,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass for training. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        return self._score_training(inputs)

    # ----- helpers --------------------------------------------------------

    def _concat_features(
        self, inputs: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        """Concatenate title, abstract, category, subcategory along last dim."""
        parts = [inputs[f"{prefix}_tokens"]]

        if f"{prefix}_abstract_tokens" in inputs:
            parts.append(inputs[f"{prefix}_abstract_tokens"])

        if f"{prefix}_category" in inputs:
            cat = inputs[f"{prefix}_category"]
            if cat.dim() == len(parts[0].shape) - 1:
                cat = cat.unsqueeze(-1)
            parts.append(cat)

        if f"{prefix}_subcategory" in inputs:
            subcat = inputs[f"{prefix}_subcategory"]
            if subcat.dim() == len(parts[0].shape) - 1:
                subcat = subcat.unsqueeze(-1)
            parts.append(subcat)

        return torch.cat(parts, dim=-1)

    def _score_training(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hist_concat = self._concat_features(inputs, "hist")
        cand_concat = self._concat_features(inputs, "cand")

        user_repr = self.user_encoder(hist_concat, training=True)

        B, C, F = cand_concat.shape
        flat_cand = cand_concat.reshape(B * C, F)
        cand_repr = self.news_encoder(flat_cand, training=True).reshape(B, C, -1)

        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
        return scores  # raw logits; loss applies log-softmax internally

