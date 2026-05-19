"""NAML (Neural News Recommendation with Attentive Multi-View Learning) -- PyTorch."""

from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import NAMLConfig

from ..layers import AdditiveAttention, PLMTokenCNNEncoder, get_activation
from .base import BaseModel

# ------------------------------------------------------------------
# Sub-encoders
# ------------------------------------------------------------------


class TitleEncoder(nn.Module):
    """Title encoder: Embedding -> Dropout -> Conv1d -> Dropout -> AdditiveAttention."""

    def __init__(self, config: NAMLConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.activation = get_activation(config.activation)

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
        self.activation = get_activation(config.activation)

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
        self.activation = get_activation(config.activation)

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
        self.activation = get_activation(config.activation)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        embedded = self.embedding(inputs)
        projected = self.activation(self.projection(embedded))
        return projected.squeeze(1)


class NewsEncoder(nn.Module):
    """Multi-view news encoder combining title, abstract, category, subcategory.

    Encoder-aware unpacking: in GloVe mode the input is the original
    packed token layout; in PLM mode the input is a 3-column tensor
    ``[news_idx, category_id, subcategory_id]`` and the text encoders
    are :class:`PLMNewsEncoder` lookups (title + abstract caches).
    """

    def __init__(
        self,
        config: NAMLConfig,
        title_encoder: nn.Module,
        abstract_encoder: nn.Module,
        category_encoder: CategoryEncoder,
        subcategory_encoder: SubcategoryEncoder,
        *,
        encoder_type: str = "glove",
    ):
        super().__init__()
        self.config = config
        self.encoder_type = encoder_type
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
            inputs: GloVe mode → ``(batch, title_len + abstract_len + 2)``
                where the last two positions are category_id and subcategory_id.
                PLM mode → ``(batch, 3)`` = ``[news_idx, category, subcategory]``.

        Returns:
            (batch, cnn_filter_num) news representation.
        """
        cfg = self.config
        if self.encoder_type == "glove":
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
            subcategory_id = inputs[
                :, cfg.max_title_length + cfg.max_abstract_length + 1 :
            ]
            title_vec = self.title_encoder(title_tokens, training=training)
            abstract_vec = self.abstract_encoder(abstract_tokens, training=training)
        else:
            # PLM mode: column 0 = parsed news id, used by BOTH title and
            # abstract PLM lookups (each has its own cache). Columns 1, 2
            # are category / subcategory.
            news_idx = inputs[:, 0]
            category_id = inputs[:, 1:2]
            subcategory_id = inputs[:, 2:3]
            title_vec = self.title_encoder(news_idx, training=training)
            abstract_vec = self.abstract_encoder(news_idx, training=training)

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
            inputs: ``(batch, history_len, feature_size)`` packed features.
                GloVe mode: feature_size = title_len + abstract_len + 2.
                PLM mode: feature_size = 3 = [news_idx, category, subcategory].

        Returns:
            (batch, cnn_filter_num) user representation.
        """
        batch_size, hist_len, feat_size = inputs.shape

        # TimeDistributed
        flat = inputs.reshape(batch_size * hist_len, feat_size)
        news_vecs = self.news_encoder(flat, training=training)
        news_vecs = news_vecs.reshape(batch_size, hist_len, -1)

        # Validity mask: encoder-dependent.
        if self.news_encoder.encoder_type == "glove":
            # Title tokens contain at least one non-zero word.
            title_tokens = inputs[:, :, : self.config.max_title_length]
            history_mask = title_tokens.any(dim=-1)  # (B, H)
        else:
            # PLM mode: news_idx (column 0) is non-zero for real history slots.
            history_mask = inputs[:, :, 0] != 0

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
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = NAMLConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        encoder_type = getattr(getattr(config, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        if encoder_type == "glove":
            # Shared word embedding (initialised from pretrained GloVe).
            vocab_size = processed_news["vocab_size"]
            embeddings_matrix = processed_news["embeddings"]
            self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
            self.embedding_layer.weight = nn.Parameter(
                torch.tensor(embeddings_matrix, dtype=torch.float32)
            )
            self.title_encoder = TitleEncoder(config, self.embedding_layer)
            self.abstract_encoder = AbstractEncoder(config, self.embedding_layer)
        else:
            # PLM mode — token-level lookups + NAML's own Conv1d/AddAttn
            # pipeline. Each view (title, abstract) has its own cache;
            # the runner builds both via attach_plm_embeddings(..., level='token').
            for required in (
                "plm_token_embeddings_by_id",
                "plm_attention_mask_by_id",
                "plm_abstract_token_embeddings_by_id",
                "plm_abstract_attention_mask_by_id",
            ):
                if required not in processed_news:
                    raise KeyError(
                        f"NAML with encoder.type='{encoder_type}' requires "
                        f"processed_news['{required}']. Make sure the encoder "
                        "yaml sets text_field='title' and "
                        "text_field_abstract='abstract' and the runner has "
                        "called attach_plm_embeddings for both."
                    )
            plm_dim = int(processed_news["plm_dim"])
            self.title_encoder = PLMTokenCNNEncoder(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                news_dim=config.cnn_filter_num,
                plm_dim=plm_dim,
                cnn_kernel_size=config.cnn_kernel_size,
                dropout_rate=config.dropout_rate,
                word_attention_query_dim=config.word_attention_query_dim,
                activation=config.activation,
            )
            self.abstract_encoder = PLMTokenCNNEncoder(
                processed_news["plm_abstract_token_embeddings_by_id"],
                processed_news["plm_abstract_attention_mask_by_id"],
                news_dim=config.cnn_filter_num,
                plm_dim=plm_dim,
                cnn_kernel_size=config.cnn_kernel_size,
                dropout_rate=config.dropout_rate,
                word_attention_query_dim=config.word_attention_query_dim,
                activation=config.activation,
            )

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
            encoder_type=encoder_type,
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
        return self.score_training_batch(inputs)

    # ----- helpers --------------------------------------------------------

    def _concat_features(
        self, inputs: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        """Concatenate the per-view inputs into a single packed tensor.

        GloVe mode: [title_tokens | abstract_tokens | category | subcategory]
            along the last dim, producing
            ``(*, title_len + abstract_len + 2)``.

        PLM mode: [news_idx | category | subcategory] along the last dim,
            producing ``(*, 3)`` where ``news_idx`` is the parsed-int news
            id that both PLM caches index off.
        """
        if self.encoder_type != "glove":
            # PLM path: news_idx is the parsed news id (`hist_features` /
            # `cand_features` from the runner). Promote to (*, 1) and
            # concat with category / subcategory (also promoted to (*, 1)).
            news_idx = inputs[f"{prefix}_features"]
            parts = [news_idx.unsqueeze(-1)]
            if f"{prefix}_category" in inputs:
                cat = inputs[f"{prefix}_category"]
                if cat.dim() == news_idx.dim():
                    cat = cat.unsqueeze(-1)
                parts.append(cat)
            if f"{prefix}_subcategory" in inputs:
                subcat = inputs[f"{prefix}_subcategory"]
                if subcat.dim() == news_idx.dim():
                    subcat = subcat.unsqueeze(-1)
                parts.append(subcat)
            return torch.cat(parts, dim=-1)

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

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hist_concat = self._concat_features(inputs, "hist")
        cand_concat = self._concat_features(inputs, "cand")

        user_repr = self.user_encoder(hist_concat, training=True)

        B, C, F = cand_concat.shape
        flat_cand = cand_concat.reshape(B * C, F)
        cand_repr = self.news_encoder(flat_cand, training=True).reshape(B, C, -1)

        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
        return scores  # raw logits; loss applies log-softmax internally
