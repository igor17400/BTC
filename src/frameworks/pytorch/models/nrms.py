"""NRMS (Neural News Recommendation with Multi-Head Self-Attention) -- PyTorch."""

from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import NRMSConfig

from ..layers import AdditiveAttention, PLMTokenNewsEncoder
from .base import BaseModel


class NewsEncoder(nn.Module):
    """GloVe-token news encoder.

    Pipeline: Embedding → Dropout → MultiheadAttention → Dropout → AdditiveAttention.

    Honors the shared encoder contract:
        - ``forward(features, training=...) -> (*leading, news_dim)`` for any
          leading shape ``(*leading, T)`` (title-token sequence as last dim).
        - ``valid_mask(features) -> bool tensor of shape (*leading,)``.

    A GloVe-side news slot is "valid" if any token in its title is non-zero.
    """

    def __init__(self, config: NRMSConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config
        self.news_dim = config.embedding_size
        self.embedding_layer = embedding_layer

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.multiheads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
        )

    @staticmethod
    def valid_mask(features: torch.Tensor) -> torch.Tensor:
        """A GloVe news slot is valid when any token is non-zero."""
        return features.any(dim=-1)

    def forward(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode news titles.

        Args:
            features: ``(*leading, T)`` int token ids.
            training: Controls dropout behaviour.

        Returns:
            ``(*leading, news_dim)`` news vectors.
        """
        # Flatten arbitrary leading dims into a single batch axis for MHA.
        leading_shape = features.shape[:-1]
        T = features.shape[-1]
        inputs = features.reshape(-1, T)  # (B*, T)

        embedded = self.embedding_layer(inputs)  # (B*, T, E)
        y = self.dropout1(embedded)

        # Padding mask: True where padded (PyTorch convention for MHA key_padding_mask)
        key_padding_mask = inputs == 0  # (B*, T)
        fully_padded = key_padding_mask.all(dim=-1)  # (B*,) True = empty title

        # Temporarily unmask one position for fully-padded rows to prevent
        # MHA softmax over all -inf → NaN. The output for these rows will
        # be zeroed out below.
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        y, _ = self.multi_head_attention(y, y, y, key_padding_mask=key_padding_mask)
        y = self.dropout2(y)

        att_mask = ~(inputs == 0)  # recompute from original inputs
        news_repr = self.additive_attention(y, mask=att_mask)

        # Zero out representations for fully-padded inputs.
        if fully_padded.any():
            news_repr = news_repr.masked_fill(fully_padded.unsqueeze(-1), 0.0)

        return news_repr.reshape(*leading_shape, self.news_dim)


class UserEncoder(nn.Module):
    """User encoder: news_encoder(*history) → MultiheadAttention → AdditiveAttention.

    Encoder-agnostic: delegates per-slot encoding and the validity mask
    to ``news_encoder`` regardless of whether features are GloVe tokens
    or PLM news ids.
    """

    def __init__(self, config: NRMSConfig, news_encoder: nn.Module):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.browsed_news_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.multiheads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.user_additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
        )

    def forward(
        self, history_features: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode user browsing history.

        Args:
            history_features: Any shape the underlying ``news_encoder``
                accepts, with the leading axes being ``(batch, history_len)``.
            training: Controls dropout.

        Returns:
            ``(batch, embedding_size)`` user representations.
        """
        # Encode each history slot and gather slot-validity mask — both
        # delegate to the news encoder so this method is encoder-agnostic.
        click_title_presents = self.news_encoder(history_features, training=training)
        history_mask_valid = self.news_encoder.valid_mask(history_features)  # (B, H)

        key_padding_mask = ~history_mask_valid  # True = padded
        fully_empty = ~history_mask_valid.any(dim=-1)  # (B,) users with no history

        # Temporarily unmask one position for users with no history
        # to prevent MHA softmax over all -inf → NaN
        if fully_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_empty, 0] = False

        # Self-attention over browsed news
        y, _ = self.browsed_news_attention(
            click_title_presents,
            click_title_presents,
            click_title_presents,
            key_padding_mask=key_padding_mask,
        )

        # Additive attention
        user_repr = self.user_additive_attention(y, mask=history_mask_valid)

        # Zero out representations for users with no history
        if fully_empty.any():
            user_repr = user_repr.masked_fill(fully_empty.unsqueeze(-1), 0.0)

        return user_repr


class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention.

    Supports three scoring modes:
    - Training: softmax over K+1 candidates (1 positive + K negatives).
    - Single candidate: sigmoid score.
    - Multiple candidates: raw dot-product scores.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NRMSConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = NRMSConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        encoder_type = getattr(config.encoder, "type", "glove")
        if encoder_type == "glove":
            # Shared word embedding (initialised from pretrained GloVe).
            vocab_size = processed_news["vocab_size"]
            embeddings_matrix = processed_news["embeddings"]
            self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
            self.embedding_layer.weight = nn.Parameter(
                torch.tensor(embeddings_matrix, dtype=torch.float32)
            )
            self.news_encoder = NewsEncoder(config, self.embedding_layer)
        else:
            # PLM mode — token-level cache + model-side pooler.
            # Pooler hyperparameters come from the NRMS spec
            # (``config.text_pooler``) so the encoder yaml stays
            # model-agnostic.
            if "plm_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"NRMS with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id']. Call "
                    "attach_plm_embeddings(..., level='token') in the runner."
                )
            pooler = config.text_pooler
            self.news_encoder = PLMTokenNewsEncoder(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                news_dim=config.embedding_size,
                pooler_type=pooler.type,
                attention_query_dim=pooler.attention_query_dim,
                num_heads=pooler.num_heads,
                head_dim=pooler.head_dim,
                dropout_rate=pooler.dropout_rate,
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

    # ----- scoring helpers ------------------------------------------------

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score a training batch with softmax over candidates.

        Uses uniform feature keys: ``hist_features`` and ``cand_features``.
        The shape of these tensors depends on the configured encoder
        (GloVe = title tokens, PLM = parsed news ids) but the model
        is unaware — the encoder handles arbitrary leading shapes.
        """
        history = inputs["hist_features"]
        candidates = inputs["cand_features"]

        user_repr = self.user_encoder(history, training=True)  # (B, E)
        cand_repr = self.news_encoder(candidates, training=True)  # (B, C, E)

        # Dot-product scores (raw logits — loss applies softmax).
        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)  # (B, C)
        return scores
