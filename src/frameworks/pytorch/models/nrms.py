"""NRMS (Neural News Recommendation with Multi-Head Self-Attention) -- PyTorch."""

from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import NRMSConfig

from ..layers import AdditiveAttention
from .base import BaseModel


class NewsEncoder(nn.Module):
    """News encoder: Embedding -> Dropout -> MultiheadAttention -> Dropout -> AdditiveAttention."""

    def __init__(self, config: NRMSConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config
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

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode news titles.

        Args:
            inputs: (batch, title_len) int token ids.
            training: Controls dropout behaviour.

        Returns:
            (batch, embedding_size) news representations.
        """
        # Word embedding + dropout
        embedded = self.embedding_layer(inputs)  # (B, T, E)
        y = self.dropout1(embedded)

        # Padding mask: True where padded (PyTorch convention for MHA key_padding_mask)
        key_padding_mask = inputs == 0  # (B, T)
        fully_padded = key_padding_mask.all(dim=-1)  # (B,) True = empty title

        # Temporarily unmask one position for fully-padded rows to prevent
        # MHA softmax over all -inf → NaN. The output for these rows will
        # be zeroed out below.
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        # Multi-head self-attention
        y, _ = self.multi_head_attention(y, y, y, key_padding_mask=key_padding_mask)
        y = self.dropout2(y)

        # Additive attention to get single vector
        # Mask for additive attention: True means *keep* (our convention)
        att_mask = ~(inputs == 0)  # recompute from original inputs
        news_repr = self.additive_attention(y, mask=att_mask)

        # Zero out representations for fully-padded inputs
        if fully_padded.any():
            news_repr = news_repr.masked_fill(fully_padded.unsqueeze(-1), 0.0)

        return news_repr


class UserEncoder(nn.Module):
    """User encoder: TimeDistributed(NewsEncoder) -> MultiheadAttention -> AdditiveAttention."""

    def __init__(self, config: NRMSConfig, news_encoder: NewsEncoder):
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

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode user browsing history.

        Args:
            inputs: (batch, history_len, title_len) int token ids.
            training: Controls dropout.

        Returns:
            (batch, embedding_size) user representations.
        """
        batch_size, hist_len, title_len = inputs.shape

        # TimeDistributed: reshape -> encode -> reshape back
        flat = inputs.reshape(batch_size * hist_len, title_len)
        news_vecs = self.news_encoder(flat, training=training)  # (B*H, E)
        click_title_presents = news_vecs.reshape(batch_size, hist_len, -1)  # (B, H, E)

        # History mask: True where at least one token is nonzero
        history_mask_valid = inputs.any(dim=-1)  # (B, H) True = valid
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

        # Shared word embedding (initialised from pretrained)
        vocab_size = processed_news["vocab_size"]
        embeddings_matrix = processed_news["embeddings"]
        self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # Build encoders
        self.news_encoder = NewsEncoder(config, self.embedding_layer)
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
        """Score a training batch with softmax over candidates."""
        history_tokens = inputs["hist_tokens"]
        candidate_tokens = inputs["cand_tokens"]

        user_repr = self.user_encoder(history_tokens, training=True)  # (B, E)

        # TimeDistributed trick: the news encoder expects 2D input (batch, title_len),
        # but we have 3D (batch, candidates, title_len). Flatten users x candidates
        # into one big batch, encode all articles at once, then reshape back.
        #   B = batch size (number of users)
        #   C = candidates (number of candidate news per user)
        #   T = title length (tokens per article)
        B, C, T = candidate_tokens.shape
        flat_cand = candidate_tokens.reshape(B * C, T)  # (B*C, T) — all articles flat
        cand_repr = self.news_encoder(flat_cand, training=True)  # (B*C, E)
        cand_repr = cand_repr.reshape(B, C, -1)  # (B, C, E) — grouped by user

        # Dot-product scores (raw logits — loss function handles softmax)
        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)  # (B, C)
        return scores
