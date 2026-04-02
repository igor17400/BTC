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

        self.qkv_dim = config.multiheads * config.head_dim
        self.input_proj = nn.Linear(config.embedding_size, self.qkv_dim)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.qkv_dim,
            num_heads=config.multiheads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=self.qkv_dim,
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
        if not training:
            self.eval()
        else:
            self.train()

        # Word embedding + project to qkv_dim + dropout
        embedded = self.embedding_layer(inputs)  # (B, T, E)
        y = self.input_proj(embedded)  # (B, T, qkv_dim)
        y = self.dropout1(y)

        # Padding mask: True where padded (PyTorch convention for MHA key_padding_mask)
        key_padding_mask = inputs == 0  # (B, T)

        # Multi-head self-attention
        y, _ = self.multi_head_attention(y, y, y, key_padding_mask=key_padding_mask)
        y = self.dropout2(y)

        # Additive attention to get single vector
        # Mask for additive attention: True means *keep* (our convention)
        att_mask = ~key_padding_mask  # (B, T) True = valid
        news_repr = self.additive_attention(y, mask=att_mask)
        return news_repr


class UserEncoder(nn.Module):
    """User encoder: TimeDistributed(NewsEncoder) -> MultiheadAttention -> AdditiveAttention."""

    def __init__(self, config: NRMSConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.qkv_dim = config.multiheads * config.head_dim

        self.browsed_news_attention = nn.MultiheadAttention(
            embed_dim=self.qkv_dim,
            num_heads=config.multiheads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.user_additive_attention = AdditiveAttention(
            input_dim=self.qkv_dim,
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

        # Self-attention over browsed news
        y, _ = self.browsed_news_attention(
            click_title_presents,
            click_title_presents,
            click_title_presents,
            key_padding_mask=key_padding_mask,
        )

        # Additive attention
        user_repr = self.user_additive_attention(y, mask=history_mask_valid)
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
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = NRMSConfig(**kwargs)
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
        """Main forward pass.

        Args:
            inputs: Dictionary with keys depending on mode:
                Training:   ``hist_tokens``, ``cand_tokens``
                Inference:  ``hist_tokens``, ``cand_tokens`` (multi) or
                            ``history_tokens``, ``single_candidate_tokens`` (single)
            training: Whether in training mode.

        Returns:
            Scores tensor.
        """
        if training:
            return self._score_training(inputs)
        elif "single_candidate_tokens" in inputs:
            return self._score_single(inputs)
        else:
            return self._score_multi(inputs)

    # ----- scoring helpers ------------------------------------------------

    def _score_training(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Softmax scores for training batch."""
        history_tokens = inputs["hist_tokens"]
        candidate_tokens = inputs["cand_tokens"]

        user_repr = self.user_encoder(history_tokens, training=True)  # (B, E)

        # TimeDistributed over candidates
        B, C, T = candidate_tokens.shape
        flat_cand = candidate_tokens.reshape(B * C, T)
        cand_repr = self.news_encoder(flat_cand, training=True)  # (B*C, E)
        cand_repr = cand_repr.reshape(B, C, -1)  # (B, C, E)

        # Dot-product scores + softmax
        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)  # (B, C)
        return torch.softmax(scores, dim=-1)

    def _score_single(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sigmoid score for a single candidate."""
        user_repr = self.user_encoder(inputs["history_tokens"], training=False)
        cand_repr = self.news_encoder(inputs["single_candidate_tokens"], training=False)
        score = torch.sum(cand_repr * user_repr, dim=-1, keepdim=True)
        return torch.sigmoid(score)

    def _score_multi(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Raw dot-product scores for multiple candidates (evaluation)."""
        history_tokens = inputs["hist_tokens"]
        candidate_tokens = inputs["cand_tokens"]

        user_repr = self.user_encoder(history_tokens, training=False)

        B, C, T = candidate_tokens.shape
        flat_cand = candidate_tokens.reshape(B * C, T)
        cand_repr = self.news_encoder(flat_cand, training=False)
        cand_repr = cand_repr.reshape(B, C, -1)

        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
        return torch.sigmoid(scores)
