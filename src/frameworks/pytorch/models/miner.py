"""MINER (Multi-Interest Matching Network) -- PyTorch.

Reference: Li et al., "MINER: Multi-Interest Matching Network for News
Recommendation", Findings of ACL 2022.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import MINERConfig

from ..layers import AdditiveAttention
from .base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(nn.Module):
    """Encode a news article from its title token sequence.

    Pipeline: Embedding -> Dropout -> MultiHeadSelfAttention -> Dropout
    -> AdditiveAttention -> news vector.
    """

    def __init__(self, config: MINERConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """inputs: (batch, title_length) -> (batch, embedding_size)."""
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded)

        # PyTorch MHA: key_padding_mask True = masked (opposite of JAX)
        key_padding_mask = inputs == 0
        fully_padded = key_padding_mask.all(dim=-1)

        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        y, _ = self.multi_head_attention(y, y, y, key_padding_mask=key_padding_mask)
        y = self.dropout2(y)

        att_mask = ~(inputs == 0)  # True = keep
        news_repr = self.additive_attention(y, mask=att_mask)

        if fully_padded.any():
            news_repr = news_repr.masked_fill(fully_padded.unsqueeze(-1), 0.0)

        return news_repr


# ---------------------------------------------------------------------------
# Poly Attention
# ---------------------------------------------------------------------------


class PolyAttention(nn.Module):
    """Extract K interest vectors from clicked news via learnable context codes."""

    def __init__(self, config: MINERConfig):
        super().__init__()
        K = config.num_interest_vectors
        E = config.embedding_size
        D = config.context_code_dim

        self.context_codes = nn.Parameter(torch.empty(K, D))
        self.linear = nn.Linear(E, D, bias=False)

        # Init: xavier_uniform with gain=tanh (~1.667)
        nn.init.xavier_uniform_(self.context_codes, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("tanh"))

    def forward(
        self,
        news_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """news_embeddings: (B, M, E), mask: (B, M) bool True=keep -> (B, K, E)."""
        projected = torch.tanh(self.linear(news_embeddings))  # (B, M, D)

        logits = torch.einsum("bmd,kd->bkm", projected, self.context_codes)

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)

        weights = F.softmax(logits, dim=-1)  # (B, K, M)

        return torch.einsum("bkm,bme->bke", weights, news_embeddings)


# ---------------------------------------------------------------------------
# User encoder
# ---------------------------------------------------------------------------


class UserEncoder(nn.Module):
    """Encode user history into interest vectors.

    Returns mean of K interest vectors for eval compatibility.
    Use ``encode_interests()`` for the full K vectors.
    """

    def __init__(self, config: MINERConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.poly_attention = PolyAttention(config)

    def _encode_news_history(
        self, inputs: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode history and return (news_embeds, mask)."""
        B, H, T = inputs.shape
        flat = inputs.reshape(B * H, T)
        flat_vecs = self.news_encoder(flat, training=training)
        news_embeds = flat_vecs.reshape(B, H, -1)
        mask = inputs.any(dim=-1)  # (B, H) True = valid
        return news_embeds, mask

    def encode_interests(
        self, inputs: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode history into K interest vectors. Returns (B, K, E)."""
        news_embeds, mask = self._encode_news_history(inputs, training=training)
        return self.poly_attention(news_embeds, mask=mask)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode history into a single user vector (mean of K interests)."""
        interest_vecs = self.encode_interests(inputs, training=training)
        return interest_vecs.mean(dim=1)


# ---------------------------------------------------------------------------
# Disagreement regularization
# ---------------------------------------------------------------------------


def _disagreement_loss(interest_vecs: torch.Tensor) -> torch.Tensor:
    """Compute L_D (Eq. 3) — mean off-diagonal cosine similarity."""
    K = interest_vecs.shape[1]

    norms = interest_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = interest_vecs / norms.detach()  # stop_gradient on norms

    cos_sim = torch.bmm(normalized, normalized.transpose(1, 2))  # (B, K, K)

    diag_mask = 1.0 - torch.eye(K, device=interest_vecs.device)
    cos_sim = cos_sim * diag_mask.unsqueeze(0)

    return cos_sim.mean()


# ---------------------------------------------------------------------------
# Full MINER model
# ---------------------------------------------------------------------------


class MINER(BaseModel):
    """Multi-Interest Matching Network for News Recommendation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: MINERConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = MINERConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        # Shared word embedding
        vocab_size = int(processed_news["vocab_size"])
        embeddings_matrix = processed_news["embeddings"]
        self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # Sub-modules
        self.news_encoder = NewsEncoder(config, self.embedding_layer)
        self.user_encoder = UserEncoder(config, self.news_encoder)

        # Target-aware attention: projects INTERESTS
        self.W_e = nn.Linear(config.embedding_size, config.embedding_size)

        # Cache for disagreement loss
        self._cached_d_loss = torch.tensor(0.0)

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Return beta * L_D computed during the last forward pass."""
        return self.config.disagreement_beta * self._cached_d_loss

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """MINER-weighted training scoring."""
        hist_tokens = inputs["hist_tokens"]
        cand_tokens = inputs["cand_tokens"]

        # Get K interest vectors
        interest_vecs = self.user_encoder.encode_interests(
            hist_tokens, training=training
        )  # (B, K, E)

        # Compute disagreement and cache it
        if training and self.config.disagreement_beta > 0:
            self._cached_d_loss = _disagreement_loss(interest_vecs)

        # Encode candidates
        B, C, T = cand_tokens.shape
        flat_cands = cand_tokens.reshape(B * C, T)
        flat_cand_vecs = self.news_encoder(flat_cands, training=training)
        cand_embeds = flat_cand_vecs.reshape(B, C, -1)  # (B, C, E)

        # Per-interest scores: (B, K, C)
        interest_scores = torch.einsum("bke,bce->bkc", interest_vecs, cand_embeds)

        # Target-aware attention: project INTERESTS
        projected_interests = F.gelu(self.W_e(interest_vecs))  # (B, K, E)

        # Each candidate attends over K projected interests: (B, C, K)
        agg_logits = torch.einsum("bce,bke->bck", cand_embeds, projected_interests)
        agg_weights = F.softmax(agg_logits, dim=2)  # softmax over K

        # Aggregate: (B, C)
        scores = (agg_weights * interest_scores.permute(0, 2, 1)).sum(dim=2)

        return scores
