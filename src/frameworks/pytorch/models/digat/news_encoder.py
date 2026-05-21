"""DIGAT news encoder (PyTorch).

MSA-based per-news encoder: embedding → MHA → additive attention.
Mirrors the original reference's ``NewsEncoder`` but with a custom
``MultiHeadSelfAttention`` (bias-free K, biased Q/V) to match reference
DIGAT parameter layout exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import DIGATConfig


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention for sequence pooling."""

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.affine = nn.Linear(feature_dim, attention_dim, bias=True)
        self.project = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.project(torch.tanh(self.affine(features))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (news encoder MSA).

    Custom implementation (not ``nn.MultiheadAttention``) to match the
    reference DIGAT exactly: a bias-free K projection and a biased Q/V
    projection, plus manual head reshape/transpose.
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(float(head_dim))
        self.W_Q = nn.Linear(d_model, self.out_dim, bias=True)
        self.W_K = nn.Linear(d_model, self.out_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out = (
            torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.out_dim)
        )
        return out


class NewsEncoder(nn.Module):
    """MSA-based news encoder: embedding → MHA → additive attention.

    GloVe mode reads token ids ``(B, num_news, T)`` and embeds them.
    PLM mode reads parsed news_idx ``(B, num_news)`` and looks up the
    cached PLM token features + attention mask via :class:`PLMTokenLookup`.
    """

    def __init__(
        self,
        config: DIGATConfig,
        word_embedding: nn.Module,
        *,
        encoder_type: str = "glove",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.word_embedding = word_embedding
        self.dropout = nn.Dropout(config.dropout_rate)
        self.msa = MultiHeadSelfAttention(
            config.embedding_size,
            config.msa_head_num,
            config.msa_head_dim,
        )
        self.news_embedding_dim = config.news_embedding_dim
        self.attention = AdditiveAttention(
            self.news_embedding_dim, config.attention_dim
        )

    def forward(
        self, title_text: torch.Tensor, title_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            title_text: GloVe — ``(batch_size, num_news, title_len)`` token ids.
                PLM — ``(batch_size, num_news)`` parsed news_idx.
            title_mask: GloVe — optional ``(B, num_news, title_len)`` bool mask
                (1 = valid). Ignored under PLM (the lookup returns its own mask).

        Returns:
            ``(batch_size, num_news, news_embedding_dim)`` news representations.
        """
        if self.encoder_type == "glove":
            batch_size, num_news, title_len = title_text.shape
            flat_text = title_text.reshape(batch_size * num_news, title_len)
            flat_mask = (
                title_mask.reshape(batch_size * num_news, title_len)
                if title_mask is not None
                else None
            )
            w = self.dropout(self.word_embedding(flat_text))
        else:
            batch_size, num_news = title_text.shape
            flat_idx = title_text.reshape(batch_size * num_news).long()
            tokens, mask = self.word_embedding(flat_idx)
            w = self.dropout(tokens)
            flat_mask = mask.bool()

        h = F.relu(self.msa(w))
        news_repr = self.attention(h, mask=flat_mask)
        return news_repr.view(batch_size, num_news, self.news_embedding_dim)
