"""GLORY local news encoder + attention primitives (PyTorch).

The attention primitives (``ScaledDotProductAttention``,
``MultiHeadAttention``, ``AttentionPooling``) follow GLORY's reference
``src/models/base/layers.py`` exactly — including the non-standard
masking scheme (``exp`` then mask) — and are shared by every other
encoder in the package.

GLORY is currently **GloVe-only**: the NewsEncoder consumes title
token ids directly from columns [0:title_size] of the packed
``news_features`` tensor. A PLM TextEncoder path was attempted but
removed because GLORY's setup pipeline pre-remaps news ids to row
positions (``build_news_feature_matrix:84`` puts ``np.arange(num_news)``
in the last column), while the shared TextEncoder expects parsed
news ids. Re-introducing PLM here requires either a row-indexed
PLM cache or dropping ``id_remap`` from GLORY's pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.core.models.configs import GLORYConfig


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention used inside ``MultiHeadAttention``.

    Note: GLORY's reference uses a non-standard masking scheme — scores
    are exponentiated before masking, then re-normalised.  We replicate
    that behaviour exactly to match the paper's initialization and
    training dynamics.
    """

    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        # Promote exp + normalize to fp32: bare torch.exp is NOT
        # autocast-promoted, so under fp16 AMP scores > ~11 overflow to
        # inf and scores < ~-16 underflow to 0, killing the attention
        # distribution. Measured ~+0.005 AUC on GLORY MIND-small.
        scores = torch.exp(scores.float())
        if attn_mask is not None:
            scores = scores * attn_mask.unsqueeze(dim=-2).float()
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(attn.to(V.dtype), V)


class MultiHeadAttention(nn.Module):
    """Multi-head attention matching GLORY's Q/K/V projection choice."""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        head_num: int,
        head_dim: int,
        residual: bool = False,
    ):
        super().__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.residual = residual
        out_dim = head_num * head_dim

        self.W_Q = nn.Linear(key_size, out_dim, bias=True)
        self.W_K = nn.Linear(query_size, out_dim, bias=False)
        self.W_V = nn.Linear(value_size, out_dim, bias=True)
        self.attn = ScaledDotProductAttention(head_dim)

        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_V.bias)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.head_num, -1)
        q = (
            self.W_Q(Q)
            .view(batch_size, -1, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.W_K(K)
            .view(batch_size, -1, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.W_V(V)
            .view(batch_size, -1, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        ctx = self.attn(q, k, v, mask)
        out = (
            ctx.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head_num * self.head_dim)
        )
        return out + Q if self.residual else out


class AttentionPooling(nn.Module):
    """Additive attention pooling over a sequence of vectors.

    Follows GLORY's implementation exactly:
    ``alpha = softmax(v^T tanh(W x)) / (sum + eps)`` with multiplicative
    masking (consistent with :class:`ScaledDotProductAttention`).
    """

    def __init__(self, emb_size: int, hidden_size: int):
        super().__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)
        # NOTE: deliberately NOT explicit-initing here. PyTorch nn.Linear
        # default is kaiming_uniform_(a=sqrt(5)) — same as the reference
        # GLORY (which has an `initialize()` method but never calls it
        # from main.py). Earlier code Xavier-inited explicitly which
        # produced different early-epoch gradient dynamics over our
        # 6 AttentionPooling instances. See investigation 2026-05-27
        # and [[project_glory_training_pipeline_gaps]].

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, N, E)
        e = torch.tanh(self.att_fc1(x))
        # Promote exp + normalize to fp32 (see ScaledDotProductAttention note).
        alpha = torch.exp(self.att_fc2(e).float())  # (B, N, 1)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2).float()
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        return torch.bmm(x.permute(0, 2, 1), alpha.to(x.dtype)).squeeze(dim=-1)


class NewsEncoder(nn.Module):
    """Local news encoder: word emb → dropout → MHA → LN → drop → pool → LN.

    Consumes a ``(*, T + E + 1 + 1 + 1)`` feature tensor where the columns
    are ``[title tokens (T), entity ids (E), category, subcategory,
    news_index]``.  Only the title tokens are used here — entity /
    category features are consumed downstream.
    """

    def __init__(
        self,
        config: GLORYConfig,
        word_embedding: nn.Embedding,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.news_dim = config.head_num * config.head_dim
        self.title_size = config.title_size
        self.entity_size = config.entity_size

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.msa = MultiHeadAttention(
            config.word_emb_dim,
            config.word_emb_dim,
            config.word_emb_dim,
            config.head_num,
            config.head_dim,
        )
        self.layernorm1 = nn.LayerNorm(self.news_dim)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.attn_pool = AttentionPooling(self.news_dim, config.attention_hidden_dim)
        self.layernorm2 = nn.LayerNorm(self.news_dim)

    def forward(
        self,
        news_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of news.

        Args:
            news_input: ``(B, N, feature_dim)`` int tensor — feature_dim
                = ``title_size + entity_size + 3``.
            mask: optional ``(B*N, title_size)`` token mask.

        Returns:
            ``(B, N, news_dim)`` news representations.
        """
        B, N = news_input.shape[:2]
        # Columns: [title | entity | cat | sub | news_idx]
        title_tokens = news_input[..., : self.title_size]
        flat_title = title_tokens.reshape(B * N, self.title_size).long()

        word_emb = self.dropout1(self.word_embedding(flat_title))  # (B*N, T, E)

        attn_out = self.msa(word_emb, word_emb, word_emb, mask)  # (B*N, T, D)
        attn_out = self.layernorm1(attn_out)
        attn_out = self.dropout2(attn_out)

        pooled = self.attn_pool(attn_out, mask)  # (B*N, D)
        pooled = self.layernorm2(pooled)

        return pooled.view(B, N, self.news_dim)
