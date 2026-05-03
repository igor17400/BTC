"""TCCM-faithful PyTorch layers.

Mirrors the official TCCM reference (``reference_codes/TCCM/model.py``,
``reference_codes/TCCM/AttentionModel.py``) bit-for-bit:

* :class:`TCCMNewsEncoder` — paper ``get_news_encoder_co1`` recipe with
  20×20 self-attn and 20×20 cross-attn, **Add** residual fusion, no
  input dropout, ``AttLayer(400)`` pooling.
* :class:`TCCMPopularityEncoder` — paper ``get_popularity_encoder``
  recipe: shared word/entity popularity-bucket embedding (200, 200),
  ``Self_Attention(400, 1)`` = 400 heads × 1 dim, Add residual, three
  Dense layers + sigmoid → ``s_p``, time embedding (505, 100) +
  Dense(64) × 2 + sigmoid → ``t'``, output ``s_p · t'^(-λ)``.
* :class:`TCCMActivityGater` — paper recipe: Dense(128, tanh) →
  Dense(64) → Dense(1, sigmoid).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import TCCMConfig

from ...layers import AdditiveAttention as _AdditiveAttention


# ---------------------------------------------------------------------------
# News encoder (paper ``co1``)
# ---------------------------------------------------------------------------


class TCCMNewsEncoder(nn.Module):
    """Knowledge-aware news encoder for TCCM (PyTorch port)."""

    def __init__(
        self,
        config: TCCMConfig,
        word_embedding_layer: nn.Embedding,
        entity_embedding_layer: nn.Embedding,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding_layer
        self.entity_embedding = entity_embedding_layer

        proj_dim = config.num_heads * config.head_dim  # 400 by default

        # Internal projections so each MHA can stay uniform 400-d. The
        # word embedding is 300-d and the entity embedding is 100-d, so
        # we project both up to ``proj_dim`` (400) before any attention
        # — same effective architecture as the Keras MHA's internal
        # input projections.
        self.title_in_proj = nn.Linear(config.embedding_size, proj_dim)
        self.entity_in_proj = nn.Linear(config.entity_embedding_dim, proj_dim)

        def _mha() -> nn.MultiheadAttention:
            return nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=config.num_heads,
                dropout=config.dropout_rate,
                batch_first=True,
            )

        self.title_mhsa = _mha()
        self.entity_mhsa = _mha()
        self.title_mhca = _mha()
        self.entity_mhca = _mha()

        self.title_dropout = nn.Dropout(config.dropout_rate)
        self.entity_dropout = nn.Dropout(config.dropout_rate)
        self.fusion_dropout = nn.Dropout(config.dropout_rate)

        self.title_attention = _AdditiveAttention(
            input_dim=proj_dim, query_vec_dim=config.attention_hidden_dim
        )
        self.entity_attention = _AdditiveAttention(
            input_dim=proj_dim, query_vec_dim=config.attention_hidden_dim
        )
        self.fusion_attention = _AdditiveAttention(
            input_dim=proj_dim, query_vec_dim=config.attention_hidden_dim
        )

    @staticmethod
    def _safe_mha(
        mha: nn.MultiheadAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run MHA with fully-padded-row safety (same trick as PP-Rec)."""
        fully_padded = key_padding_mask.all(dim=-1)
        if fully_padded.any():
            kpm = key_padding_mask.clone()
            kpm[fully_padded, 0] = False
        else:
            kpm = key_padding_mask
        out, _ = mha(q, k, v, key_padding_mask=kpm, need_weights=False)
        if fully_padded.any():
            mask_q = (~fully_padded).unsqueeze(-1).unsqueeze(-1).to(out.dtype)
            out = out * mask_q
        return out

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode news features into a news vector.

        Args:
            inputs: ``(batch, T+E)`` int64 — concatenated title tokens and
                entity indices.

        Returns:
            ``(batch, news_dim)`` news vector.
        """
        title_len = self.config.max_title_length
        title_tokens = inputs[:, :title_len].long()
        entity_indices = inputs[:, title_len:].long()
        title_pad = title_tokens == 0
        entity_pad = entity_indices == 0
        title_keep = ~title_pad
        entity_keep = ~entity_pad

        title_emb = self.word_embedding(title_tokens)
        entity_emb = self.entity_embedding(entity_indices)
        title_h = self.title_in_proj(title_emb)
        entity_h = self.entity_in_proj(entity_emb)

        title_co = self._safe_mha(
            self.title_mhca, title_h, entity_h, entity_h, entity_pad
        )
        entity_co = self._safe_mha(
            self.entity_mhca, entity_h, title_h, title_h, title_pad
        )

        title_self = self._safe_mha(
            self.title_mhsa, title_h, title_h, title_h, title_pad
        )
        title_seq = title_self + title_co
        title_seq = self.title_dropout(title_seq)
        title_vec = self.title_attention(title_seq, mask=title_keep)

        entity_self = self._safe_mha(
            self.entity_mhsa, entity_h, entity_h, entity_h, entity_pad
        )
        entity_seq = entity_self + entity_co
        entity_seq = self.entity_dropout(entity_seq)
        entity_vec = self.entity_attention(entity_seq, mask=entity_keep)

        stacked = torch.stack([title_vec, entity_vec], dim=1)
        stacked = self.fusion_dropout(stacked)
        return self.fusion_attention(stacked)


# ---------------------------------------------------------------------------
# Popularity encoder (paper ``get_popularity_encoder``)
# ---------------------------------------------------------------------------


class TCCMPopularityEncoder(nn.Module):
    """Content-aware popularity encoder for TCCM (PyTorch port)."""

    def __init__(self, config: TCCMConfig):
        super().__init__()
        self.config = config

        self.token_pop_embedding = nn.Embedding(
            config.pop_token_embedding_bins,
            config.pop_token_embedding_dim,
        )
        self.title_dropout1 = nn.Dropout(config.dropout_rate)
        self.entity_dropout1 = nn.Dropout(config.dropout_rate)

        # Project bucket embeddings (dim 200) to ``proj_dim`` (400) so the
        # MHA can operate uniformly with embed_dim=400. With
        # num_heads=400, head_dim=embed_dim/num_heads=1 — matches the
        # paper's ``Self_Attention(400, 1)``.
        proj_dim = config.pop_num_heads * config.pop_head_dim
        self.title_proj = nn.Linear(config.pop_token_embedding_dim, proj_dim)
        self.entity_proj = nn.Linear(config.pop_token_embedding_dim, proj_dim)

        def _mha() -> nn.MultiheadAttention:
            return nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=config.pop_num_heads,
                dropout=config.dropout_rate,
                batch_first=True,
            )

        self.title_co_mhca = _mha()
        self.entity_co_mhca = _mha()
        self.title_mhsa = _mha()
        self.entity_mhsa = _mha()

        self.title_dropout2 = nn.Dropout(config.dropout_rate)
        self.entity_dropout2 = nn.Dropout(config.dropout_rate)
        self.fusion_dropout = nn.Dropout(config.dropout_rate)

        # AttLayer(400, seed) — hidden dim equals proj_dim.
        self.title_attention = _AdditiveAttention(
            input_dim=proj_dim, query_vec_dim=proj_dim
        )
        self.entity_attention = _AdditiveAttention(
            input_dim=proj_dim, query_vec_dim=proj_dim
        )
        self.fusion_attention = _AdditiveAttention(
            input_dim=proj_dim, query_vec_dim=proj_dim
        )

        c1, c2, c3 = config.pop_content_dims
        self.content_d1 = nn.Linear(proj_dim, c1)
        self.content_d2 = nn.Linear(c1, c2)
        self.content_d3 = nn.Linear(c2, c3)
        self.content_out = nn.Linear(c3, 1)

        self.time_embedding = nn.Embedding(
            config.timeliness_embedding_bins, config.timeliness_embedding_dim
        )
        r1, r2 = config.pop_recency_dims
        self.time_d1 = nn.Linear(config.timeliness_embedding_dim, r1)
        self.time_d2 = nn.Linear(r1, r2)
        self.time_out = nn.Linear(r2, 1)
        self.timeliness_lambda = float(config.timeliness_lambda)

    def forward(
        self,
        bucket_input: torch.Tensor,
        time_input: torch.Tensor,
        title_len: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            bucket_input: ``(B, T+E)`` long.
            time_input: ``(B,)`` long.
            title_len: ``T`` so the layer can split the entity slice.

        Returns:
            ``(B,)`` popularity score.
        """
        title_buckets = bucket_input[:, :title_len].long()
        entity_buckets = bucket_input[:, title_len:].long()

        title_emb = self.title_dropout1(self.token_pop_embedding(title_buckets))
        entity_emb = self.entity_dropout1(self.token_pop_embedding(entity_buckets))
        title_h = self.title_proj(title_emb)
        entity_h = self.entity_proj(entity_emb)

        title_co, _ = self.title_co_mhca(title_h, entity_h, entity_h, need_weights=False)
        entity_co, _ = self.entity_co_mhca(entity_h, title_h, title_h, need_weights=False)

        title_self, _ = self.title_mhsa(title_h, title_h, title_h, need_weights=False)
        title_seq = title_self + title_co
        title_seq = self.title_dropout2(title_seq)
        title_vec = self.title_attention(title_seq)

        entity_self, _ = self.entity_mhsa(
            entity_h, entity_h, entity_h, need_weights=False
        )
        entity_seq = entity_self + entity_co
        entity_seq = self.entity_dropout2(entity_seq)
        entity_vec = self.entity_attention(entity_seq)

        stacked = torch.stack([title_vec, entity_vec], dim=1)
        stacked = self.fusion_dropout(stacked)
        pop_vec = self.fusion_attention(stacked)

        x = torch.tanh(self.content_d1(pop_vec))
        x = self.content_d2(x)
        x = self.content_d3(x)
        s_p = torch.sigmoid(self.content_out(x)).squeeze(-1)

        time_emb = self.time_embedding(time_input.long())
        r = torch.tanh(self.time_d1(time_emb))
        r = self.time_d2(r)
        t_prime = torch.sigmoid(self.time_out(r)).squeeze(-1)
        t_factor = torch.clamp(t_prime, min=1e-6).pow(-self.timeliness_lambda)
        return s_p * t_factor


# ---------------------------------------------------------------------------
# Activity gater (paper recipe: Dense(128, tanh) → Dense(64) → Dense(1, sigmoid))
# ---------------------------------------------------------------------------


class TCCMActivityGater(nn.Module):
    """Per-user gate balancing relevance vs. popularity (TCCM-faithful)."""

    def __init__(self, config: TCCMConfig):
        super().__init__()
        d1, d2 = config.activity_gate_dims
        self.dense1 = nn.Linear(config.news_dim, d1)
        self.dense2 = nn.Linear(d1, d2)
        self.dense_out = nn.Linear(d2, 1)

    def forward(self, user_vec: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense1(user_vec))
        x = self.dense2(x)
        return torch.sigmoid(self.dense_out(x)).squeeze(-1)
