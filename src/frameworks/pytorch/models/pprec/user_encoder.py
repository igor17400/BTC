"""PP-Rec user encoder (PyTorch) — Content-Popularity Joint Attention.

Mirrors the Keras reference:

    news_encoder(history) -> (B, H, news_dim)
        -> MHSA over H slots
    popularity_embedding(history_ctr) -> (B, H, pop_emb_dim)
    Concat[user_vecs, pop_emb] -> AttentivePoolingQKY
        -> user vector (B, news_dim)

History validity is determined from the packed news_idx column (column
0 of the per-news features), so the encoder handles both packed
``(B, H, k)`` and plain ``(B, H)`` inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import PPRecConfig

from ...layers import AttentivePoolingQKY
from .news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    """Popularity-aware user encoder with CPJA."""

    def __init__(self, config: PPRecConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder

        self.user_mhsa = nn.MultiheadAttention(
            embed_dim=config.news_dim,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )

        self.popularity_embedding = nn.Embedding(
            config.popularity_embedding_bins, config.popularity_embedding_dim
        )
        cpja_key_dim = config.news_dim + config.popularity_embedding_dim
        self.cpja = AttentivePoolingQKY(
            key_dim=cpja_key_dim, query_vec_dim=config.attention_hidden_dim
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_ctr: torch.Tensor | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Encode user history into a user vector.

        Args:
            history_features: ``(B, H)`` plain news_idx, or ``(B, H, k)``
                packed ``[news_idx | entities | category]``.
            history_ctr: ``(B, H)`` int discretised CTR (optional).
            training: Controls dropout.

        Returns:
            ``(B, news_dim)`` user representation.
        """
        user_vecs = self.news_encoder(
            history_features, training=training
        )  # (B, H, news_dim)
        history_keep = self.news_encoder.valid_mask(history_features)  # (B, H)

        key_padding_mask = ~history_keep
        fully_empty = ~history_keep.any(dim=-1)
        if fully_empty.any():
            kpm = key_padding_mask.clone()
            kpm[fully_empty, 0] = False
        else:
            kpm = key_padding_mask

        user_vecs, _ = self.user_mhsa(
            user_vecs,
            user_vecs,
            user_vecs,
            key_padding_mask=kpm,
            need_weights=False,
        )
        if fully_empty.any():
            user_vecs = user_vecs * (~fully_empty).unsqueeze(-1).unsqueeze(-1).to(
                user_vecs.dtype
            )

        # Popularity embedding (CTR-only per paper footnote 2).
        B, H = history_keep.shape
        if history_ctr is None:
            history_ctr = torch.zeros(B, H, dtype=torch.long, device=user_vecs.device)
        pop_emb = self.popularity_embedding(history_ctr.long())

        key_input = torch.cat([user_vecs, pop_emb], dim=-1)
        return self.cpja(key_input, user_vecs, mask=history_keep)
