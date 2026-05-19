"""LSTUR user encoder (PyTorch) — GRU + long-term user embedding.

Pipeline:
    news_encoder(history) -> (B, H, news_dim)
    user_embedding(user_idx) -> (B, gru_unit)        # long-term repr
        -> Bernoulli mask during training (paper §3.2)
    "ini": GRU(news_dim -> gru_unit) with long-term repr as h0
           -> last hidden state                       # short-term repr
    "con": GRU(news_dim -> gru_unit) with zero h0
           -> last hidden state
           -> Linear(2*gru_unit -> gru_unit) over concat(short, long)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import LSTURConfig

from .news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    """LSTUR user encoder.

    Args:
        config: LSTUR hyperparameters. ``config.type`` selects the
            ``"ini"`` or ``"con"`` GRU initialisation variant.
        news_encoder: Shared :class:`NewsEncoder` — used to encode each
            history slot. Its ``output_dim`` sets the GRU input size.
        num_users: Long-term user-embedding vocabulary size.
    """

    def __init__(
        self,
        config: LSTURConfig,
        news_encoder: NewsEncoder,
        num_users: int,
    ):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.num_users = num_users

        gru_input_size = news_encoder.output_dim

        self.user_embedding = nn.Embedding(num_users, config.gru_unit, padding_idx=None)
        nn.init.zeros_(self.user_embedding.weight)

        # Bernoulli mask on long-term user embeddings during training (paper §3.2).
        self.user_embedding_dropout = nn.Dropout(p=config.user_embedding_dropout_rate)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.gru_unit,
            batch_first=True,
        )

        if config.type == "con":
            self.concat_dense: nn.Module = nn.Linear(
                config.gru_unit * 2, config.gru_unit
            )

    def forward(
        self,
        inputs: torch.Tensor | list[torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Encode a user from browsing history + user id.

        Args:
            inputs: ``[history_features, user_ids]``.
                - ``history_features``: ``(B, H)`` plain news_idx, or
                  ``(B, H, k)`` packed
                  ``[news_idx | category | subcategory]``.
                - ``user_ids``: ``(B,)`` or ``(B, 1)`` int.
            training: Controls dropout.

        Returns:
            ``(B, gru_unit)`` user representations.
        """
        history_features, user_indices = inputs

        if user_indices.dim() == 1:
            user_indices = user_indices.unsqueeze(-1)
        long_u_emb = self.user_embedding(user_indices).squeeze(1)  # (B, gru_unit)
        long_u_emb = self.user_embedding_dropout(long_u_emb)

        history_repr = self.news_encoder(
            history_features, training=training
        )  # (B, H, news_dim)

        if self.config.type == "ini":
            h0 = long_u_emb.unsqueeze(0)  # (1, B, gru_unit)
            output, _ = self.gru(history_repr, h0)
            user_present = output[:, -1, :]
        elif self.config.type == "con":
            output, _ = self.gru(history_repr)
            short_uemb = output[:, -1, :]
            concat_emb = torch.cat([short_uemb, long_u_emb], dim=-1)
            user_present = self.concat_dense(concat_emb)
        else:
            raise ValueError(f"Invalid user encoder type: {self.config.type}")

        return user_present
