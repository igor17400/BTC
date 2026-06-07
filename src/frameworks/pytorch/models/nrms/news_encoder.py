"""NRMS news encoder (PyTorch) — encoder-agnostic, IP2-style.

Receives a configured :class:`TextEncoder` (GloVe or PLM) and runs the
NRMS per-news pipeline at the encoder's **native** text dim. The
additive pool output is projected to ``embedding_size`` (the model's
news_dim) when the dims differ — preserving the full BERT signal
through the per-news attention, as the reference IP2 NRMS+PLM design
does.

Pipeline (single code path for GloVe and PLM):
    text_encoder(news_idx) -> (tokens, mask)        # tokens @ text_dim
        -> Dropout -> MHSA -> Dropout               # MHSA @ text_dim
        -> AdditiveAttention                        # pool @ text_dim
        -> Linear(text_dim -> news_dim)             # Identity if equal
        -> news vector @ news_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import NRMSConfig

from ...layers import AdditiveAttention, TextEncoder


class NewsEncoder(nn.Module):
    """NRMS per-news encoder. Operates at ``text_encoder.output_dim``.

    Args:
        config: NRMS hyperparameters. ``news_num_heads`` MUST divide
            ``text_encoder.output_dim``. ``embedding_size`` is the
            target news_dim (the dim the user encoder operates at).
        text_encoder: Configured :class:`TextEncoder` (GloVe or PLM).
    """

    def __init__(self, config: NRMSConfig, text_encoder: TextEncoder):
        super().__init__()
        text_dim = text_encoder.output_dim
        if text_dim % config.news_num_heads != 0:
            raise ValueError(
                f"news_num_heads={config.news_num_heads} does not divide "
                f"text_encoder.output_dim={text_dim}. Override "
                "spec.model.architecture.news_encoder.num_heads in the "
                f"experiment yaml to a divisor of {text_dim} "
                "(e.g. 12 for BERT-base 768d -> head_dim 64)."
            )
        self.config = config
        self.text_dim = int(text_dim)
        self.news_dim = config.embedding_size
        self.text_encoder = text_encoder

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=config.news_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=text_dim,
            query_vec_dim=config.attention_hidden_dim,
        )
        # Post-pool projection to news_dim. Identity when the text dim
        # already matches the model's news_dim (GloVe at 300, news_dim=300).
        if text_dim != self.news_dim:
            self.projection: nn.Module = nn.Linear(text_dim, self.news_dim)
        else:
            self.projection = nn.Identity()

    @staticmethod
    def valid_mask(news_idx: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed id is non-zero."""
        return news_idx != 0

    def forward(self, news_idx: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode news titles.

        Args:
            news_idx: ``(*leading,)`` int tensor of parsed news ids.
            training: Controls dropout behaviour.

        Returns:
            ``(*leading, news_dim)`` news vectors.
        """
        leading_shape = news_idx.shape
        flat_idx = news_idx.reshape(-1)

        tokens, mask = self.text_encoder(flat_idx)  # (N*, T, text_dim), (N*, T)

        y = self.dropout1(tokens)

        key_padding_mask = mask == 0  # True = padded
        fully_padded = key_padding_mask.all(dim=-1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        y, _ = self.multi_head_attention(y, y, y, key_padding_mask=key_padding_mask)
        y = self.dropout2(y)

        valid_mask = mask != 0
        pooled = self.additive_attention(y, mask=valid_mask)  # (N*, text_dim)

        if fully_padded.any():
            pooled = pooled.masked_fill(fully_padded.unsqueeze(-1), 0.0)

        news_repr = self.projection(pooled)  # (N*, news_dim)
        return news_repr.reshape(*leading_shape, self.news_dim)
