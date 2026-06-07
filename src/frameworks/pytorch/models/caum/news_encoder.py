"""CAUM news encoder (PyTorch) — encoder-agnostic.

Three parallel branches feed a final fusion Linear:

  - Title branch:    text_encoder(news_idx) -> (tokens, mask)
                         -> Dropout -> MHSA -> Dropout
                         -> AdditiveAttention with mask
  - Entity branch:   Embedding -> Dropout -> MHSA -> Dropout
                         -> AdditiveAttention with mask
  - Category branch: Embedding -> Dropout -> Linear -> tanh? (no activation)

The title pipeline runs at the encoder's native text dim
(``text_encoder.output_dim`` — 300 for GloVe, 768 for BERT-base) and is
projected to ``news_dim`` by the fusion Linear together with the optional
entity / category vectors. This preserves the full BERT signal through
the per-news MHSA, matching the IP2-style design used elsewhere.

Input layout (packed, last dim):
    [news_idx | entity_id_1 .. entity_id_E | category_id]
    Trailing columns drop when their flag is False.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import CAUMConfig

from ...layers import AdditiveAttention, TextEncoder


class _EntityBranch(nn.Module):
    """Entity pipeline: Embedding -> Dropout -> MHSA -> Dropout -> AdditivePool."""

    def __init__(self, config: CAUMConfig, num_entities: int, *, frozen: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, config.entity_embedding_dim)
        if frozen:
            self.embedding.weight.requires_grad = False

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=config.entity_embedding_dim,
            num_heads=config.entity_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.additive_attention = AdditiveAttention(
            input_dim=config.entity_embedding_dim,
            query_vec_dim=config.news_attention_hidden_dim,
        )

    def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
        """``entity_indices``: ``(N*, E)`` ints → ``(N*, entity_embedding_dim)``."""
        entity_mask = entity_indices != 0  # (N*, E)
        emb = self.dropout1(self.embedding(entity_indices))  # (N*, E, D)

        key_padding_mask = ~entity_mask
        fully_padded = key_padding_mask.all(dim=-1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        y, _ = self.mhsa(emb, emb, emb, key_padding_mask=key_padding_mask)
        y = self.dropout2(y)

        pooled = self.additive_attention(y, mask=entity_mask)
        if fully_padded.any():
            pooled = pooled.masked_fill(fully_padded.unsqueeze(-1), 0.0)
        return pooled


class _CategoryBranch(nn.Module):
    """Category pipeline: Embedding -> Dropout -> Linear."""

    def __init__(self, config: CAUMConfig, num_categories: int, *, frozen: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, config.category_embedding_dim)
        if frozen:
            self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(
            config.category_embedding_dim, config.category_embedding_dim
        )

    def forward(self, category_id: torch.Tensor) -> torch.Tensor:
        """``category_id``: ``(N*,)`` ints → ``(N*, category_embedding_dim)``."""
        emb = self.dropout(self.embedding(category_id))
        return self.dense(emb)


class NewsEncoder(nn.Module):
    """CAUM news encoder. Outputs ``(*leading, news_dim)``.

    Args:
        config: CAUM hyperparameters. ``news_num_heads`` MUST divide
            ``text_encoder.output_dim``.
        text_encoder: :class:`TextEncoder` for the title cache (GloVe or PLM).
        num_entities: Entity vocab size (excludes padding index 0).
            Ignored when ``config.use_entity`` is False.
        num_categories: Category vocab size (excludes padding index 0).
            Ignored when ``config.use_category`` is False.
    """

    def __init__(
        self,
        config: CAUMConfig,
        text_encoder: TextEncoder,
        num_entities: int = 0,
        num_categories: int = 0,
    ):
        super().__init__()
        text_dim = text_encoder.output_dim
        if text_dim % config.news_num_heads != 0:
            raise ValueError(
                f"CAUM title MHSA: text_encoder.output_dim={text_dim} is not "
                f"divisible by news_num_heads={config.news_num_heads}. Override "
                "spec.model.architecture.news_encoder.num_heads in the experiment "
                f"yaml to a divisor of {text_dim} (e.g. 12 for BERT-base 768d → "
                "head_dim 64)."
            )

        self.config = config
        self.text_encoder = text_encoder
        self.text_dim = int(text_dim)

        # Title branch — MHSA + AddAttn at text_dim.
        self.title_dropout = nn.Dropout(config.dropout_rate)
        self.title_mhsa = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=config.news_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.title_dropout2 = nn.Dropout(config.dropout_rate)
        self.title_attention = AdditiveAttention(
            input_dim=text_dim,
            query_vec_dim=config.news_attention_hidden_dim,
        )

        self.entity_branch: nn.Module | None = None
        if config.use_entity:
            self.entity_branch = _EntityBranch(config, num_entities)
        self.category_branch: nn.Module | None = None
        if config.use_category:
            self.category_branch = _CategoryBranch(config, num_categories)

        # Fusion → news_dim.
        fusion_in = text_dim
        if self.entity_branch is not None:
            fusion_in += config.entity_embedding_dim
        if self.category_branch is not None:
            fusion_in += config.category_embedding_dim
        self.fusion_dense = nn.Linear(fusion_in, config.news_dim)

        self._n_entity = config.max_entities if config.use_entity else 0
        self._n_cols = 1 + self._n_entity + int(config.use_category)

    @staticmethod
    def valid_mask(packed_features: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero."""
        if packed_features.dim() >= 2 and packed_features.shape[-1] > 1:
            return packed_features[..., 0] != 0
        return packed_features != 0

    def _split_columns(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return ``(news_idx, entity_indices, category_id)`` from inputs."""
        if self._n_cols == 1:
            news_idx = (
                inputs if inputs.shape[-1] != 1 or inputs.dim() == 1 else inputs[..., 0]
            )
            return news_idx, None, None

        news_idx = inputs[..., 0]
        col = 1
        entity_indices: torch.Tensor | None = None
        category_id: torch.Tensor | None = None
        if self.entity_branch is not None:
            entity_indices = inputs[..., col : col + self._n_entity]
            col += self._n_entity
        if self.category_branch is not None:
            category_id = inputs[..., col]
        return news_idx, entity_indices, category_id

    def _encode_title(self, news_idx: torch.Tensor) -> torch.Tensor:
        """Encode title tokens through TextEncoder + MHSA + additive pool.

        Args:
            news_idx: ``(N*,)`` int tensor of parsed news ids.

        Returns:
            ``(N*, text_dim)`` title vectors.
        """
        tokens, mask = self.text_encoder(news_idx)  # (N*, T, text_dim), (N*, T)
        y = self.title_dropout(tokens)

        valid_mask = mask != 0  # (N*, T) bool
        key_padding_mask = ~valid_mask
        fully_padded = key_padding_mask.all(dim=-1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        y, _ = self.title_mhsa(y, y, y, key_padding_mask=key_padding_mask)
        y = self.title_dropout2(y)

        pooled = self.title_attention(y, mask=valid_mask)  # (N*, text_dim)
        if fully_padded.any():
            pooled = pooled.masked_fill(fully_padded.unsqueeze(-1), 0.0)
        return pooled

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode a batch of news articles.

        Args:
            inputs: Packed ``(..., k)`` int with column order
                ``[news_idx | entities | category]`` (trailing columns
                drop when their flag is False), or plain ``(...,)`` int
                when no structured views are enabled.
            training: Controls dropout.

        Returns:
            ``(..., news_dim)`` news vectors.
        """
        news_idx, entity_indices, category_id = self._split_columns(inputs)
        leading_shape = news_idx.shape

        flat_news_idx = news_idx.reshape(-1)
        title_vec = self._encode_title(flat_news_idx)  # (N*, text_dim)

        parts: list[torch.Tensor] = [title_vec]
        if self.entity_branch is not None:
            assert entity_indices is not None
            flat_entities = entity_indices.reshape(-1, self._n_entity)
            parts.append(self.entity_branch(flat_entities))
        if self.category_branch is not None:
            assert category_id is not None
            parts.append(self.category_branch(category_id.reshape(-1)))

        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else title_vec
        news_vec = self.fusion_dense(fused)  # (N*, news_dim)
        return news_vec.reshape(*leading_shape, self.config.news_dim)
