"""PP-Rec news encoder (PyTorch) — encoder-agnostic ``co1`` recipe.

Knowledge-aware news encoder for PP-Rec / TCCM. Mirrors the Keras
reference: title MHSA + entity MHSA + bidirectional cross-attention
(MHCA) between title words and entities, then concat + Dense fusion to
``news_dim``.

Pipeline (single code path for GloVe and PLM):
    text_encoder(news_idx) -> (tokens, mask)                  # tokens @ text_dim
        -> Linear(text_dim -> embedding_size)?                # Identity if equal
        -> Dropout
                      ┌── MHSA(num_heads × head_dim) ─────┐
                      ├── MHCA(co1) with entity tokens ───┤
                      ├── concat -> Linear -> Dropout
                      └── AdditiveAttention with mask → title_vec (news_dim)
    entity_emb(entities)
        -> MHSA on entities
        -> concat with co-attention(entity ← title)
        -> Linear -> Dropout -> AdditiveAttention → entity_vec (news_dim)
    category_emb(category)                                    # (optional)
        -> Dropout                                            # @ category_emb_dim
    concat([title, entity, category]) -> Linear → news vector (news_dim)

Input layout (packed, last dim):
    [news_idx | entity_id_1 .. entity_id_E | category_id]
    Trailing columns drop when the corresponding branch is disabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import PPRecConfig

from ...layers import (
    AdditiveAttention,
    CrossAttention,
    MultiHeadSelfAttention,
    TextEncoder,
)


class NewsEncoder(nn.Module):
    """PP-Rec news encoder (paper ``co1`` variant).

    Args:
        config: PP-Rec hyperparameters.
        text_encoder: :class:`TextEncoder` for the title cache (GloVe or PLM).
            When ``text_encoder.output_dim`` differs from
            ``config.embedding_size``, a Linear projects down to
            ``embedding_size`` before MHSA — keeps every downstream MHA /
            cross-attn dim equal to the paper's ``embedding_size`` so
            ``num_heads`` doesn't need to be re-tuned per encoder.
        entity_embedding_layer: ``nn.Embedding`` for entity ids. ``None``
            when ``config.use_entity`` is False (skips both the entity
            branch and the bidirectional MHCA).
        category_embedding_layer: ``nn.Embedding`` for category ids.
            ``None`` to skip the category branch.
    """

    def __init__(
        self,
        config: PPRecConfig,
        text_encoder: TextEncoder,
        entity_embedding_layer: nn.Embedding | None = None,
        category_embedding_layer: nn.Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder
        self.entity_embedding = entity_embedding_layer
        self.category_embedding = category_embedding_layer

        text_dim = text_encoder.output_dim
        emb_dim = config.embedding_size
        self.text_dim = int(text_dim)
        self.text_projection: nn.Module = (
            nn.Linear(text_dim, emb_dim) if text_dim != emb_dim else nn.Identity()
        )

        co_out_dim = config.co_num_heads * config.co_head_dim

        # --- Word (title) branch — MHSA at embedding_size ---
        self.word_dropout = nn.Dropout(config.dropout_rate)
        self.word_mhsa = MultiHeadSelfAttention(
            in_features=emb_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dropout_rate=config.dropout_rate,
        )
        # Title MHSA emits (B, T, emb_dim). After concat with title_co
        # (B, T, co_out_dim) project to news_dim.
        word_concat_dim = emb_dim + (
            co_out_dim if entity_embedding_layer is not None else 0
        )
        self.word_proj = nn.Linear(word_concat_dim, config.news_dim)
        self.word_dropout2 = nn.Dropout(config.dropout_rate)
        self.word_attention = AdditiveAttention(
            input_dim=config.news_dim,
            query_vec_dim=config.attention_hidden_dim,
        )

        # --- Entity branch + bidirectional MHCA ---
        if entity_embedding_layer is not None:
            self.entity_mhsa = MultiHeadSelfAttention(
                in_features=config.entity_embedding_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                dropout_rate=config.dropout_rate,
            )
            entity_concat_dim = config.entity_embedding_dim + co_out_dim
            self.entity_proj = nn.Linear(entity_concat_dim, config.news_dim)
            self.entity_dropout = nn.Dropout(config.dropout_rate)
            self.entity_attention = AdditiveAttention(
                input_dim=config.news_dim,
                query_vec_dim=config.attention_hidden_dim,
            )
            # Bidirectional cross-attention (paper ``co1``). CrossAttention
            # folds the query projection into W_q natively, so different
            # Q/KV dims are handled without a separate pre-projection.
            self.title_mhca = CrossAttention(
                q_dim=emb_dim,
                kv_dim=config.entity_embedding_dim,
                num_heads=config.co_num_heads,
                head_dim=config.co_head_dim,
                dropout_rate=config.dropout_rate,
            )
            self.entity_mhca = CrossAttention(
                q_dim=config.entity_embedding_dim,
                kv_dim=emb_dim,
                num_heads=config.co_num_heads,
                head_dim=config.co_head_dim,
                dropout_rate=config.dropout_rate,
            )

        # --- Category branch ---
        if category_embedding_layer is not None:
            self.category_dropout = nn.Dropout(config.dropout_rate)

        # --- Fusion: [title, entity, category] → news_dim ---
        fusion_dim = config.news_dim
        if entity_embedding_layer is not None:
            fusion_dim += config.news_dim
        if category_embedding_layer is not None:
            fusion_dim += config.category_embedding_dim
        self.fusion_dense = nn.Linear(fusion_dim, config.news_dim)

        self._n_entity = (
            config.max_entities if entity_embedding_layer is not None else 0
        )
        self._n_cols = 1 + self._n_entity + int(category_embedding_layer is not None)

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
        if self.entity_embedding is not None:
            entity_indices = inputs[..., col : col + self._n_entity]
            col += self._n_entity
        if self.category_embedding is not None:
            category_id = inputs[..., col]
        return news_idx, entity_indices, category_id

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode a batch of news articles.

        Args:
            inputs: Packed ``(..., k)`` int with column order
                ``[news_idx | entities | category]`` (trailing columns
                drop when their branch is disabled), or plain ``(...,)``
                int when only the title branch is enabled.
            training: Controls dropout.

        Returns:
            ``(..., news_dim)`` news vectors.
        """
        news_idx, entity_indices, category_id = self._split_columns(inputs)
        leading_shape = news_idx.shape

        flat_news_idx = news_idx.reshape(-1)
        # --- Title path ---
        tokens, title_mask_raw = self.text_encoder(
            flat_news_idx
        )  # (N*, T, text_dim), (N*, T)
        title_keep = title_mask_raw != 0
        title_pad = ~title_keep
        title_emb = self.text_projection(tokens)
        title_emb = self.word_dropout(title_emb)  # (N*, T, emb_dim)

        has_entity = self.entity_embedding is not None
        has_category = self.category_embedding is not None

        if has_entity:
            assert entity_indices is not None
            flat_entities = entity_indices.reshape(-1, self._n_entity)
            entity_pad = flat_entities == 0
            entity_keep = ~entity_pad
            entity_emb = self.entity_embedding(flat_entities)  # (N*, E, ent_dim)

            # Bidirectional MHCA on RAW embeddings.
            title_co = self.title_mhca(
                title_emb, entity_emb, key_padding_mask=entity_pad
            )  # (N*, T, co_out_dim)
            entity_co = self.entity_mhca(
                entity_emb, title_emb, key_padding_mask=title_pad
            )  # (N*, E, co_out_dim)

        # --- Title self-attention + concat ---
        title_vecs = self.word_mhsa(title_emb, key_padding_mask=title_pad)
        if has_entity:
            title_vecs = torch.cat([title_vecs, title_co], dim=-1)
        title_vecs = self.word_proj(title_vecs)
        title_vecs = self.word_dropout2(title_vecs)
        title_vec = self.word_attention(title_vecs, mask=title_keep)

        vecs: list[torch.Tensor] = [title_vec]

        # --- Entity self-attention + concat ---
        if has_entity:
            entity_vecs = self.entity_mhsa(entity_emb, key_padding_mask=entity_pad)
            entity_vecs = torch.cat([entity_vecs, entity_co], dim=-1)
            entity_vecs = self.entity_proj(entity_vecs)
            entity_vecs = self.entity_dropout(entity_vecs)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_keep)
            vecs.append(entity_vec)

        # --- Category branch ---
        if has_category:
            assert category_id is not None
            cat_emb = self.category_embedding(category_id.reshape(-1))  # (N*, cat_dim)
            cat_emb = self.category_dropout(cat_emb)
            vecs.append(cat_emb)

        # --- Fusion ---
        fused = torch.cat(vecs, dim=-1) if len(vecs) > 1 else title_vec
        news_vec = self.fusion_dense(fused)  # (N*, news_dim)
        return news_vec.reshape(*leading_shape, self.config.news_dim)
