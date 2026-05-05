"""CAUM (Candidate-Aware User Modeling) -- PyTorch.

Reference: Qi et al., "News Recommendation with Candidate-aware User
Modeling", SIGIR 2022.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.core.models.configs import CAUMConfig

from ..layers import AdditiveAttention
from .base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(nn.Module):
    """Encode news from title, entities, and category.

    Following the reference code:
        title:    Embedding → Dropout → MHSA → Dropout → AdditivePool
        entity:   Embedding → Dropout → MHSA → Dropout → AdditivePool
        category: Embedding → Dropout → Dense
        fusion:   Concat → Dense(news_dim)
    """

    def __init__(
        self,
        config: CAUMConfig,
        embedding_layer: nn.Embedding,
        entity_embedding: nn.Embedding | None = None,
        category_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.entity_embedding = entity_embedding
        self.category_embedding = category_embedding

        # Title branch
        self.title_dropout = nn.Dropout(config.dropout_rate)
        self.title_mhsa = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.news_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.title_dropout2 = nn.Dropout(config.dropout_rate)
        self.title_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.news_attention_hidden_dim,
        )

        # Entity branch
        if entity_embedding is not None:
            self.entity_dropout = nn.Dropout(config.dropout_rate)
            self.entity_mhsa = nn.MultiheadAttention(
                embed_dim=config.entity_embedding_dim,
                num_heads=config.entity_num_heads,
                dropout=config.dropout_rate,
                batch_first=True,
            )
            self.entity_dropout2 = nn.Dropout(config.dropout_rate)
            self.entity_attention = AdditiveAttention(
                input_dim=config.entity_embedding_dim,
                query_vec_dim=config.news_attention_hidden_dim,
            )

        # Category branch
        if category_embedding is not None:
            self.category_dropout = nn.Dropout(config.dropout_rate)
            self.category_dense = nn.Linear(
                config.category_embedding_dim, config.category_embedding_dim
            )

        # Fusion
        fusion_in = config.embedding_size
        if entity_embedding is not None:
            fusion_in += config.entity_embedding_dim
        if category_embedding is not None:
            fusion_in += config.category_embedding_dim
        self.fusion_dense = nn.Linear(fusion_in, config.news_dim)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """inputs: (batch, feature_dim) → (batch, news_dim)."""
        offset = 0
        title_len = self.config.max_title_length

        # Title
        title_tokens = inputs[:, offset : offset + title_len]
        offset += title_len
        title_mask = title_tokens != 0

        title_emb = self.title_dropout(self.embedding_layer(title_tokens))

        key_padding_mask = ~title_mask
        fully_padded = key_padding_mask.all(dim=-1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False

        title_vecs, _ = self.title_mhsa(
            title_emb, title_emb, title_emb, key_padding_mask=key_padding_mask
        )
        title_vecs = self.title_dropout2(title_vecs)
        title_vec = self.title_attention(title_vecs, mask=title_mask)
        if fully_padded.any():
            title_vec = title_vec.masked_fill(fully_padded.unsqueeze(-1), 0.0)

        vecs = [title_vec]

        # Entity
        has_entity = self.entity_embedding is not None and self.config.use_entity
        if has_entity:
            entity_len = self.config.max_entities
            entity_indices = inputs[:, offset : offset + entity_len]
            offset += entity_len
            entity_mask = entity_indices != 0

            entity_emb = self.entity_dropout(self.entity_embedding(entity_indices))

            ent_kpm = ~entity_mask
            ent_fully_padded = ent_kpm.all(dim=-1)
            if ent_fully_padded.any():
                ent_kpm = ent_kpm.clone()
                ent_kpm[ent_fully_padded, 0] = False

            entity_vecs, _ = self.entity_mhsa(
                entity_emb, entity_emb, entity_emb, key_padding_mask=ent_kpm
            )
            entity_vecs = self.entity_dropout2(entity_vecs)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_mask)
            if ent_fully_padded.any():
                entity_vec = entity_vec.masked_fill(ent_fully_padded.unsqueeze(-1), 0.0)
            vecs.append(entity_vec)

        # Category
        has_category = self.category_embedding is not None and self.config.use_category
        if has_category:
            category_idx = inputs[:, offset : offset + 1]
            offset += 1

            cat_emb = self.category_embedding(category_idx).squeeze(1)
            cat_emb = self.category_dropout(cat_emb)
            cat_vec = self.category_dense(cat_emb)
            vecs.append(cat_vec)

        # Fusion
        if len(vecs) > 1:
            fused = torch.cat(vecs, dim=-1)
        else:
            fused = vecs[0]
        return self.fusion_dense(fused)


# ---------------------------------------------------------------------------
# Candidate-aware user interaction model
# ---------------------------------------------------------------------------


class DenseAttentionScorer(nn.Module):
    """Two-layer MLP scorer for Candi-Att."""

    def __init__(self, hidden_dim: int, mid_dim: int):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, mid_dim)
        self.dense3 = nn.Linear(mid_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dense3(torch.tanh(self.dense2(torch.tanh(self.dense1(inputs)))))


class InterModel(nn.Module):
    """Candidate-aware user interest model.

    Architecture:
        1. Candi-CNN: circular-shift window + candidate → Dense
        2. Candi-SelfAtt: concat candidate → Dense → MHSA
        3. Fusion: concat [cnn, selfatt] → Dropout → Dense
        4. Candi-Att: DNN scorer → softmax → weighted sum
        5. Score: dot(user_vec, candidate)
    """

    def __init__(self, config: CAUMConfig):
        super().__init__()
        D = config.news_dim
        self.config = config

        self.dropout_cand = nn.Dropout(config.dropout_rate)
        self.dropout_clicks = nn.Dropout(config.dropout_rate)

        self.cnn_projection = nn.Linear(4 * D, D)
        self.selfatt_input_projection = nn.Linear(2 * D, D)
        self.selfatt_mha = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=config.candi_selfatt_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )

        self.fusion_dropout = nn.Dropout(config.dropout_rate)
        self.fusion_projection = nn.Linear(2 * D, D)

        self.dense_att = DenseAttentionScorer(
            hidden_dim=config.candi_att_hidden_dim,
            mid_dim=config.candi_att_mid_dim,
        )

    def forward(
        self,
        cand_vec: torch.Tensor,
        clicked_vecs: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """cand_vec: (B,D), clicked_vecs: (B,H,D) → (B,) scores."""
        D = self.config.news_dim
        B, H, _ = clicked_vecs.shape

        can_vec_dropped = self.dropout_cand(cand_vec)
        user_vecs = self.dropout_clicks(clicked_vecs)

        cand_repeated = cand_vec.unsqueeze(1).expand(-1, H, -1)

        # Candi-CNN
        left = torch.cat([user_vecs[:, -1:, :], user_vecs[:, :-1, :]], dim=1)
        right = torch.cat([user_vecs[:, 1:, :], user_vecs[:, :1, :]], dim=1)
        cnn_input = torch.cat([left, user_vecs, right, cand_repeated], dim=-1)
        cnn_out = self.cnn_projection(cnn_input)

        # Candi-SelfAtt
        selfatt_input = torch.cat([cand_repeated, user_vecs], dim=-1)
        selfatt_input = self.selfatt_input_projection(selfatt_input)

        history_mask = clicked_vecs.any(dim=-1)  # (B, H)
        key_padding_mask = ~history_mask

        fully_empty = key_padding_mask.all(dim=-1)
        if fully_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_empty, 0] = False

        selfatt_out, _ = self.selfatt_mha(
            selfatt_input,
            selfatt_input,
            selfatt_input,
            key_padding_mask=key_padding_mask,
        )

        # Fusion
        fused = torch.cat([cnn_out, selfatt_out], dim=-1)
        fused = self.fusion_dropout(fused)
        fused = self.fusion_projection(fused)

        # Candi-Att
        att_input = torch.cat([fused, cand_repeated], dim=-1)
        flat_att = att_input.reshape(B * H, 2 * D)
        flat_scores = self.dense_att(flat_att)
        att_scores = flat_scores.reshape(B, H)

        att_scores = att_scores.masked_fill(~history_mask, -1e9)
        att_weights = torch.softmax(att_scores, dim=-1)

        user_vec = (fused * att_weights.unsqueeze(-1)).sum(dim=1)

        return (user_vec * can_vec_dropped).sum(dim=-1)


# ---------------------------------------------------------------------------
# Dummy user encoder (BaseModel contract)
# ---------------------------------------------------------------------------


class UserEncoder(nn.Module):
    """Fallback user encoder — mean pool over history."""

    def __init__(self, config: CAUMConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, H, T = inputs.shape
        flat = inputs.reshape(B * H, T)
        flat_vecs = self.news_encoder(flat, training=training)
        news_embeds = flat_vecs.reshape(B, H, -1)

        mask = inputs.any(dim=-1).float()
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (news_embeds * mask.unsqueeze(-1)).sum(dim=1) / count


# ---------------------------------------------------------------------------
# Full CAUM model
# ---------------------------------------------------------------------------


class CAUM(BaseModel):
    """News Recommendation with Candidate-aware User Modeling."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CAUMConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = CAUMConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        # Word embedding
        vocab_size = processed_news["vocab_size"]
        embeddings_matrix = processed_news["embeddings"]
        self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # Entity embedding (frozen)
        entity_emb = None
        if config.use_entity:
            entity_indices = processed_news.get("entity_indices")
            num_entities = processed_news.get("num_entities")
            if num_entities is None and entity_indices is not None:
                num_entities = int(np.max(entity_indices)) + 1
            num_entities = num_entities or 1
            entity_emb = nn.Embedding(num_entities, config.entity_embedding_dim)
            entity_emb.weight.requires_grad = False

        # Category embedding (frozen)
        category_emb = None
        if config.use_category:
            num_categories = processed_news.get("num_categories", 1)
            category_emb = nn.Embedding(
                num_categories + 1, config.category_embedding_dim
            )
            category_emb.weight.requires_grad = False

        self.news_encoder = NewsEncoder(
            config,
            self.embedding_layer,
            entity_emb,
            category_emb,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)
        self.inter_model = InterModel(config)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        hist_tokens = inputs["hist_tokens"]
        cand_tokens = inputs["cand_tokens"]

        B, H, T_h = hist_tokens.shape

        flat_hist = hist_tokens.reshape(B * H, T_h)
        clicked_vecs = self.news_encoder(flat_hist, training=training).reshape(B, H, -1)

        B, C, T_c = cand_tokens.shape
        flat_cand = cand_tokens.reshape(B * C, T_c)
        cand_vecs = self.news_encoder(flat_cand, training=training).reshape(B, C, -1)

        scores = []
        for i in range(self.config.max_impressions_length):
            cand_i = cand_vecs[:, i, :]
            score_i = self.inter_model(cand_i, clicked_vecs, training=training)
            scores.append(score_i)

        return torch.stack(scores, dim=1)
