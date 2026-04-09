"""PP-Rec: News Recommendation with Personalized User Interest and
Time-aware News Popularity (ACL 2021).

PyTorch port of the Keras implementation, mirroring the same class
structure: PPRecNewsEncoder + PopularityPredictor + PPRecUserEncoder +
ActivityGater + PPRec.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import PPRecConfig

from ..layers import AdditiveAttention, AttentivePoolingQKY
from .base import BaseModel

# ---------------------------------------------------------------------------
# News Encoder (paper co1 variant with bidirectional cross-attention)
# ---------------------------------------------------------------------------


class PPRecNewsEncoder(nn.Module):
    """Knowledge-aware news encoder for PP-Rec (paper ``co1`` variant).

    Architecture (mirrors the Keras implementation):
        word_emb  -> Dropout -> MHSA(20,20) ─┐
        entity_emb         -> MHSA(20,20)  ─┤
                                            ├─ bidirectional MHCA(5,40)
        word_co  + word_mhsa -> Dense -> Dropout -> AdditiveAttn -> 400d
        entity_co + entity_mhsa -> Dense -> Dropout -> AdditiveAttn -> 400d
        category_emb -> Dropout -> 200d
        Concat -> Dense(news_dim) -> news vector
    """

    def __init__(
        self,
        config: PPRecConfig,
        word_embedding_layer: nn.Embedding,
        entity_embedding_layer: nn.Embedding | None = None,
        category_embedding_layer: nn.Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        co_out_dim = config.co_num_heads * config.co_head_dim
        self.word_embedding = word_embedding_layer
        self.entity_embedding = entity_embedding_layer
        self.category_embedding = category_embedding_layer

        # --- Word (title) branch ---
        self.word_dropout = nn.Dropout(config.dropout_rate)
        self.word_mhsa = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        # Title MHSA produces (B,T,embed_dim). After concat with title_co
        # (B,T,co_num_heads*co_head_dim=200), project back to news_dim.
        word_concat_dim = config.embedding_size + (
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
            self.entity_mhsa = nn.MultiheadAttention(
                embed_dim=config.entity_embedding_dim,
                num_heads=config.num_heads,
                dropout=config.dropout_rate,
                batch_first=True,
            )
            entity_concat_dim = config.entity_embedding_dim + (co_out_dim)
            self.entity_proj = nn.Linear(entity_concat_dim, config.news_dim)
            self.entity_dropout = nn.Dropout(config.dropout_rate)
            self.entity_attention = AdditiveAttention(
                input_dim=config.news_dim,
                query_vec_dim=config.attention_hidden_dim,
            )
            # Cross-attention: word query against entity key/value (and vice versa).
            # nn.MultiheadAttention requires embed_dim == kdim == vdim, but here Q
            # is in word space (300) and K/V is in entity space (100). Use kdim/vdim
            # to allow different dims. embed_dim is the QUERY dim, kdim/vdim are
            # the K/V dims. Output is num_heads * head_dim = 200d.
            self.title_mhca = nn.MultiheadAttention(
                embed_dim=co_out_dim,  # 200, output dim
                num_heads=config.co_num_heads,
                dropout=config.dropout_rate,
                kdim=config.entity_embedding_dim,
                vdim=config.entity_embedding_dim,
                batch_first=True,
            )
            self.title_query_proj = nn.Linear(config.embedding_size, co_out_dim)
            self.entity_mhca = nn.MultiheadAttention(
                embed_dim=co_out_dim,  # 200, output dim
                num_heads=config.co_num_heads,
                dropout=config.dropout_rate,
                kdim=config.embedding_size,
                vdim=config.embedding_size,
                batch_first=True,
            )
            self.entity_query_proj = nn.Linear(config.entity_embedding_dim, co_out_dim)

        # --- Category branch ---
        if category_embedding_layer is not None:
            self.category_dropout = nn.Dropout(config.dropout_rate)

        # --- Fusion: [title, entity, category] -> Dense(news_dim) ---
        fusion_dim = config.news_dim
        if entity_embedding_layer is not None:
            fusion_dim += config.news_dim
        if category_embedding_layer is not None:
            fusion_dim += config.category_embedding_dim
        self.fusion_dense = nn.Linear(fusion_dim, config.news_dim)

    def _safe_mha(
        self,
        mha: nn.MultiheadAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run MHA with fully-padded-row safety.

        PyTorch nn.MultiheadAttention produces NaN when a query attends to
        all-masked keys. We temporarily unmask one position for fully-padded
        rows, then zero out the output for those rows.
        """
        fully_padded = key_padding_mask.all(dim=-1)
        if fully_padded.any():
            kpm = key_padding_mask.clone()
            kpm[fully_padded, 0] = False
        else:
            kpm = key_padding_mask
        out, _ = mha(q, k, v, key_padding_mask=kpm, need_weights=False)
        if fully_padded.any():
            # Zero out outputs along the query dim where the row was fully padded
            mask_q = (~fully_padded).unsqueeze(-1).unsqueeze(-1).to(out.dtype)
            out = out * mask_q
        return out

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode news features into a news vector.

        Args:
            inputs: ``(batch, feature_dim)`` int32 tensor — concatenation of
                ``[title_tokens, entity_indices, category_index]`` (entity
                and category are optional).

        Returns:
            ``(batch, news_dim)`` news representation.
        """
        offset = 0
        title_len = self.config.max_title_length

        # --- Slice features ---
        title_tokens = inputs[:, offset : offset + title_len].long()
        offset += title_len
        title_pad = title_tokens == 0  # (B, T) True = padded
        title_keep = ~title_pad  # for AdditiveAttention (True = keep)

        has_entity = self.entity_embedding is not None and self.config.use_entity
        if has_entity:
            entity_len = self.config.max_entities
            entity_indices = inputs[:, offset : offset + entity_len].long()
            offset += entity_len
            entity_pad = entity_indices == 0
            entity_keep = ~entity_pad
        else:
            entity_indices = None
            entity_pad = None
            entity_keep = None

        has_category = self.category_embedding is not None
        if has_category:
            category_idx = inputs[:, offset : offset + 1].long()
        else:
            category_idx = None

        # --- Raw embeddings ---
        title_emb = self.word_dropout(self.word_embedding(title_tokens))  # (B, T, 300)
        if has_entity:
            entity_emb = self.entity_embedding(entity_indices)  # (B, E, 100)

        # --- Bidirectional MHCA on RAW embeddings ---
        if has_entity:
            # Word-conditioned-on-entity: Q from title, K/V from entities → 200d
            title_q_proj = self.title_query_proj(title_emb)  # (B, T, 200)
            title_co = self._safe_mha(
                self.title_mhca, title_q_proj, entity_emb, entity_emb, entity_pad
            )  # (B, T, 200)
            # Entity-conditioned-on-word: Q from entities, K/V from title → 200d
            entity_q_proj = self.entity_query_proj(entity_emb)  # (B, E, 200)
            entity_co = self._safe_mha(
                self.entity_mhca, entity_q_proj, title_emb, title_emb, title_pad
            )  # (B, E, 200)

        # --- Title self-attention + concat with co-attention ---
        title_vecs = self._safe_mha(
            self.word_mhsa, title_emb, title_emb, title_emb, title_pad
        )  # (B, T, 300)
        if has_entity:
            title_vecs = torch.cat([title_vecs, title_co], dim=-1)  # (B, T, 500)
        title_vecs = self.word_proj(title_vecs)  # (B, T, news_dim)
        title_vecs = self.word_dropout2(title_vecs)
        title_vec = self.word_attention(title_vecs, mask=title_keep)  # (B, news_dim)

        vecs = [title_vec]

        # --- Entity self-attention + concat with co-attention ---
        if has_entity:
            entity_vecs = self._safe_mha(
                self.entity_mhsa, entity_emb, entity_emb, entity_emb, entity_pad
            )  # (B, E, 100)
            entity_vecs = torch.cat([entity_vecs, entity_co], dim=-1)  # (B, E, 300)
            entity_vecs = self.entity_proj(entity_vecs)  # (B, E, news_dim)
            entity_vecs = self.entity_dropout(entity_vecs)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_keep)
            vecs.append(entity_vec)

        # --- Category branch ---
        if has_category:
            cat_vec = self.category_embedding(category_idx)  # (B, 1, cat_dim)
            cat_vec = cat_vec.squeeze(1)
            cat_vec = self.category_dropout(cat_vec)
            vecs.append(cat_vec)

        # --- Fusion ---
        fused = torch.cat(vecs, dim=-1) if len(vecs) > 1 else vecs[0]
        return self.fusion_dense(fused)


# ---------------------------------------------------------------------------
# Popularity Predictor
# ---------------------------------------------------------------------------


class PopularityPredictor(nn.Module):
    """Time-aware news popularity predictor (PyTorch).

    content scorer: news_dim -> 256 -> 256 -> 128 -> 1
    recency scorer: recency_emb_dim -> 64 -> 64 -> 1
    gate:           [news_vec, recency_emb] -> 128 -> 64 -> 1 (sigmoid)
    ctr scaler:     Dense(1, no_bias) initialized to ``ctr_scaler_init``
    """

    def __init__(self, config: PPRecConfig):
        super().__init__()
        self.config = config

        # Content scorer: news_dim -> pop_content_dims -> 1
        c1, c2, c3 = config.pop_content_dims
        self.content_dense1 = nn.Linear(config.news_dim, c1)
        self.content_dense2 = nn.Linear(c1, c2)
        self.content_dense3 = nn.Linear(c2, c3)
        self.content_out = nn.Linear(c3, 1, bias=False)

        if config.use_recency:
            r1, r2 = config.pop_recency_dims
            g1, g2 = config.pop_gate_dims
            self.recency_embedding = nn.Embedding(
                config.recency_embedding_bins, config.recency_embedding_dim
            )
            self.recency_dense1 = nn.Linear(config.recency_embedding_dim, r1)
            self.recency_dense2 = nn.Linear(r1, r2)
            self.recency_out = nn.Linear(r2, 1, bias=False)
            gate_in = config.news_dim + config.recency_embedding_dim
            self.gate_dense1 = nn.Linear(gate_in, g1)
            self.gate_dense2 = nn.Linear(g1, g2)
            self.gate_out = nn.Linear(g2, 1)

        if config.use_ctr:
            self.ctr_scaler = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.ctr_scaler.weight.fill_(config.ctr_scaler_init)

    def forward(
        self,
        bias_news_vec: torch.Tensor,
        recency_indices: torch.Tensor | None = None,
        ctr_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute popularity score.

        Args:
            bias_news_vec: ``(batch, news_dim)`` from bias news encoder.
            recency_indices: ``(batch,)`` int recency bucket indices.
            ctr_values: ``(batch,)`` float CTR values.

        Returns:
            ``(batch,)`` popularity score.
        """
        x = torch.tanh(self.content_dense1(bias_news_vec))
        x = torch.tanh(self.content_dense2(x))
        x = self.content_dense3(x)
        content_score = self.content_out(x).squeeze(-1)

        pop_score = content_score

        if self.config.use_recency and recency_indices is not None:
            recency_emb = self.recency_embedding(recency_indices.long())
            r = torch.tanh(self.recency_dense1(recency_emb))
            r = torch.tanh(self.recency_dense2(r))
            recency_score = self.recency_out(r).squeeze(-1)

            gate_input = torch.cat([bias_news_vec, recency_emb], dim=-1)
            g = torch.tanh(self.gate_dense1(gate_input))
            g = torch.tanh(self.gate_dense2(g))
            gate = torch.sigmoid(self.gate_out(g)).squeeze(-1)

            pop_score = (1.0 - gate) * content_score + gate * recency_score

        if self.config.use_ctr and ctr_values is not None:
            ctr_input = ctr_values.unsqueeze(-1).float()
            ctr_score = self.ctr_scaler(ctr_input).squeeze(-1)
            pop_score = pop_score + ctr_score

        return pop_score


# ---------------------------------------------------------------------------
# User Encoder (CPJA)
# ---------------------------------------------------------------------------


class PPRecUserEncoder(nn.Module):
    """Popularity-aware user encoder with Content-Popularity Joint Attention."""

    def __init__(self, config: PPRecConfig, news_encoder: PPRecNewsEncoder):
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
            history_features: ``(batch, history_len, feature_dim)``.
            history_ctr: ``(batch, history_len)`` int discretized CTR.

        Returns:
            ``(batch, news_dim)`` user representation.
        """
        B, H, F = history_features.shape

        # TimeDistributed: encode each history news item
        flat = history_features.reshape(B * H, F)
        news_vecs = self.news_encoder(flat, training=training)
        user_vecs = news_vecs.reshape(B, H, -1)

        # History mask (a row is valid if any feature is non-zero)
        history_keep = (history_features != 0).any(dim=-1)  # (B, H) True = keep
        key_padding_mask = ~history_keep
        fully_empty = ~history_keep.any(dim=-1)
        if fully_empty.any():
            kpm = key_padding_mask.clone()
            kpm[fully_empty, 0] = False
        else:
            kpm = key_padding_mask

        user_vecs, _ = self.user_mhsa(
            user_vecs, user_vecs, user_vecs, key_padding_mask=kpm, need_weights=False
        )
        if fully_empty.any():
            user_vecs = user_vecs * (~fully_empty).unsqueeze(-1).unsqueeze(-1).to(
                user_vecs.dtype
            )

        # Popularity embedding (CTR-only per paper footnote 2)
        if history_ctr is None:
            history_ctr = torch.zeros(B, H, dtype=torch.long, device=user_vecs.device)
        pop_emb = self.popularity_embedding(history_ctr.long())

        # CPJA: attention from concat[user_vecs, pop_emb], weighted sum of user_vecs
        key_input = torch.cat([user_vecs, pop_emb], dim=-1)
        return self.cpja(key_input, user_vecs, mask=history_keep)


# ---------------------------------------------------------------------------
# Activity Gater
# ---------------------------------------------------------------------------


class ActivityGater(nn.Module):
    """Per-user gate balancing relevance vs popularity."""

    def __init__(self, config: PPRecConfig):
        super().__init__()
        h = config.activity_gate_hidden_dim
        self.dense1 = nn.Linear(config.news_dim, h)
        self.dense2 = nn.Linear(h, 1)

    def forward(self, user_vec: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense1(user_vec))
        return torch.sigmoid(self.dense2(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Main PP-Rec Model
# ---------------------------------------------------------------------------


class PPRec(BaseModel):
    """PP-Rec PyTorch implementation. Two news encoders, popularity predictor,
    popularity-aware user encoder, and activity gate."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: PPRecConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = PPRecConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        cfg = config
        pn = processed_news

        # Shared word embedding (initialised from GloVe)
        word_emb = nn.Embedding(pn["vocab_size"], cfg.embedding_size)
        word_emb.weight = nn.Parameter(
            torch.tensor(pn["embeddings"], dtype=torch.float32)
        )

        bias_word_emb = nn.Embedding(pn["vocab_size"], cfg.embedding_size)
        bias_word_emb.weight = nn.Parameter(
            torch.tensor(pn["embeddings"], dtype=torch.float32)
        )

        # Entity embedding (trainable per paper co1)
        entity_emb = None
        if cfg.use_entity and "entity_embeddings" in pn:
            entity_emb = nn.Embedding(pn["entity_vocab_size"], cfg.entity_embedding_dim)
            entity_emb.weight = nn.Parameter(
                torch.tensor(pn["entity_embeddings"], dtype=torch.float32)
            )

        # Category embedding
        cat_emb = None
        if "num_categories" in pn:
            cat_emb = nn.Embedding(pn["num_categories"] + 1, cfg.category_embedding_dim)

        self.news_encoder = PPRecNewsEncoder(cfg, word_emb, entity_emb, cat_emb)
        self.bias_news_encoder = PPRecNewsEncoder(
            cfg, bias_word_emb, entity_emb, cat_emb
        )
        self.user_encoder = PPRecUserEncoder(cfg, self.news_encoder)
        self.popularity_predictor = PopularityPredictor(cfg)
        self.activity_gater = ActivityGater(cfg) if cfg.use_activity_gate else None

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass for training. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        return self.score_training_batch(inputs)

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_ctr = inputs.get("cand_ctr")
        cand_recency = inputs.get("cand_recency")

        user_vec = self.user_encoder(hist_features, hist_ctr, training=True)

        B, C, F = cand_features.shape
        flat_cand = cand_features.reshape(B * C, F)
        rel_cand_vecs = self.news_encoder(flat_cand, training=True).reshape(B, C, -1)
        bias_cand_vecs = self.bias_news_encoder(flat_cand, training=True).reshape(
            B, C, -1
        )

        rel_scores = torch.sum(rel_cand_vecs * user_vec.unsqueeze(1), dim=-1)

        bias_flat = bias_cand_vecs.reshape(B * C, self.config.news_dim)
        recency_flat = cand_recency.reshape(B * C) if cand_recency is not None else None
        ctr_flat = cand_ctr.reshape(B * C) if cand_ctr is not None else None
        pop_scores = self.popularity_predictor(
            bias_flat, recency_indices=recency_flat, ctr_values=ctr_flat
        ).reshape(B, C)

        if self.activity_gater is not None:
            eta = self.activity_gater(user_vec).unsqueeze(-1)
            scores = 2.0 * eta * rel_scores + 2.0 * (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores

        return scores
