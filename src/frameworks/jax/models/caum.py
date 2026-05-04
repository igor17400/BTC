"""CAUM (Candidate-Aware User Modeling) -- Flax NNX.

Reference: Qi et al., "News Recommendation with Candidate-aware User
Modeling", SIGIR 2022.

CAUM incorporates candidate news information into user modeling so that
the user representation is tailored per candidate.  Three candidate-aware
modules accomplish this:

1. Candi-SelfAtt — candidate-aware multi-head self-attention over the
   click sequence to capture long-range global user interests.
2. Candi-CNN — candidate-aware CNN over adjacent clicks to capture
   short-term local user interests.
3. Candi-Att — candidate-aware attention to aggregate clicked news
   weighted by their relevance to the candidate.

Because the user representation depends on the candidate, the standard
"precompute user vectors -> dot product" evaluation cannot be used.
CAUM uses a custom evaluator that precomputes news vectors and then
scores each impression by running the inter-model per candidate.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import CAUMConfig

from ..layers import AdditiveAttention
from .base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(nnx.Module):
    """Encode a news article from title, entities, and category.

    Following the reference code:
        title:    Embedding -> Dropout -> MHSA(20h×20d) -> Dropout -> AdditivePool
        entity:   Embedding -> Dropout -> MHSA(4h×40d)  -> Dropout -> AdditivePool
        category: Embedding -> Reshape -> Dropout -> Dense
        fusion:   Concat([title, entity, category]) -> Dense(news_dim)

    When entities/category are disabled, falls back to title-only.
    Input is [title_tokens | entity_indices | category_index].
    """

    def __init__(
        self,
        config: CAUMConfig,
        embedding_layer: nnx.Embed,
        entity_embedding_layer: nnx.Embed | None = None,
        category_embedding_layer: nnx.Embed | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding_layer = embedding_layer
        self.entity_embedding = entity_embedding_layer
        self.category_embedding = category_embedding_layer

        # --- Title branch ---
        self.title_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.title_mhsa = nnx.MultiHeadAttention(
            num_heads=config.news_num_heads,
            in_features=config.embedding_size,
            qkv_features=config.news_num_heads * config.news_head_dim,
            decode=False,
            rngs=rngs,
        )
        self.title_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.title_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.news_attention_hidden_dim,
            rngs=rngs,
        )

        # --- Entity branch (reference: 4h×40d=160, frozen embeddings) ---
        if entity_embedding_layer is not None:
            entity_out_dim = config.entity_num_heads * config.entity_head_dim
            self.entity_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
            self.entity_mhsa = nnx.MultiHeadAttention(
                num_heads=config.entity_num_heads,
                in_features=config.entity_embedding_dim,
                qkv_features=entity_out_dim,
                decode=False,
                rngs=rngs,
            )
            self.entity_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
            self.entity_attention = AdditiveAttention(
                input_dim=config.entity_embedding_dim,
                query_vec_dim=config.news_attention_hidden_dim,
                rngs=rngs,
            )

        # --- Category branch ---
        if category_embedding_layer is not None:
            self.category_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
            self.category_dense = nnx.Linear(
                config.category_embedding_dim,
                config.category_embedding_dim,
                rngs=rngs,
            )

        # --- Fusion -> news_dim ---
        # Compute input dim for fusion: title_mhsa_out + entity_mhsa_out + category_dim
        fusion_in = config.embedding_size  # title
        if entity_embedding_layer is not None:
            fusion_in += config.entity_embedding_dim
        if category_embedding_layer is not None:
            fusion_in += config.category_embedding_dim
        self.fusion_dense = nnx.Linear(fusion_in, config.news_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Encode news features.

        Args:
            inputs: (batch, feature_dim) where feature_dim is
                [title_tokens | entity_indices | category_index].

        Returns:
            (batch, news_dim) news vector.
        """
        offset = 0
        title_len = self.config.max_title_length

        # --- Title ---
        title_tokens = inputs[:, offset : offset + title_len]
        offset += title_len
        title_mask = jnp.not_equal(title_tokens, 0)
        title_attn_mask = title_mask[:, None, None, :]

        title_emb = self.title_dropout(
            self.embedding_layer(title_tokens), deterministic=not training
        )
        title_vecs = self.title_mhsa(
            title_emb, title_emb, mask=title_attn_mask, deterministic=not training
        )
        title_vecs = self.title_dropout2(title_vecs, deterministic=not training)
        title_vec = self.title_attention(title_vecs, mask=title_mask)

        vecs = [title_vec]

        # --- Entity ---
        has_entity = self.entity_embedding is not None and self.config.use_entity
        if has_entity:
            entity_len = self.config.max_entities
            entity_indices = inputs[:, offset : offset + entity_len]
            offset += entity_len
            entity_mask = jnp.not_equal(entity_indices, 0)
            entity_attn_mask = entity_mask[:, None, None, :]

            entity_emb = self.entity_dropout(
                self.entity_embedding(entity_indices), deterministic=not training
            )
            entity_vecs = self.entity_mhsa(
                entity_emb,
                entity_emb,
                mask=entity_attn_mask,
                deterministic=not training,
            )
            entity_vecs = self.entity_dropout2(entity_vecs, deterministic=not training)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_mask)
            vecs.append(entity_vec)

        # --- Category ---
        has_category = self.category_embedding is not None and self.config.use_category
        if has_category:
            category_idx = inputs[:, offset : offset + 1]
            offset += 1

            cat_emb = self.category_embedding(category_idx)
            cat_emb = cat_emb.reshape(cat_emb.shape[0], -1)
            cat_emb = self.category_dropout(cat_emb, deterministic=not training)
            cat_vec = self.category_dense(cat_emb)
            vecs.append(cat_vec)

        # --- Fusion ---
        if len(vecs) > 1:
            fused = jnp.concatenate(vecs, axis=-1)
        else:
            fused = vecs[0]
        return self.fusion_dense(fused)


# ---------------------------------------------------------------------------
# Candidate-aware user interaction model (inter_model)
# ---------------------------------------------------------------------------


class DenseAttentionScorer(nnx.Module):
    """Two-layer MLP that scores a concatenated [click, candidate] pair."""

    def __init__(self, hidden_dim: int, mid_dim: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, mid_dim, rngs=rngs)
        self.dense3 = nnx.Linear(mid_dim, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """inputs: (batch, 2*news_dim) -> (batch, 1)."""
        return self.dense3(jnp.tanh(self.dense2(jnp.tanh(self.dense1(inputs)))))


class InterModel(nnx.Module):
    """Candidate-aware user interest model.

    Takes pre-encoded clicked news vectors and a single candidate news
    vector, and produces a scalar matching score.

    Architecture (following the reference code):
        1. Candi-CNN: circular-shift window over clicks + candidate -> Dense
        2. Candi-SelfAtt: concat candidate to each click -> Dense -> MHSA
        3. Fusion: concat [cnn, selfatt] -> Dropout -> Dense
        4. Candi-Att: concat [fused, candidate] -> DenseAttScorer -> softmax
           -> weighted sum -> user_vec
        5. Score: dot(user_vec, dropout(candidate))
    """

    def __init__(self, config: CAUMConfig, *, rngs: nnx.Rngs):
        D = config.news_dim
        self.config = config

        self.dropout_cand = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.dropout_clicks = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # Candi-CNN: projects [left, center, right, candidate] -> D
        self.cnn_projection = nnx.Linear(4 * D, D, rngs=rngs)

        # Candi-SelfAtt: projects [candidate, click] -> D, then MHSA
        self.selfatt_input_projection = nnx.Linear(2 * D, D, rngs=rngs)
        self.selfatt_mha = nnx.MultiHeadAttention(
            num_heads=config.candi_selfatt_num_heads,
            in_features=D,
            qkv_features=config.candi_selfatt_num_heads * config.candi_selfatt_head_dim,
            decode=False,
            rngs=rngs,
        )

        # Fusion
        self.fusion_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.fusion_projection = nnx.Linear(2 * D, D, rngs=rngs)

        # Candi-Att (DNN scorer applied per click)
        self.dense_att = DenseAttentionScorer(
            hidden_dim=config.candi_att_hidden_dim,
            mid_dim=config.candi_att_mid_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        cand_vec: jax.Array,
        clicked_vecs: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Score a candidate against clicked news.

        Args:
            cand_vec: (B, D) single candidate news vector.
            clicked_vecs: (B, H, D) encoded clicked news vectors.

        Returns:
            (B,) scalar matching scores.
        """
        D = self.config.news_dim
        B, H = clicked_vecs.shape[:2]

        can_vec_dropped = self.dropout_cand(cand_vec, deterministic=not training)
        user_vecs = self.dropout_clicks(clicked_vecs, deterministic=not training)

        # Repeat candidate across history length: (B, H, D)
        cand_repeated = jnp.repeat(cand_vec[:, None, :], H, axis=1)

        # ----- Candi-CNN -----
        left = jnp.concatenate([user_vecs[:, -1:, :], user_vecs[:, :-1, :]], axis=1)
        right = jnp.concatenate([user_vecs[:, 1:, :], user_vecs[:, :1, :]], axis=1)
        cnn_input = jnp.concatenate(
            [left, user_vecs, right, cand_repeated], axis=-1
        )  # (B, H, 4*D)
        cnn_out = self.cnn_projection(cnn_input)  # (B, H, D)

        # ----- Candi-SelfAtt -----
        selfatt_input = jnp.concatenate(
            [cand_repeated, user_vecs], axis=-1
        )  # (B, H, 2*D)
        selfatt_input = self.selfatt_input_projection(selfatt_input)  # (B, H, D)

        # History mask for MHSA (True = valid position)
        history_mask = jnp.any(jnp.not_equal(clicked_vecs, 0.0), axis=-1)  # (B, H)
        attn_mask = history_mask[:, None, None, :]  # (B, 1, 1, H)

        selfatt_out = self.selfatt_mha(
            selfatt_input,
            selfatt_input,
            mask=attn_mask,
            deterministic=not training,
        )  # (B, H, D)

        # ----- Fusion -----
        fused = jnp.concatenate([cnn_out, selfatt_out], axis=-1)  # (B, H, 2*D)
        fused = self.fusion_dropout(fused, deterministic=not training)
        fused = self.fusion_projection(fused)  # (B, H, D)

        # ----- Candi-Att -----
        att_input = jnp.concatenate([fused, cand_repeated], axis=-1)  # (B, H, 2*D)

        flat_att = att_input.reshape(B * H, 2 * D)
        flat_scores = self.dense_att(flat_att)  # (B*H, 1)
        att_scores = flat_scores.reshape(B, H)  # (B, H)

        # Mask out padding positions before softmax
        att_scores = jnp.where(history_mask, att_scores, -1e9)
        att_weights = jax.nn.softmax(att_scores, axis=-1)  # (B, H)

        # Weighted sum -> user vector
        user_vec = jnp.sum(fused * att_weights[:, :, None], axis=1)  # (B, D)

        # Score: dot(user_vec, candidate)
        return jnp.sum(user_vec * can_vec_dropped, axis=-1)  # (B,)


# ---------------------------------------------------------------------------
# Dummy user encoder (satisfies BaseModel contract)
# ---------------------------------------------------------------------------


class UserEncoder(nnx.Module):
    """Fallback user encoder for BaseModel contract compatibility.

    CAUM's actual user modeling is candidate-aware and lives in
    InterModel.  This encoder simply mean-pools the news encoder
    output over history.
    """

    def __init__(self, config: CAUMConfig, news_encoder: NewsEncoder):
        self.config = config
        self.news_encoder = news_encoder

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """inputs: (B, H, T) -> (B, news_dim)."""
        B, H, T = inputs.shape

        flat = inputs.reshape(B * H, T)
        flat_vecs = self.news_encoder(flat, training=training)
        news_embeds = flat_vecs.reshape(B, H, -1)

        mask = jnp.any(jnp.not_equal(inputs, 0), axis=-1)  # (B, H)
        mask_f = mask.astype(news_embeds.dtype)
        count = jnp.maximum(jnp.sum(mask_f, axis=-1, keepdims=True), 1.0)
        return jnp.sum(news_embeds * mask_f[:, :, None], axis=1) / count


# ---------------------------------------------------------------------------
# Full CAUM model
# ---------------------------------------------------------------------------


class CAUM(BaseModel):
    """News Recommendation with Candidate-aware User Modeling.

    During training, ``__call__`` encodes all clicked news once, then
    loops over each candidate to score it via the candidate-aware
    ``inter_model``.

    During evaluation, a custom evaluator precomputes news vectors and
    scores each impression by running the inter_model per candidate.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CAUMConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = CAUMConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        # Shared word embedding initialised from pre-trained weights
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        vocab_size = int(processed_news["vocab_size"])
        self.embedding_layer = nnx.Embed(
            num_embeddings=vocab_size,
            features=config.embedding_size,
            rngs=rngs,
        )
        self.embedding_layer.embedding.value = jnp.asarray(embeddings_matrix)

        # Entity embedding (frozen)
        entity_emb_layer = None
        if config.use_entity:
            entity_indices = processed_news.get("entity_indices")
            num_entities = processed_news.get("num_entities")
            if num_entities is None and entity_indices is not None:
                num_entities = int(np.max(entity_indices)) + 1
            num_entities = num_entities or 1
            entity_emb_layer = nnx.Embed(
                num_embeddings=num_entities,
                features=config.entity_embedding_dim,
                rngs=rngs,
            )

        # Category embedding (frozen)
        category_emb_layer = None
        if config.use_category:
            num_categories = int(processed_news.get("num_categories", 1))
            category_emb_layer = nnx.Embed(
                num_embeddings=num_categories + 1,
                features=config.category_embedding_dim,
                rngs=rngs,
            )

        # Sub-modules
        self.news_encoder = NewsEncoder(
            config,
            self.embedding_layer,
            entity_embedding_layer=entity_emb_layer,
            category_embedding_layer=category_emb_layer,
            rngs=rngs,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)
        self.inter_model = InterModel(config, rngs=rngs)

    def score_training_batch(
        self,
        hist_tokens: jax.Array,
        cand_tokens: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score a training batch.

        Encodes clicked history once, then scores each candidate
        through the candidate-aware inter_model.

        Returns:
            (B, C) logit scores.
        """
        B, H, T_h = hist_tokens.shape

        # Encode clicked news: (B*H, T) -> (B, H, D)
        flat_hist = hist_tokens.reshape(B * H, T_h)
        clicked_vecs = self.news_encoder(flat_hist, training=training).reshape(B, H, -1)

        # Encode candidate news: (B*C, T) -> (B, C, D)
        B, C, T_c = cand_tokens.shape
        flat_cand = cand_tokens.reshape(B * C, T_c)
        cand_vecs = self.news_encoder(flat_cand, training=training).reshape(B, C, -1)

        # Score each candidate via inter_model
        scores = []
        for i in range(self.config.max_impressions_length):
            cand_i = cand_vecs[:, i, :]  # (B, D)
            score_i = self.inter_model(cand_i, clicked_vecs, training=training)  # (B,)
            scores.append(score_i)

        return jnp.stack(scores, axis=1)  # (B, C)

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        return self.score_training_batch(
            inputs["hist_tokens"], inputs["cand_tokens"], training=training
        )
