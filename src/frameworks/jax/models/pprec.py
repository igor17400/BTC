"""PP-Rec: News Recommendation with Personalized User Interest and
Time-aware News Popularity (ACL 2021).

Flax NNX (JAX) port of the Keras implementation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import PPRecConfig

from ..layers import AdditiveAttention, AttentivePoolingQKY
from .base import BaseModel

# ---------------------------------------------------------------------------
# News Encoder (paper co1 variant with bidirectional cross-attention)
# ---------------------------------------------------------------------------


class PPRecNewsEncoder(nnx.Module):
    """Knowledge-aware news encoder for PP-Rec (paper ``co1`` variant).

    Architecture (mirrors Keras and PyTorch ports):
        word_emb -> Dropout -> MHSA(20,20)
        entity_emb         -> MHSA(20,20)
        Bidirectional MHCA(5,40) between word and entity (raw embeddings)
        Each branch: concat[mhsa, mhca] -> Dense -> Dropout -> AdditiveAttn
        Concat[title, entity, category] -> Dense(news_dim)
    """

    def __init__(
        self,
        config: PPRecConfig,
        word_embedding_layer: nnx.Embed,
        entity_embedding_layer: nnx.Embed | None = None,
        category_embedding_layer: nnx.Embed | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        co_out_dim = config.co_num_heads * config.co_head_dim
        self.word_embedding = word_embedding_layer
        self.entity_embedding = entity_embedding_layer
        self.category_embedding = category_embedding_layer

        # --- Word (title) branch ---
        self.word_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.word_mhsa = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.embedding_size,
            qkv_features=config.num_heads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        # Flax NNX MHA has out_features = in_features by default (NOT qkv_features),
        # so the MHSA output stays at embedding_size, not num_heads*head_dim.
        word_concat_dim = config.embedding_size + (
            co_out_dim if entity_embedding_layer is not None else 0
        )
        self.word_proj = nnx.Linear(word_concat_dim, config.news_dim, rngs=rngs)
        self.word_dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.word_attention = AdditiveAttention(
            input_dim=config.news_dim,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

        # --- Entity branch + bidirectional MHCA ---
        if entity_embedding_layer is not None:
            self.entity_mhsa = nnx.MultiHeadAttention(
                num_heads=config.num_heads,
                in_features=config.entity_embedding_dim,
                qkv_features=config.num_heads * config.head_dim,
                decode=False,
                rngs=rngs,
            )
            entity_concat_dim = config.entity_embedding_dim + co_out_dim
            self.entity_proj = nnx.Linear(entity_concat_dim, config.news_dim, rngs=rngs)
            self.entity_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
            self.entity_attention = AdditiveAttention(
                input_dim=config.news_dim,
                query_vec_dim=config.attention_hidden_dim,
                rngs=rngs,
            )

            # Cross-attention requires Q and K/V to share input dim in Flax NNX,
            # so pre-project both modalities to a common 200-d cross-attention dim.
            self.title_q_proj = nnx.Linear(config.embedding_size, co_out_dim, rngs=rngs)
            self.entity_kv_proj = nnx.Linear(
                config.entity_embedding_dim, co_out_dim, rngs=rngs
            )
            self.entity_q_proj = nnx.Linear(
                config.entity_embedding_dim, co_out_dim, rngs=rngs
            )
            self.title_kv_proj = nnx.Linear(
                config.embedding_size, co_out_dim, rngs=rngs
            )
            self.title_mhca = nnx.MultiHeadAttention(
                num_heads=config.co_num_heads,
                in_features=co_out_dim,
                qkv_features=co_out_dim,
                decode=False,
                rngs=rngs,
            )
            self.entity_mhca = nnx.MultiHeadAttention(
                num_heads=config.co_num_heads,
                in_features=co_out_dim,
                qkv_features=co_out_dim,
                decode=False,
                rngs=rngs,
            )

        # --- Category branch ---
        if category_embedding_layer is not None:
            self.category_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # --- Fusion ---
        fusion_dim = config.news_dim
        if entity_embedding_layer is not None:
            fusion_dim += config.news_dim
        if category_embedding_layer is not None:
            fusion_dim += config.category_embedding_dim
        self.fusion_dense = nnx.Linear(fusion_dim, config.news_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        offset = 0
        title_len = self.config.max_title_length
        title_tokens = inputs[:, offset : offset + title_len]
        offset += title_len
        title_keep = jnp.not_equal(title_tokens, 0)  # (B, T)

        has_entity = self.entity_embedding is not None and self.config.use_entity
        if has_entity:
            entity_len = self.config.max_entities
            entity_indices = inputs[:, offset : offset + entity_len]
            offset += entity_len
            entity_keep = jnp.not_equal(entity_indices, 0)  # (B, E)
        else:
            entity_indices = None
            entity_keep = None

        has_category = self.category_embedding is not None
        category_idx = inputs[:, offset : offset + 1] if has_category else None

        # --- Raw embeddings ---
        title_emb = self.word_dropout(
            self.word_embedding(title_tokens), deterministic=not training
        )
        if has_entity:
            entity_emb = self.entity_embedding(entity_indices)

        # --- Bidirectional MHCA on RAW embeddings ---
        if has_entity:
            # Q=title, KV=entity → output 200d
            title_q = self.title_q_proj(title_emb)  # (B, T, 200)
            entity_kv = self.entity_kv_proj(entity_emb)  # (B, E, 200)
            entity_kv_mask = entity_keep[:, None, None, :]  # (B, 1, 1, E)
            title_co = self.title_mhca(
                title_q, entity_kv, mask=entity_kv_mask, deterministic=not training
            )

            # Q=entity, KV=title → output 200d
            entity_q = self.entity_q_proj(entity_emb)  # (B, E, 200)
            title_kv = self.title_kv_proj(title_emb)  # (B, T, 200)
            title_kv_mask = title_keep[:, None, None, :]  # (B, 1, 1, T)
            entity_co = self.entity_mhca(
                entity_q, title_kv, mask=title_kv_mask, deterministic=not training
            )

        # --- Title self-attention + concat ---
        title_self_mask = title_keep[:, None, None, :]
        title_vecs = self.word_mhsa(
            title_emb, title_emb, mask=title_self_mask, deterministic=not training
        )
        if has_entity:
            title_vecs = jnp.concatenate([title_vecs, title_co], axis=-1)
        title_vecs = self.word_proj(title_vecs)
        title_vecs = self.word_dropout2(title_vecs, deterministic=not training)
        title_vec = self.word_attention(title_vecs, mask=title_keep)

        vecs = [title_vec]

        # --- Entity self-attention + concat ---
        if has_entity:
            entity_self_mask = entity_keep[:, None, None, :]
            entity_vecs = self.entity_mhsa(
                entity_emb,
                entity_emb,
                mask=entity_self_mask,
                deterministic=not training,
            )
            entity_vecs = jnp.concatenate([entity_vecs, entity_co], axis=-1)
            entity_vecs = self.entity_proj(entity_vecs)
            entity_vecs = self.entity_dropout(entity_vecs, deterministic=not training)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_keep)
            vecs.append(entity_vec)

        # --- Category branch ---
        if has_category:
            cat_vec = self.category_embedding(category_idx)
            cat_vec = jnp.squeeze(cat_vec, axis=1)
            cat_vec = self.category_dropout(cat_vec, deterministic=not training)
            vecs.append(cat_vec)

        # --- Fusion ---
        fused = jnp.concatenate(vecs, axis=-1) if len(vecs) > 1 else vecs[0]
        return self.fusion_dense(fused)


# ---------------------------------------------------------------------------
# Popularity Predictor
# ---------------------------------------------------------------------------


class PopularityPredictor(nnx.Module):
    """Time-aware news popularity predictor (Flax NNX)."""

    def __init__(self, config: PPRecConfig, *, rngs: nnx.Rngs):
        self.config = config

        c1, c2, c3 = config.pop_content_dims
        self.content_dense1 = nnx.Linear(config.news_dim, c1, rngs=rngs)
        self.content_dense2 = nnx.Linear(c1, c2, rngs=rngs)
        self.content_dense3 = nnx.Linear(c2, c3, rngs=rngs)
        self.content_out = nnx.Linear(c3, 1, use_bias=False, rngs=rngs)

        if config.use_recency:
            r1, r2 = config.pop_recency_dims
            g1, g2 = config.pop_gate_dims
            self.recency_embedding = nnx.Embed(
                num_embeddings=config.recency_embedding_bins,
                features=config.recency_embedding_dim,
                rngs=rngs,
            )
            self.recency_dense1 = nnx.Linear(
                config.recency_embedding_dim, r1, rngs=rngs
            )
            self.recency_dense2 = nnx.Linear(r1, r2, rngs=rngs)
            self.recency_out = nnx.Linear(r2, 1, use_bias=False, rngs=rngs)

            gate_in = config.news_dim + config.recency_embedding_dim
            self.gate_dense1 = nnx.Linear(gate_in, g1, rngs=rngs)
            self.gate_dense2 = nnx.Linear(g1, g2, rngs=rngs)
            self.gate_out = nnx.Linear(g2, 1, rngs=rngs)

        if config.use_ctr:
            self.ctr_scaler = nnx.Linear(1, 1, use_bias=False, rngs=rngs)
            # Initialise to ctr_scaler_init
            self.ctr_scaler.kernel.value = jnp.full(
                (1, 1), config.ctr_scaler_init, dtype=jnp.float32
            )

    def __call__(
        self,
        bias_news_vec: jax.Array,
        recency_indices: jax.Array | None = None,
        ctr_values: jax.Array | None = None,
    ) -> jax.Array:
        x = jnp.tanh(self.content_dense1(bias_news_vec))
        x = jnp.tanh(self.content_dense2(x))
        x = self.content_dense3(x)
        content_score = jnp.squeeze(self.content_out(x), axis=-1)

        pop_score = content_score

        if self.config.use_recency and recency_indices is not None:
            recency_emb = self.recency_embedding(recency_indices.astype(jnp.int32))
            r = jnp.tanh(self.recency_dense1(recency_emb))
            r = jnp.tanh(self.recency_dense2(r))
            recency_score = jnp.squeeze(self.recency_out(r), axis=-1)

            gate_input = jnp.concatenate([bias_news_vec, recency_emb], axis=-1)
            g = jnp.tanh(self.gate_dense1(gate_input))
            g = jnp.tanh(self.gate_dense2(g))
            gate = jnp.squeeze(jax.nn.sigmoid(self.gate_out(g)), axis=-1)

            pop_score = (1.0 - gate) * content_score + gate * recency_score

        if self.config.use_ctr and ctr_values is not None:
            ctr_input = jnp.expand_dims(ctr_values.astype(jnp.float32), axis=-1)
            ctr_score = jnp.squeeze(self.ctr_scaler(ctr_input), axis=-1)
            pop_score = pop_score + ctr_score

        return pop_score


# ---------------------------------------------------------------------------
# User Encoder (CPJA)
# ---------------------------------------------------------------------------


class PPRecUserEncoder(nnx.Module):
    """Popularity-aware user encoder with Content-Popularity Joint Attention."""

    def __init__(
        self,
        config: PPRecConfig,
        news_encoder: PPRecNewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder

        self.user_mhsa = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.news_dim,
            qkv_features=config.num_heads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        self.popularity_embedding = nnx.Embed(
            num_embeddings=config.popularity_embedding_bins,
            features=config.popularity_embedding_dim,
            rngs=rngs,
        )
        cpja_key_dim = config.news_dim + config.popularity_embedding_dim
        self.cpja = AttentivePoolingQKY(
            key_dim=cpja_key_dim,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        history_features: jax.Array,
        history_ctr: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        B, H, F = history_features.shape

        # TimeDistributed encoding
        flat = history_features.reshape(B * H, F)
        news_vecs = self.news_encoder(flat, training=training)
        user_vecs = news_vecs.reshape(B, H, -1)

        history_keep = jnp.any(jnp.not_equal(history_features, 0), axis=-1)
        attn_mask = history_keep[:, None, None, :]
        user_vecs = self.user_mhsa(
            user_vecs, user_vecs, mask=attn_mask, deterministic=not training
        )

        if history_ctr is None:
            history_ctr = jnp.zeros((B, H), dtype=jnp.int32)
        pop_emb = self.popularity_embedding(history_ctr.astype(jnp.int32))

        key_input = jnp.concatenate([user_vecs, pop_emb], axis=-1)
        return self.cpja(key_input, user_vecs, mask=history_keep)


# ---------------------------------------------------------------------------
# Activity Gater
# ---------------------------------------------------------------------------


class ActivityGater(nnx.Module):
    """Per-user gate balancing relevance vs popularity."""

    def __init__(self, config: PPRecConfig, *, rngs: nnx.Rngs):
        h = config.activity_gate_hidden_dim
        self.dense1 = nnx.Linear(config.news_dim, h, rngs=rngs)
        self.dense2 = nnx.Linear(h, 1, rngs=rngs)

    def __call__(self, user_vec: jax.Array) -> jax.Array:
        x = jnp.tanh(self.dense1(user_vec))
        return jnp.squeeze(jax.nn.sigmoid(self.dense2(x)), axis=-1)


# ---------------------------------------------------------------------------
# Main PP-Rec Model
# ---------------------------------------------------------------------------


class PPRec(BaseModel):
    """PP-Rec Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: PPRecConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = PPRecConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        cfg = config
        pn = processed_news
        embeddings_matrix = np.asarray(pn["embeddings"])
        vocab_size = int(pn["vocab_size"])

        # Two separate word embeddings (relevance + bias) per paper
        word_emb = nnx.Embed(
            num_embeddings=vocab_size, features=cfg.embedding_size, rngs=rngs
        )
        word_emb.embedding.value = jnp.asarray(embeddings_matrix)

        bias_word_emb = nnx.Embed(
            num_embeddings=vocab_size, features=cfg.embedding_size, rngs=rngs
        )
        bias_word_emb.embedding.value = jnp.asarray(embeddings_matrix)

        # Entity embedding (trainable, initialised from TransE)
        entity_emb = None
        if cfg.use_entity and "entity_embeddings" in pn:
            entity_matrix = np.asarray(pn["entity_embeddings"])
            entity_emb = nnx.Embed(
                num_embeddings=int(pn["entity_vocab_size"]),
                features=cfg.entity_embedding_dim,
                rngs=rngs,
            )
            entity_emb.embedding.value = jnp.asarray(entity_matrix)

        # Category embedding
        cat_emb = None
        if "num_categories" in pn:
            cat_emb = nnx.Embed(
                num_embeddings=int(pn["num_categories"]) + 1,
                features=cfg.category_embedding_dim,
                rngs=rngs,
            )

        self.news_encoder = PPRecNewsEncoder(
            cfg, word_emb, entity_emb, cat_emb, rngs=rngs
        )
        self.bias_news_encoder = PPRecNewsEncoder(
            cfg, bias_word_emb, entity_emb, cat_emb, rngs=rngs
        )
        self.user_encoder = PPRecUserEncoder(cfg, self.news_encoder, rngs=rngs)
        self.popularity_predictor = PopularityPredictor(cfg, rngs=rngs)
        if cfg.use_activity_gate:
            self.activity_gater = ActivityGater(cfg, rngs=rngs)
        else:
            self.activity_gater = None

    # ---- Scoring helpers ------------------------------------------------

    def score_training_batch(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = True,
    ) -> jax.Array:
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_ctr = inputs.get("cand_ctr")
        cand_recency = inputs.get("cand_recency")

        user_vec = self.user_encoder(hist_features, hist_ctr, training=training)

        B, C, F = cand_features.shape
        flat_cand = cand_features.reshape(B * C, F)
        rel_cand_vecs = self.news_encoder(flat_cand, training=training).reshape(
            B, C, -1
        )
        bias_cand_vecs = self.bias_news_encoder(flat_cand, training=training).reshape(
            B, C, -1
        )

        rel_scores = jnp.sum(rel_cand_vecs * user_vec[:, None, :], axis=-1)

        bias_flat = bias_cand_vecs.reshape(B * C, self.config.news_dim)
        recency_flat = cand_recency.reshape(B * C) if cand_recency is not None else None
        ctr_flat = cand_ctr.reshape(B * C) if cand_ctr is not None else None
        pop_scores = self.popularity_predictor(
            bias_flat, recency_indices=recency_flat, ctr_values=ctr_flat
        ).reshape(B, C)

        if self.activity_gater is not None:
            eta = self.activity_gater(user_vec)[:, None]
            scores = 2.0 * eta * rel_scores + 2.0 * (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores

        return scores

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Forward pass for training. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        return self.score_training_batch(inputs, training=training)
