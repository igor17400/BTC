"""TCCM-faithful Keras layers.

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

import keras
from keras import layers, ops

from src.core.models.configs import TCCMConfig
from src.frameworks.keras.layers import AdditiveAttention, GlorotUniformMHA


# ---------------------------------------------------------------------------
# News encoder (paper ``co1``)
# ---------------------------------------------------------------------------


class TCCMNewsEncoder(keras.Model):
    """Knowledge-aware news encoder for TCCM.

    Architecture (matches ``model.get_news_encoder_co1`` exactly)::

        title_emb  = WordEmb(title_tokens)        no input dropout
        entity_emb = EntityEmb(entity_indices)

        title_co  = MHCA(20, 20)([title_emb, entity_emb, entity_emb])  # 400d
        entity_co = MHCA(20, 20)([entity_emb, title_emb, title_emb])   # 400d

        title_self  = MHSA(20, 20)(title_emb)                          # 400d
        title_seq   = Add(title_self, title_co); Dropout(0.2)
        title_vec   = AttLayer(400)(title_seq)                         # 400d

        entity_self = MHSA(20, 20)(entity_emb)                         # 400d
        entity_seq  = Add(entity_self, entity_co); Dropout(0.2)
        entity_vec  = AttLayer(400)(entity_seq)                        # 400d

        stacked = stack([title_vec, entity_vec], axis=1); Dropout(0.2)
        news_vec = AttLayer(400)(stacked)                              # 400d
    """

    def __init__(
        self,
        config: TCCMConfig,
        word_embedding_layer: layers.Embedding,
        entity_embedding_layer: layers.Embedding,
        name: str = "tccm_news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.word_embedding = word_embedding_layer
        self.entity_embedding = entity_embedding_layer

        proj_dim = config.num_heads * config.head_dim  # 400 by default

        self.title_mhsa = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            output_shape=proj_dim,
            kernel_initializer=GlorotUniformMHA(),
            name=f"{name}_title_mhsa",
        )
        self.entity_mhsa = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            output_shape=proj_dim,
            kernel_initializer=GlorotUniformMHA(),
            name=f"{name}_entity_mhsa",
        )
        co_proj = config.co_num_heads * config.co_head_dim  # 400 by default
        self.title_mhca = layers.MultiHeadAttention(
            num_heads=config.co_num_heads,
            key_dim=config.co_head_dim,
            output_shape=co_proj,
            kernel_initializer=GlorotUniformMHA(),
            name=f"{name}_title_mhca",
        )
        self.entity_mhca = layers.MultiHeadAttention(
            num_heads=config.co_num_heads,
            key_dim=config.co_head_dim,
            output_shape=co_proj,
            kernel_initializer=GlorotUniformMHA(),
            name=f"{name}_entity_mhca",
        )

        self.title_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.entity_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.fusion_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

        # AttLayer(400) — hidden attention dim equals news_dim.
        self.title_attention = AdditiveAttention(
            query_vec_dim=config.attention_hidden_dim,
            seed=config.seed,
            name=f"{name}_title_additive",
        )
        self.entity_attention = AdditiveAttention(
            query_vec_dim=config.attention_hidden_dim,
            seed=config.seed,
            name=f"{name}_entity_additive",
        )
        self.fusion_attention = AdditiveAttention(
            query_vec_dim=config.attention_hidden_dim,
            seed=config.seed,
            name=f"{name}_fusion_additive",
        )

    def call(self, inputs, training=None):
        """Encode news features into a news vector.

        Args:
            inputs: ``(batch, T+E)`` int32 — concatenated title tokens and
                entity indices.

        Returns:
            ``(batch, news_dim)`` news vector.
        """
        title_len = self.config.max_title_length
        title_tokens = inputs[:, :title_len]
        entity_indices = inputs[:, title_len:]
        title_mask = ops.not_equal(title_tokens, 0)
        entity_mask = ops.not_equal(entity_indices, 0)

        title_emb = self.word_embedding(title_tokens)
        entity_emb = self.entity_embedding(entity_indices)

        title_co = self.title_mhca(
            title_emb,
            entity_emb,
            entity_emb,
            key_mask=entity_mask,
            value_mask=entity_mask,
            training=training,
        )
        entity_co = self.entity_mhca(
            entity_emb,
            title_emb,
            title_emb,
            key_mask=title_mask,
            value_mask=title_mask,
            training=training,
        )

        title_self = self.title_mhsa(
            title_emb,
            title_emb,
            title_emb,
            key_mask=title_mask,
            value_mask=title_mask,
            training=training,
        )
        title_seq = title_self + title_co
        title_seq = self.title_dropout(title_seq, training=training)
        title_vec = self.title_attention(title_seq, mask=title_mask)

        entity_self = self.entity_mhsa(
            entity_emb,
            entity_emb,
            entity_emb,
            key_mask=entity_mask,
            value_mask=entity_mask,
            training=training,
        )
        entity_seq = entity_self + entity_co
        entity_seq = self.entity_dropout(entity_seq, training=training)
        entity_vec = self.entity_attention(entity_seq, mask=entity_mask)

        stacked = ops.stack([title_vec, entity_vec], axis=1)
        stacked = self.fusion_dropout(stacked, training=training)
        return self.fusion_attention(stacked)


# ---------------------------------------------------------------------------
# Popularity encoder (paper ``get_popularity_encoder``)
# ---------------------------------------------------------------------------


class TCCMPopularityEncoder(keras.Model):
    """Content-aware popularity encoder for TCCM.

    Input layout::

        bucket_input : (B, T+E)  int32 — per-token CTR bucket indices
        time_input   : (B,)      int32 — clamped age in hours

    Output: ``(B,)`` popularity score = ``s_p · t'^(-λ)``.
    """

    def __init__(self, config: TCCMConfig, name: str = "tccm_pop_encoder"):
        super().__init__(name=name)
        self.config = config

        # Shared word/entity popularity-bucket embedding (paper reference
        # uses one Embedding(200, 200) for both branches).
        self.token_pop_embedding = layers.Embedding(
            config.pop_token_embedding_bins,
            config.pop_token_embedding_dim,
            name=f"{name}_token_pop_embedding",
        )
        self.title_dropout1 = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.entity_dropout1 = layers.Dropout(config.dropout_rate, seed=config.seed)

        # Bidirectional cross-attention + per-branch self-attention. The
        # paper uses ``Self_Attention(400, 1)`` = 400 heads × 1 dim, so
        # ``num_heads × key_dim = 400`` and the output is also 400-d.
        proj_dim = config.pop_num_heads * config.pop_head_dim

        def _mha(name_suffix: str) -> layers.MultiHeadAttention:
            return layers.MultiHeadAttention(
                num_heads=config.pop_num_heads,
                key_dim=config.pop_head_dim,
                output_shape=proj_dim,
                dropout=config.dropout_rate,
                kernel_initializer=GlorotUniformMHA(),
                name=f"{name}_{name_suffix}",
            )

        self.title_co_mhca = _mha("title_mhca")
        self.entity_co_mhca = _mha("entity_mhca")
        self.title_mhsa = _mha("title_mhsa")
        self.entity_mhsa = _mha("entity_mhsa")

        self.title_dropout2 = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.entity_dropout2 = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.fusion_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

        # ``AttLayer(400, seed)`` — hidden dim equals proj_dim.
        self.title_attention = AdditiveAttention(
            query_vec_dim=proj_dim, seed=config.seed, name=f"{name}_title_additive"
        )
        self.entity_attention = AdditiveAttention(
            query_vec_dim=proj_dim, seed=config.seed, name=f"{name}_entity_additive"
        )
        self.fusion_attention = AdditiveAttention(
            query_vec_dim=proj_dim, seed=config.seed, name=f"{name}_fusion_additive"
        )

        # Content MLP head: pop_vec → 256 → 256 → 128 → sigmoid(1).
        c1, c2, c3 = config.pop_content_dims
        self.content_d1 = layers.Dense(c1, activation="tanh", name=f"{name}_content_d1")
        self.content_d2 = layers.Dense(c2, name=f"{name}_content_d2")
        self.content_d3 = layers.Dense(c3, name=f"{name}_content_d3")
        self.content_out = layers.Dense(
            1, activation="sigmoid", name=f"{name}_content_out"
        )

        # Timeliness module: emb → 64 → 64 → sigmoid(1) → reciprocal-power.
        self.time_embedding = layers.Embedding(
            config.timeliness_embedding_bins,
            config.timeliness_embedding_dim,
            name=f"{name}_time_embedding",
        )
        r1, r2 = config.pop_recency_dims
        self.time_d1 = layers.Dense(r1, activation="tanh", name=f"{name}_time_d1")
        self.time_d2 = layers.Dense(r2, name=f"{name}_time_d2")
        self.time_out = layers.Dense(1, activation="sigmoid", name=f"{name}_time_out")
        self.timeliness_lambda = float(config.timeliness_lambda)

    def call(
        self,
        bucket_input,
        time_input,
        title_len: int,
        training=None,
    ):
        """Forward pass.

        Args:
            bucket_input: ``(B, T+E)`` int32.
            time_input: ``(B,)`` int32.
            title_len: ``T`` so the layer can split the entity slice.

        Returns:
            ``(B,)`` popularity score.
        """
        title_buckets = bucket_input[:, :title_len]
        entity_buckets = bucket_input[:, title_len:]
        title_emb = self.token_pop_embedding(title_buckets)
        entity_emb = self.token_pop_embedding(entity_buckets)
        title_emb = self.title_dropout1(title_emb, training=training)
        entity_emb = self.entity_dropout1(entity_emb, training=training)

        title_co = self.title_co_mhca(
            title_emb, entity_emb, entity_emb, training=training
        )
        entity_co = self.entity_co_mhca(
            entity_emb, title_emb, title_emb, training=training
        )

        title_self = self.title_mhsa(title_emb, title_emb, title_emb, training=training)
        title_seq = title_self + title_co
        title_seq = self.title_dropout2(title_seq, training=training)
        title_vec = self.title_attention(title_seq)

        entity_self = self.entity_mhsa(
            entity_emb, entity_emb, entity_emb, training=training
        )
        entity_seq = entity_self + entity_co
        entity_seq = self.entity_dropout2(entity_seq, training=training)
        entity_vec = self.entity_attention(entity_seq)

        stacked = ops.stack([title_vec, entity_vec], axis=1)
        stacked = self.fusion_dropout(stacked, training=training)
        pop_vec = self.fusion_attention(stacked)

        x = self.content_d1(pop_vec)
        x = self.content_d2(x)
        x = self.content_d3(x)
        s_p = ops.squeeze(self.content_out(x), axis=-1)

        time_emb = self.time_embedding(time_input)
        r = self.time_d1(time_emb)
        r = self.time_d2(r)
        t_prime = ops.squeeze(self.time_out(r), axis=-1)
        # Reciprocal-power timeliness factor ``t'^(-λ)`` with a small floor
        # so the reciprocal stays finite when t' approaches 0.
        t_factor = ops.power(ops.maximum(t_prime, 1e-6), -self.timeliness_lambda)

        return s_p * t_factor

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)


# ---------------------------------------------------------------------------
# Activity gater (paper recipe: Dense(128, tanh) → Dense(64) → Dense(1, sigmoid))
# ---------------------------------------------------------------------------


class TCCMActivityGater(keras.Model):
    """Per-user gate balancing relevance vs. popularity (TCCM-faithful).

    Architecture (matches ``create_pe_model`` lines 240–245)::

        Dense(128, tanh) → Dense(64) → Dense(1, sigmoid)
    """

    def __init__(self, config: TCCMConfig, name: str = "tccm_activity_gater"):
        super().__init__(name=name)
        d1, d2 = config.activity_gate_dims
        self.dense1 = layers.Dense(d1, activation="tanh", name=f"{name}_d1")
        self.dense2 = layers.Dense(d2, name=f"{name}_d2")
        self.dense_out = layers.Dense(1, activation="sigmoid", name=f"{name}_out")

    def call(self, user_vec, training=None):
        x = self.dense1(user_vec)
        x = self.dense2(x)
        return ops.squeeze(self.dense_out(x), axis=-1)
