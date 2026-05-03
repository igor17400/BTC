"""CAUM (Candidate-Aware User Modeling) -- Keras.

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
"precompute user vectors → dot product" evaluation cannot be used.
CAUM uses a custom evaluator that precomputes news vectors and then
scores each impression by running the inter-model per candidate.
"""

from __future__ import annotations

from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import CAUMConfig
from src.frameworks.keras.layers import AdditiveAttention, GlorotUniformMHA
from src.frameworks.keras.models.base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(keras.Model):
    """Encode a news article from its title token sequence.

    Pipeline: Embedding → Dropout → MultiHeadSelfAttention → Dropout
    → AdditiveAttention → Dense(news_dim) → news vector.
    """

    def __init__(
        self,
        config: CAUMConfig,
        embedding_layer: layers.Embedding,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        self.dropout1 = layers.Dropout(
            config.dropout_rate, seed=config.seed, name="embedding_dropout"
        )
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=config.news_num_heads,
            key_dim=config.news_head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="title_word_self_attention",
        )
        self.dropout2 = layers.Dropout(
            config.dropout_rate, seed=config.seed, name="attention_dropout"
        )
        self.additive_attention = AdditiveAttention(
            query_vec_dim=config.news_attention_hidden_dim,
            seed=config.seed,
            name="title_additive_attention",
        )
        self.projection = layers.Dense(config.news_dim, name="news_projection")

    def call(self, inputs, training=None):
        """inputs: (batch, title_length) → (batch, news_dim)."""
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded, training=training)

        padding_mask = ops.not_equal(inputs, 0)
        y = self.multi_head_attention(
            y,
            y,
            y,
            key_mask=padding_mask,
            value_mask=padding_mask,
            training=training,
        )
        y = self.dropout2(y, training=training)

        news_vec = self.additive_attention(y, mask=padding_mask)
        return self.projection(news_vec)


# ---------------------------------------------------------------------------
# Candidate-aware user interaction model (inter_model)
# ---------------------------------------------------------------------------


class DenseAttentionScorer(layers.Layer):
    """Two-layer MLP that scores a concatenated [click, candidate] pair.

    Used inside Candi-Att (TimeDistributed over the click sequence).
    """

    def __init__(self, hidden_dim: int, mid_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation="tanh", name="att_dense1")
        self.dense2 = layers.Dense(mid_dim, activation="tanh", name="att_dense2")
        self.dense3 = layers.Dense(1, name="att_score")

    def call(self, inputs):
        """inputs: (batch, 2*news_dim) → (batch, 1)."""
        return self.dense3(self.dense2(self.dense1(inputs)))


class InterModel(keras.Model):
    """Candidate-aware user interest model.

    Takes pre-encoded clicked news vectors and a single candidate news
    vector, and produces a scalar matching score.

    Architecture (following the reference code):
        1. Candi-CNN: circular-shift window over clicks + candidate → Dense
        2. Candi-SelfAtt: concat candidate to each click → Dense → MHSA
        3. Fusion: concat [cnn, selfatt] → Dropout → Dense
        4. Candi-Att: concat [fused, candidate] → DenseAttScorer → softmax
           → weighted sum → user_vec
        5. Score: dot(user_vec, dropout(candidate))
    """

    def __init__(self, config: CAUMConfig, name: str = "inter_model"):
        super().__init__(name=name)
        D = config.news_dim
        self.config = config

        self.dropout_cand = layers.Dropout(
            config.dropout_rate, seed=config.seed, name="cand_dropout"
        )
        self.dropout_clicks = layers.Dropout(
            config.dropout_rate, seed=config.seed, name="clicks_dropout"
        )

        # Candi-CNN: projects [left, center, right, candidate] → D
        self.cnn_projection = layers.Dense(D, name="candi_cnn_proj")

        # Candi-SelfAtt: projects [candidate, click] → D, then MHSA
        self.selfatt_input_projection = layers.Dense(D, name="candi_selfatt_input_proj")
        self.selfatt_mha = layers.MultiHeadAttention(
            num_heads=config.candi_selfatt_num_heads,
            key_dim=config.candi_selfatt_head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="candi_selfatt_mha",
        )

        # Fusion
        self.fusion_dropout = layers.Dropout(
            config.dropout_rate, seed=config.seed, name="fusion_dropout"
        )
        self.fusion_projection = layers.Dense(D, name="fusion_proj")

        # Candi-Att (DNN scorer applied per click)
        self.dense_att = DenseAttentionScorer(
            hidden_dim=config.candi_att_hidden_dim,
            mid_dim=config.candi_att_mid_dim,
            name="dense_att_scorer",
        )

    def call(self, inputs, training=None):
        """Score a candidate against clicked news.

        Args:
            inputs: list of [cand_vec, clicked_vecs]
                cand_vec: (B, D) single candidate news vector.
                clicked_vecs: (B, H, D) encoded clicked news vectors.

        Returns:
            (B,) scalar matching scores.
        """
        cand_vec, clicked_vecs = inputs
        D = self.config.news_dim
        H = ops.shape(clicked_vecs)[1]

        can_vec_dropped = self.dropout_cand(cand_vec, training=training)
        user_vecs = self.dropout_clicks(clicked_vecs, training=training)

        # Repeat candidate across history length: (B, H, D)
        cand_repeated = ops.repeat(ops.expand_dims(cand_vec, axis=1), H, axis=1)

        # ----- Candi-CNN -----
        # Circular shift: left by 1, right by 1
        left = ops.concatenate([user_vecs[:, -1:, :], user_vecs[:, :-1, :]], axis=1)
        right = ops.concatenate([user_vecs[:, 1:, :], user_vecs[:, :1, :]], axis=1)
        # Concat [left, center, right, candidate] → Dense
        cnn_input = ops.concatenate(
            [left, user_vecs, right, cand_repeated], axis=-1
        )  # (B, H, 4*D)
        cnn_out = self.cnn_projection(cnn_input)  # (B, H, D)

        # ----- Candi-SelfAtt -----
        # Concat candidate feature to each click → project → MHSA
        selfatt_input = ops.concatenate(
            [cand_repeated, user_vecs], axis=-1
        )  # (B, H, 2*D)
        selfatt_input = self.selfatt_input_projection(selfatt_input)  # (B, H, D)

        # Compute history mask for MHSA (True = valid position)
        # clicked_vecs is already encoded — check for all-zero vectors
        history_mask = ops.any(ops.not_equal(clicked_vecs, 0.0), axis=-1)  # (B, H)

        selfatt_out = self.selfatt_mha(
            selfatt_input,
            selfatt_input,
            selfatt_input,
            key_mask=history_mask,
            value_mask=history_mask,
            training=training,
        )  # (B, H, D)

        # ----- Fusion -----
        fused = ops.concatenate([cnn_out, selfatt_out], axis=-1)  # (B, H, 2*D)
        fused = self.fusion_dropout(fused, training=training)
        fused = self.fusion_projection(fused)  # (B, H, D)

        # ----- Candi-Att -----
        # Concat [fused, candidate] for each click → DNN scorer
        att_input = ops.concatenate([fused, cand_repeated], axis=-1)  # (B, H, 2*D)

        # Apply scorer per time step (manual TimeDistributed)
        B = ops.shape(att_input)[0]
        flat_att = ops.reshape(att_input, (B * H, 2 * D))
        flat_scores = self.dense_att(flat_att)  # (B*H, 1)
        att_scores = ops.reshape(flat_scores, (B, H))  # (B, H)

        # Mask out padding positions before softmax
        att_scores = ops.where(
            history_mask, att_scores, ops.full_like(att_scores, -1e9)
        )
        att_weights = keras.activations.softmax(att_scores, axis=-1)  # (B, H)

        # Weighted sum → user vector
        user_vec = ops.sum(
            fused * ops.expand_dims(att_weights, axis=-1), axis=1
        )  # (B, D)

        # Score: dot(user_vec, candidate)
        score = ops.sum(user_vec * can_vec_dropped, axis=-1)  # (B,)

        return score


# ---------------------------------------------------------------------------
# Dummy user encoder (satisfies BaseModel contract)
# ---------------------------------------------------------------------------


class UserEncoder(keras.Model):
    """Fallback user encoder for BaseModel contract compatibility.

    CAUM's actual user modeling is candidate-aware and lives in
    InterModel.  This encoder simply mean-pools the news encoder
    output over history, used only if the default evaluator is
    called (which should not happen — CAUM uses a custom evaluator).
    """

    def __init__(
        self,
        config: CAUMConfig,
        news_encoder: NewsEncoder,
        name: str = "user_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

    def call(self, inputs, training=None):
        """inputs: (B, H, T) → (B, news_dim)."""
        B = ops.shape(inputs)[0]
        H = ops.shape(inputs)[1]
        T = ops.shape(inputs)[2]

        flat = ops.reshape(inputs, (B * H, T))
        flat_vecs = self.news_encoder(flat, training=training)
        news_embeds = ops.reshape(flat_vecs, (B, H, -1))

        # Mean pool over valid positions
        mask = ops.any(ops.not_equal(inputs, 0), axis=-1)  # (B, H)
        mask_f = ops.cast(mask, dtype=news_embeds.dtype)
        count = ops.maximum(ops.sum(mask_f, axis=-1, keepdims=True), 1.0)
        return ops.sum(news_embeds * ops.expand_dims(mask_f, axis=-1), axis=1) / count


# ---------------------------------------------------------------------------
# Full CAUM model
# ---------------------------------------------------------------------------


class CAUM(BaseModel):
    """News Recommendation with Candidate-aware User Modeling.

    During training, ``call()`` encodes all clicked news once, then
    loops over each candidate to score it via the candidate-aware
    ``inter_model``.

    During evaluation, a custom evaluator precomputes news vectors and
    scores each impression by running the inter_model per candidate.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CAUMConfig | None = None,
        name: str = "caum",
        **config_overrides,
    ):
        super().__init__(name=name)

        if config is None:
            config = CAUMConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        self.embedding_layer = None
        self.news_encoder = None
        self.user_encoder = None
        self.inter_model = None

        dummy_input_shape = {
            "hist_tokens": (
                None,
                config.max_history_length,
                config.max_title_length,
            ),
            "cand_tokens": (
                None,
                config.max_impressions_length,
                config.max_title_length,
            ),
        }
        self.build(dummy_input_shape)

    def build(self, input_shape) -> None:
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(
                self.processed_news["embeddings"]
            ),
            trainable=True,
            name="word_embedding",
        )

        self.news_encoder = NewsEncoder(self.config, self.embedding_layer)
        self.user_encoder = UserEncoder(self.config, self.news_encoder)
        self.inter_model = InterModel(self.config)

        # Force-build sub-models with concrete shapes so kernel initializers
        # run eagerly (required for JAX tracing).
        import numpy as np

        T = self.config.max_title_length
        H = self.config.max_history_length
        D = self.config.news_dim

        dummy_tokens = np.zeros((1, T), dtype="int32")
        self.news_encoder(dummy_tokens, training=False)

        dummy_hist = np.zeros((1, H, T), dtype="int32")
        self.user_encoder(dummy_hist, training=False)

        dummy_cand = np.zeros((1, D), dtype="float32")
        dummy_clicked = np.zeros((1, H, D), dtype="float32")
        self.inter_model([dummy_cand, dummy_clicked], training=False)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Training forward pass.

        Encodes clicked history once, then scores each candidate
        through the candidate-aware inter_model.

        Returns:
            (B, C) logit scores.
        """
        hist_tokens = inputs["hist_tokens"]
        cand_tokens = inputs["cand_tokens"]

        B = ops.shape(hist_tokens)[0]
        H = ops.shape(hist_tokens)[1]
        T_h = ops.shape(hist_tokens)[2]

        # Encode clicked news: (B*H, T) → (B, H, D)
        flat_hist = ops.reshape(hist_tokens, (B * H, T_h))
        clicked_vecs = ops.reshape(
            self.news_encoder(flat_hist, training=training), (B, H, -1)
        )

        # Encode candidate news: (B*C, T) → (B, C, D)
        C = ops.shape(cand_tokens)[1]
        T_c = ops.shape(cand_tokens)[2]
        flat_cand = ops.reshape(cand_tokens, (B * C, T_c))
        cand_vecs = ops.reshape(
            self.news_encoder(flat_cand, training=training), (B, C, -1)
        )

        # Score each candidate via inter_model
        scores = []
        for i in range(self.config.max_impressions_length):
            cand_i = cand_vecs[:, i, :]  # (B, D)
            score_i = self.inter_model(
                [cand_i, clicked_vecs], training=training
            )  # (B,)
            scores.append(score_i)

        return ops.stack(scores, axis=1)  # (B, C)

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "embedding_size": self.config.embedding_size,
                "news_dim": self.config.news_dim,
                "news_num_heads": self.config.news_num_heads,
                "news_head_dim": self.config.news_head_dim,
                "news_attention_hidden_dim": self.config.news_attention_hidden_dim,
                "candi_selfatt_num_heads": self.config.candi_selfatt_num_heads,
                "candi_selfatt_head_dim": self.config.candi_selfatt_head_dim,
                "candi_cnn_half_window": self.config.candi_cnn_half_window,
                "candi_att_hidden_dim": self.config.candi_att_hidden_dim,
                "candi_att_mid_dim": self.config.candi_att_mid_dim,
                "dropout_rate": self.config.dropout_rate,
            }
        )
        return base_config
