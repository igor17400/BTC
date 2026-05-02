"""MINER (Multi-Interest Matching Network) -- Keras.

Reference: Li et al., "MINER: Multi-Interest Matching Network for News
Recommendation", Findings of ACL 2022.
"""

from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import MINERConfig
from src.frameworks.keras.layers import AdditiveAttention, GlorotUniformMHA
from src.frameworks.keras.models.base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(keras.Model):
    """Encode a news article from its title token sequence."""

    def __init__(
        self,
        config: MINERConfig,
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
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="title_word_self_attention",
        )
        self.dropout2 = layers.Dropout(
            config.dropout_rate, seed=config.seed, name="attention_dropout"
        )
        self.additive_attention = AdditiveAttention(
            query_vec_dim=config.attention_hidden_dim,
            seed=config.seed,
            name="title_additive_attention",
        )

    def call(self, inputs, training=None):
        """inputs: (batch, title_length) -> (batch, embedding_size)."""
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded, training=training)

        padding_mask = ops.not_equal(inputs, 0)
        y = self.multi_head_attention(
            y, y, y, key_mask=padding_mask, value_mask=padding_mask,
            training=training,
        )
        y = self.dropout2(y, training=training)

        return self.additive_attention(y, mask=padding_mask)


# ---------------------------------------------------------------------------
# Poly Attention
# ---------------------------------------------------------------------------


class PolyAttention(layers.Layer):
    """Extract K interest vectors from clicked news via learnable context codes."""

    def __init__(self, config: MINERConfig, **kwargs):
        super().__init__(**kwargs)
        self.K = config.num_interest_vectors
        self.D = config.context_code_dim
        self.E = config.embedding_size

    def build(self, input_shape):
        self.context_codes = self.add_weight(
            name="context_codes",
            shape=(self.K, self.D),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
        )
        # Scale by tanh gain (~1.667) to match reference
        self.context_codes.assign(self.context_codes * (5.0 / 3.0))

        self.W_h = self.add_weight(
            name="W_h",
            shape=(self.E, self.D),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, news_embeddings, mask=None):
        """news_embeddings: (B, M, E), mask: (B, M) bool True=keep -> (B, K, E)."""
        projected = ops.tanh(ops.matmul(news_embeddings, self.W_h))  # (B, M, D)

        logits = ops.einsum("bmd,kd->bkm", projected, self.context_codes)

        if mask is not None:
            logits = ops.where(
                ops.expand_dims(mask, axis=1), logits, ops.full_like(logits, -1e9)
            )

        weights = keras.activations.softmax(logits, axis=-1)  # (B, K, M)

        return ops.einsum("bkm,bme->bke", weights, news_embeddings)


# ---------------------------------------------------------------------------
# User encoder
# ---------------------------------------------------------------------------


class UserEncoder(keras.Model):
    """Encode user history into interest vectors.

    Returns mean of K interest vectors for eval compatibility.
    Use ``encode_interests()`` for the full K vectors.
    """

    def __init__(
        self,
        config: MINERConfig,
        news_encoder: NewsEncoder,
        name: str = "user_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.poly_attention = PolyAttention(config, name="poly_attention")

    def encode_interests(self, inputs, training=None):
        """Encode history into K interest vectors. Returns (B, K, E)."""
        B = ops.shape(inputs)[0]
        H = ops.shape(inputs)[1]
        T = ops.shape(inputs)[2]

        flat = ops.reshape(inputs, (B * H, T))
        flat_vecs = self.news_encoder(flat, training=training)
        news_embeds = ops.reshape(flat_vecs, (B, H, -1))

        mask = ops.any(ops.not_equal(inputs, 0), axis=-1)  # (B, H)

        return self.poly_attention(news_embeds, mask=mask)

    def call(self, inputs, training=None):
        """Encode history into a single user vector (mean of K interests)."""
        interest_vecs = self.encode_interests(inputs, training=training)
        return ops.mean(interest_vecs, axis=1)


# ---------------------------------------------------------------------------
# Full MINER model
# ---------------------------------------------------------------------------


class MINER(BaseModel):
    """Multi-Interest Matching Network for News Recommendation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: MINERConfig | None = None,
        name: str = "miner",
        **config_overrides,
    ):
        super().__init__(name=name)

        if config is None:
            config = MINERConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        self.embedding_layer = None
        self.news_encoder = None
        self.user_encoder = None
        self.W_e = None

        dummy_input_shape = {
            "hist_tokens": (None, config.max_history_length, config.max_title_length),
            "cand_tokens": (None, config.max_impressions_length, config.max_title_length),
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

        # Target-aware attention: projects INTERESTS
        self.W_e = layers.Dense(
            self.config.embedding_size, name="target_aware_projection"
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """MINER-weighted training scoring."""
        hist_tokens = inputs["hist_tokens"]
        cand_tokens = inputs["cand_tokens"]

        # Get K interest vectors
        interest_vecs = self.user_encoder.encode_interests(
            hist_tokens, training=training
        )  # (B, K, E)

        # Encode candidates
        B = ops.shape(cand_tokens)[0]
        C = ops.shape(cand_tokens)[1]
        T = ops.shape(cand_tokens)[2]
        flat_cands = ops.reshape(cand_tokens, (B * C, T))
        cand_embeds = ops.reshape(
            self.news_encoder(flat_cands, training=training), (B, C, -1)
        )  # (B, C, E)

        # Per-interest scores: (B, K, C)
        interest_scores = ops.einsum("bke,bce->bkc", interest_vecs, cand_embeds)

        # Target-aware attention: project INTERESTS
        projected_interests = keras.activations.gelu(
            self.W_e(interest_vecs)
        )  # (B, K, E)

        # Each candidate attends over K projected interests: (B, C, K)
        agg_logits = ops.einsum("bce,bke->bck", cand_embeds, projected_interests)
        agg_weights = keras.activations.softmax(agg_logits, axis=2)

        # Aggregate: (B, C)
        scores = ops.sum(
            agg_weights * ops.transpose(interest_scores, (0, 2, 1)), axis=2
        )

        return scores

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "embedding_size": self.config.embedding_size,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "attention_hidden_dim": self.config.attention_hidden_dim,
            "num_interest_vectors": self.config.num_interest_vectors,
            "context_code_dim": self.config.context_code_dim,
            "dropout_rate": self.config.dropout_rate,
        })
        return base_config
