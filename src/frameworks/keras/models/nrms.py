from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import NRMSConfig
from src.frameworks.keras.layers import AdditiveAttention, GlorotUniformMHA
from src.frameworks.keras.models.base import BaseModel


class NewsEncoder(keras.Model):
    """News encoder component for NRMS.

    Processes news titles through word embeddings, dropout, multi-head self-attention,
    and additive attention to produce news representations.
    """

    def __init__(
        self,
        config: NRMSConfig,
        embedding_layer: layers.Embedding,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers as instance attributes so they're tracked
        self.dropout1 = layers.Dropout(
            self.config.dropout_rate, seed=self.config.seed, name="embedding_dropout"
        )
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.config.multiheads,
            key_dim=self.config.head_dim,
            dropout=self.config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="title_word_self_attention",
        )
        self.dropout2 = layers.Dropout(
            self.config.dropout_rate, seed=self.config.seed, name="attention_dropout"
        )
        self.additive_attention = AdditiveAttention(
            query_vec_dim=self.config.attention_hidden_dim,
            seed=self.config.seed,
            name="title_additive_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.

        Args:
            input_shape: Input shape (batch_size, title_length)

        Returns:
            Output shape (batch_size, embedding_size)
        """
        return (input_shape[0], self.config.embedding_size)

    def call(self, inputs, training=None):
        """Forward pass for news encoding.

        Args:
            inputs: News token sequences (batch_size, title_length)
            training: Whether in training mode

        Returns:
            News representations (batch_size, embedding_size)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(inputs)

        # Dropout after embedding
        y = self.dropout1(embedded_sequences, training=training)

        # Create padding mask for attention (2D: batch_size, seq_len)
        padding_mask = ops.not_equal(inputs, 0)

        # Multi-Head Self-Attention over words
        # Note: For self-attention, we only need the key/value mask, not attention_mask
        y = self.multi_head_attention(
            y, y, y, key_mask=padding_mask, value_mask=padding_mask, training=training
        )

        # Apply dropout after self-attention
        y = self.dropout2(y, training=training)

        # Additive Attention to get single news vector
        news_representation = self.additive_attention(y, mask=padding_mask)

        return news_representation


class UserEncoder(keras.Model):
    """User encoder component for NRMS.

    Processes user history through news encoder, multi-head self-attention,
    and additive attention to produce user representations.
    """

    def __init__(
        self, config: NRMSConfig, news_encoder: NewsEncoder, name: str = "user_encoder"
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

        # Create layers as instance attributes so they're tracked
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="time_distributed_news_encoder"
        )
        self.browsed_news_attention = layers.MultiHeadAttention(
            num_heads=self.config.multiheads,
            key_dim=self.config.head_dim,
            dropout=self.config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="browsed_news_self_attention",
        )
        self.user_additive_attention = AdditiveAttention(
            query_vec_dim=self.config.attention_hidden_dim,
            seed=self.config.seed,
            name="user_additive_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the user encoder.

        Args:
            input_shape: Input shape (batch_size, history_length, title_length)

        Returns:
            Output shape (batch_size, embedding_size)
        """
        return (input_shape[0], self.config.embedding_size)

    def call(self, inputs, training=None):
        """Forward pass for user encoding.

        Args:
            inputs: User history token sequences (batch_size, history_length, title_length)
            training: Whether in training mode

        Returns:
            User representations (batch_size, embedding_size)
        """
        # Apply news encoder to each news item in history
        click_title_presents = self.time_distributed(inputs, training=training)

        # Create mask for history items (2D: batch_size, history_len)
        history_mask = ops.any(ops.not_equal(inputs, 0), axis=-1)

        # Self-Attention over browsed news
        # Note: For self-attention, we only need the key/value mask, not attention_mask
        y = self.browsed_news_attention(
            click_title_presents,
            click_title_presents,
            click_title_presents,
            key_mask=history_mask,
            value_mask=history_mask,
            training=training,
        )

        # Additive Attention to get single user vector
        user_representation = self.user_additive_attention(y, mask=history_mask)

        return user_representation


class NRMSScorer(keras.Model):
    """Scoring component for NRMS.

    Handles different scoring scenarios: training (multiple candidates with softmax),
    single candidate scoring (with sigmoid), and multiple candidate scoring (raw scores).
    """

    def __init__(
        self,
        config: NRMSConfig,
        news_encoder: NewsEncoder,
        user_encoder: UserEncoder,
        name: str = "nrms_scorer",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

        # Store TimeDistributed layers for candidate processing
        self.candidate_encoder_train = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_candidates"
        )

    def build(self, input_shape):
        super().build(input_shape)

    def score_training_batch(self, history_tokens, candidate_tokens, training=None):
        """Score a training batch (raw logits — loss handles softmax)."""
        user_repr = self.user_encoder(history_tokens, training=training)

        # Process candidates using stored TimeDistributed layer
        candidate_repr = self.candidate_encoder_train(
            candidate_tokens, training=training
        )

        # Calculate scores using dot product (raw logits)
        scores = layers.Dot(axes=-1, name="dot_product_train")(
            [candidate_repr, user_repr]
        )

        return scores



class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model.

    Refactored with clean architecture using separate components for better
    maintainability, testability, and code organization.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NRMSConfig | None = None,
        name: str = "nrms",
        **config_overrides,
    ):
        """Build an NRMS model.

        Args:
            processed_news: Dataset's processed news dict (vocab_size,
                embeddings, num_categories, ...).
            config: Optional pre-built NRMSConfig. If ``None``, one is
                constructed from ``config_overrides``.
            name: Keras layer name.
            **config_overrides: Field overrides forwarded to NRMSConfig
                when ``config`` is None. Used by ``build_model_from_spec``
                which dumps every config field as a kwarg.
        """
        super().__init__(name=name)

        if config is None:
            config = NRMSConfig(**config_overrides)
        self.config = config

        # Store processed news data and validate
        self.processed_news = processed_news
        self._validate_processed_news()

        # BaseModel contract — set at __init__ time so build() can rely on it
        self.process_user_id = config.process_user_id

        # Components are created in build()
        self.embedding_layer = None
        self.news_encoder = None
        self.user_encoder = None
        self.scorer = None

        # Build the model immediately with dummy input shape
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
        """Create all model components."""
        # Create shared embedding layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(
                self.processed_news["embeddings"]
            ),
            trainable=True,
            name="word_embedding",
        )

        # Create component encoders
        self.news_encoder = NewsEncoder(self.config, self.embedding_layer)
        self.user_encoder = UserEncoder(self.config, self.news_encoder)
        self.scorer = NRMSScorer(self.config, self.news_encoder, self.user_encoder)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass for training. Returns raw logits (B, C).

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        return self.scorer.score_training_batch(
            inputs["hist_tokens"], inputs["cand_tokens"], training=training
        )

    def get_config(self):
        """Returns the configuration of the NRMS model for serialization."""
        base_config = super().get_config()
        base_config.update(
            {
                "embedding_size": self.config.embedding_size,
                "multiheads": self.config.multiheads,
                "head_dim": self.config.head_dim,
                "attention_hidden_dim": self.config.attention_hidden_dim,
                "dropout_rate": self.config.dropout_rate,
                "seed": self.config.seed,
                "max_title_length": self.config.max_title_length,
                "max_history_length": self.config.max_history_length,
                "max_impressions_length": self.config.max_impressions_length,
                "process_user_id": self.config.process_user_id,
            }
        )
        return base_config
