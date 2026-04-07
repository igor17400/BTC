from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import NRMSConfig
from src.frameworks.keras.layers import AdditiveAttention
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
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
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
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
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
        self.candidate_encoder_eval = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_eval"
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

    def score_single(self, history_tokens, candidate_tokens, training=None):
        """Score a single candidate with sigmoid."""
        user_repr = self.user_encoder(history_tokens, training=training)
        candidate_repr = self.news_encoder(candidate_tokens, training=training)

        # Calculate score using dot product
        score = layers.Dot(axes=-1, name="dot_product_single")(
            [candidate_repr, user_repr]
        )

        # Apply sigmoid for probability
        return layers.Activation("sigmoid", name="sigmoid_activation")(score)

    def score_candidates(self, history_tokens, candidate_tokens, training=False):
        """Score multiple candidates with sigmoid (evaluation)."""
        # Get user representations for all items in batch
        user_repr = self.user_encoder(
            history_tokens, training=training
        )  # (batch_size, embedding_dim)

        # Process all candidates using TimeDistributed layer
        candidate_repr = self.candidate_encoder_eval(
            candidate_tokens, training=training
        )  # (batch_size, num_candidates, embedding_dim)

        # Vectorized dot product: expand user_repr to match candidate dimensions
        user_repr_expanded = ops.expand_dims(
            user_repr, axis=1
        )  # (batch_size, 1, embedding_dim)

        # Calculate scores using vectorized dot product
        scores = ops.sum(
            candidate_repr * user_repr_expanded, axis=-1
        )  # (batch_size, num_candidates)

        # Apply sigmoid activation for consistency with single candidate scoring
        return ops.sigmoid(scores)


class NRMS(BaseModel):
    """Neural News Recommendation with Multi-Head Self-Attention (NRMS) model.

    Refactored with clean architecture using separate components for better
    maintainability, testability, and code organization.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        embedding_size: int = 300,
        multiheads: int = 16,
        head_dim: int = 16,
        attention_hidden_dim: int = 200,
        dropout_rate: float = 0.2,
        seed: int = 42,
        max_title_length: int = 50,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        process_user_id: bool = False,
        name: str = "nrms",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration object
        self.config = NRMSConfig(
            embedding_size=embedding_size,
            multiheads=multiheads,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            dropout_rate=dropout_rate,
            seed=seed,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            process_user_id=process_user_id,
        )

        # Store processed news data
        self.processed_news = processed_news
        self._validate_processed_news()

        # Set BaseModel attributes for fast evaluation (required by BaseModel)
        self.process_user_id = process_user_id

        # Initialize model components (will be created in build)
        self.embedding_layer = None
        self.news_encoder = None
        self.user_encoder = None
        self.scorer = None
        self.training_model = None
        self.scorer_model = None

        # Build the model immediately with dummy input shape
        dummy_input_shape = {
            "hist_tokens": (None, max_history_length, max_title_length),
            "cand_tokens": (None, max_impressions_length, max_title_length),
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

        # Build training and scorer models for compatibility
        self.training_model, self.scorer_model = self._build_compatibility_models()

        super().build(input_shape)

    def _build_compatibility_models(self) -> tuple[keras.Model, keras.Model]:
        """Build training and scorer models for backward compatibility."""
        # ----- Training model -----
        history_input = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length),
            dtype="int32",
            name="hist_tokens",
        )
        candidates_input = keras.Input(
            shape=(self.config.max_impressions_length, self.config.max_title_length),
            dtype="int32",
            name="cand_tokens",
        )

        training_output = self.scorer.score_training_batch(
            history_input, candidates_input
        )
        training_model = keras.Model(
            inputs=[history_input, candidates_input],
            outputs=training_output,
            name="nrms_training_model",
        )

        # ----- Scorer model -----
        history_input_score = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length),
            dtype="int32",
            name="history_tokens_score",
        )
        single_candidate_input = keras.Input(
            shape=(self.config.max_title_length,),
            dtype="int32",
            name="single_candidate_tokens_score",
        )

        scorer_output = self.scorer.score_single(
            history_input_score, single_candidate_input
        )
        scorer_model = keras.Model(
            inputs=[history_input_score, single_candidate_input],
            outputs=scorer_output,
            name="nrms_scorer_model",
        )

        return training_model, scorer_model

    def _build_newsencoder(self) -> keras.Model:
        """Legacy method for backward compatibility - returns news encoder."""
        return self.news_encoder

    def _build_userencoder(self) -> keras.Model:
        """Legacy method for backward compatibility - returns user encoder."""
        return self.user_encoder

    def _build_graph_models(self) -> tuple[keras.Model, keras.Model]:
        """Legacy method for backward compatibility - returns training and scorer models."""
        return self.training_model, self.scorer_model

    def _validate_inputs(self, inputs: dict, training: bool = None) -> None:
        """Validate input format and shapes based on mode."""
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary")

        input_keys = set(inputs.keys())

        if training:
            # Training mode expects hist_tokens and cand_tokens
            required_keys = {"hist_tokens", "cand_tokens"}
            if not required_keys.issubset(input_keys):
                raise ValueError(
                    f"Training mode requires keys: {required_keys}, "
                    f"but got keys: {list(input_keys)}"
                )
        else:
            # Inference mode can have different formats
            valid_combinations = [
                {"hist_tokens", "cand_tokens"},  # Training format for validation
                {
                    "history_tokens",
                    "single_candidate_tokens",
                },  # Single candidate scoring
            ]

            if not any(
                combination.issubset(input_keys) for combination in valid_combinations
            ):
                raise ValueError(
                    f"Inference mode expects one of: {valid_combinations}, "
                    f"but got keys: {list(input_keys)}"
                )

    def call(self, inputs, training=None):
        """Main forward pass — dispatches to scoring methods.

        Args:
            inputs: Dictionary with input tensors.
                Training:  ``hist_tokens``, ``cand_tokens``
                Inference: ``hist_tokens``, ``cand_tokens`` (multi) or
                           ``history_tokens``, ``single_candidate_tokens`` (single)
            training: Whether in training mode.

        Returns:
            Scores tensor.
        """
        if training:
            return self.score_training_batch(inputs)
        elif "single_candidate_tokens" in inputs:
            return self.score_single(inputs)
        else:
            return self.score_candidates(inputs)

    # ----- scoring helpers (delegate to scorer) ---------------------------

    def score_training_batch(self, inputs, training=None):
        """Score a training batch with softmax over candidates."""
        return self.scorer.score_training_batch(
            inputs["hist_tokens"], inputs["cand_tokens"], training=training
        )

    def score_single(self, inputs, training=None):
        """Score a single candidate with sigmoid."""
        return self.scorer.score_single(
            inputs["history_tokens"],
            inputs["single_candidate_tokens"],
            training=training,
        )

    def score_candidates(self, inputs, training=False):
        """Score multiple candidates with sigmoid (evaluation)."""
        return self.scorer.score_candidates(
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
