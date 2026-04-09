from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import NAMLConfig
from src.frameworks.keras.layers import AdditiveAttention
from src.frameworks.keras.models.base import BaseModel


class TitleEncoder(keras.Model):
    """Title encoder component for NAML.

    Processes news titles through word embeddings, CNN, and additive attention.
    """

    def __init__(
        self,
        config: NAMLConfig,
        embedding_layer: layers.Embedding,
        name: str = "title_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers
        self.dropout1 = layers.Dropout(
            self.config.dropout_rate,
            seed=self.config.seed,
            name="title_embedding_dropout",
        )
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.activation,
            padding="same",
            name="title_cnn",
        )
        self.dropout2 = layers.Dropout(
            self.config.dropout_rate, seed=self.config.seed, name="title_cnn_dropout"
        )
        self.additive_attention = AdditiveAttention(
            self.config.word_attention_query_dim,
            seed=self.config.seed,
            name="title_word_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the title encoder.

        Args:
            input_shape: Input shape (batch_size, title_length)

        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for title encoding.

        Args:
            inputs: Title token sequences (batch_size, title_length)
            training: Whether in training mode

        Returns:
            Title representations (batch_size, cnn_filter_num)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(inputs)

        # Dropout after embedding
        y = self.dropout1(embedded_sequences, training=training)

        # CNN
        y = self.cnn(y)

        # Dropout after CNN
        y = self.dropout2(y, training=training)

        # Create padding mask for attention
        padding_mask = ops.not_equal(inputs, 0)

        # Additive Attention to get single title vector
        title_representation = self.additive_attention(y, mask=padding_mask)

        return title_representation


class AbstractEncoder(keras.Model):
    """Abstract encoder component for NAML.

    Processes news abstracts through word embeddings, CNN, and additive attention.
    """

    def __init__(
        self,
        config: NAMLConfig,
        embedding_layer: layers.Embedding,
        name: str = "abstract_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers
        self.dropout1 = layers.Dropout(
            self.config.dropout_rate,
            seed=self.config.seed,
            name="abstract_embedding_dropout",
        )
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.activation,
            padding="same",
            name="abstract_cnn",
        )
        self.dropout2 = layers.Dropout(
            self.config.dropout_rate, seed=self.config.seed, name="abstract_cnn_dropout"
        )
        self.additive_attention = AdditiveAttention(
            self.config.word_attention_query_dim,
            seed=self.config.seed,
            name="abstract_word_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the abstract encoder.

        Args:
            input_shape: Input shape (batch_size, abstract_length)

        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for abstract encoding.

        Args:
            inputs: Abstract token sequences (batch_size, abstract_length)
            training: Whether in training mode

        Returns:
            Abstract representations (batch_size, cnn_filter_num)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(inputs)

        # Dropout after embedding
        y = self.dropout1(embedded_sequences, training=training)

        # CNN
        y = self.cnn(y)

        # Dropout after CNN
        y = self.dropout2(y, training=training)

        # Create padding mask for attention
        padding_mask = ops.not_equal(inputs, 0)

        # Additive Attention to get single abstract vector
        abstract_representation = self.additive_attention(y, mask=padding_mask)

        return abstract_representation


class CategoryEncoder(keras.Model):
    """Category encoder component for NAML.

    Embeds and projects category IDs into the same dimension as CNN outputs.
    """

    def __init__(
        self, config: NAMLConfig, num_categories: int, name: str = "category_encoder"
    ):
        super().__init__(name=name)
        self.config = config
        self.num_categories = num_categories

        # Create layers
        self.embedding = layers.Embedding(
            self.num_categories + 1,
            self.config.category_embedding_dim,
            trainable=True,
            name="category_embedding",
        )
        self.projection = layers.Dense(
            self.config.cnn_filter_num,
            activation=self.config.activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.GlorotUniform(),
            name="category_projection",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the category encoder.

        Args:
            input_shape: Input shape (batch_size, 1)

        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for category encoding.

        Args:
            inputs: Category IDs (batch_size, 1)
            training: Whether in training mode

        Returns:
            Category representations (batch_size, cnn_filter_num)
        """
        # Embedding
        embedded = self.embedding(inputs)

        # Projection to cnn_filter_num dimensions
        projected = self.projection(embedded)

        # Reshape from (batch_size, 1, cnn_filter_num) to (batch_size, cnn_filter_num)
        category_representation = ops.squeeze(projected, axis=1)

        return category_representation


class SubcategoryEncoder(keras.Model):
    """Subcategory encoder component for NAML.

    Embeds and projects subcategory IDs into the same dimension as CNN outputs.
    """

    def __init__(
        self,
        config: NAMLConfig,
        num_subcategories: int,
        name: str = "subcategory_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.num_subcategories = num_subcategories

        # Create layers
        self.embedding = layers.Embedding(
            self.num_subcategories + 1,
            self.config.subcategory_embedding_dim,
            trainable=True,
            name="subcategory_embedding",
        )
        self.projection = layers.Dense(
            self.config.cnn_filter_num,
            activation=self.config.activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.GlorotUniform(),
            name="subcategory_projection",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the subcategory encoder.

        Args:
            input_shape: Input shape (batch_size, 1)

        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for subcategory encoding.

        Args:
            inputs: Subcategory IDs (batch_size, 1)
            training: Whether in training mode

        Returns:
            Subcategory representations (batch_size, cnn_filter_num)
        """
        # Embedding
        embedded = self.embedding(inputs)

        # Projection to cnn_filter_num dimensions
        projected = self.projection(embedded)

        # Reshape from (batch_size, 1, cnn_filter_num) to (batch_size, cnn_filter_num)
        subcategory_representation = ops.squeeze(projected, axis=1)

        return subcategory_representation


class NewsEncoder(keras.Model):
    """News encoder component for NAML.

    Combines multiple views (title, abstract, category, subcategory) of a news article
    using view-level attention to produce a unified news representation.
    """

    def __init__(
        self,
        config: NAMLConfig,
        title_encoder: TitleEncoder,
        abstract_encoder: AbstractEncoder,
        category_encoder: CategoryEncoder,
        subcategory_encoder: SubcategoryEncoder,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        # View-level attention
        self.view_attention = AdditiveAttention(
            self.config.view_attention_query_dim,
            seed=self.config.seed,
            name="view_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.

        Args:
            input_shape: Tuple representing concatenated input shape

        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for news encoding.

        Args:
            inputs: Concatenated tensor (batch_size, title_length + abstract_length + 2)
                   containing [title_tokens, abstract_tokens, category_id, subcategory_id]
            training: Whether in training mode

        Returns:
            News representations (batch_size, cnn_filter_num)
        """
        # Split the concatenated input: [title, abstract, category, subcategory]
        title_tokens = inputs[:, : self.config.max_title_length]
        abstract_tokens = inputs[
            :,
            self.config.max_title_length : self.config.max_title_length
            + self.config.max_abstract_length,
        ]
        category_id = inputs[
            :,
            self.config.max_title_length
            + self.config.max_abstract_length : self.config.max_title_length
            + self.config.max_abstract_length
            + 1,
        ]
        subcategory_id = inputs[
            :, self.config.max_title_length + self.config.max_abstract_length + 1 :
        ]

        # Encode each view
        title_vec = self.title_encoder(title_tokens, training=training)
        abstract_vec = self.abstract_encoder(abstract_tokens, training=training)
        category_vec = self.category_encoder(category_id, training=training)
        subcategory_vec = self.subcategory_encoder(subcategory_id, training=training)

        # Stack views for attention (batch_size, 4, cnn_filter_num)
        views = ops.stack(
            [title_vec, abstract_vec, category_vec, subcategory_vec], axis=1
        )

        # Apply view-level attention to combine views
        news_representation = self.view_attention(views)

        return news_representation


class UserEncoder(keras.Model):
    """User encoder component for NAML.

    Processes user browsing history through news encoder and additive attention
    to produce user representations.
    """

    def __init__(
        self, config: NAMLConfig, news_encoder: NewsEncoder, name: str = "user_encoder"
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

        # TimeDistributed layer for processing history
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_user"
        )

        # User-level attention
        self.user_attention = AdditiveAttention(
            self.config.user_attention_query_dim,
            seed=self.config.seed,
            name="user_additive_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the user encoder.

        Args:
            input_shape: Tuple representing concatenated input shape

        Returns:
            Output shape (batch_size, cnn_filter_num)
        """
        return (input_shape[0], self.config.cnn_filter_num)

    def call(self, inputs, training=None):
        """Forward pass for user encoding.

        Args:
            inputs: Concatenated tensor (batch_size, history_length, title_length + abstract_length + 2)
                   containing user's browsing history
            training: Whether in training mode

        Returns:
            User representations (batch_size, cnn_filter_num)
        """
        # Process all history items using TimeDistributed layer
        news_vectors = self.time_distributed(inputs, training=training)
        # Result: (batch_size, history_length, cnn_filter_num)

        # Create mask for valid history items by checking title tokens
        # Extract title tokens from the concatenated input to create the mask
        title_tokens = inputs[:, :, : self.config.max_title_length]
        history_mask = ops.any(ops.not_equal(title_tokens, 0), axis=-1)

        # Apply user-level attention
        user_representation = self.user_attention(news_vectors, mask=history_mask)

        return user_representation


class NAML(BaseModel):
    """Neural Attentive Multi-View Learning (NAML) model for news recommendation.

    This model is based on the paper: "Neural News Recommendation with Attentive Multi-View Learning"
    by C. Wu et al. It learns news representations from multiple views (title, abstract, category)
    and user representations from their browsing history.

    Key features:
    - Multi-view news encoding: Processes title, abstract, and categories separately.
    - View-level attention: Combines different news views into a unified representation.
    - Attentive user encoding: Uses additive attention over historical news.

    Refactored with clean architecture using separate components for better
    maintainability, testability, and code organization.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NAMLConfig | None = None,
        name: str = "naml",
        **config_overrides,
    ):
        """Build a NAML model.

        Args:
            processed_news: Dataset's processed news dict (vocab_size,
                embeddings, num_categories, num_subcategories, ...).
            config: Optional pre-built NAMLConfig. If ``None``, one is
                constructed from ``config_overrides``.
            name: Keras layer name.
            **config_overrides: Field overrides forwarded to NAMLConfig
                when ``config`` is None. Used by ``build_model_from_spec``
                which dumps every config field as a kwarg.
        """
        super().__init__(name=name)

        if config is None:
            config = NAMLConfig(**config_overrides)
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
            "hist_abstract_tokens": (
                None,
                config.max_history_length,
                config.max_abstract_length,
            ),
            "cand_abstract_tokens": (
                None,
                config.max_impressions_length,
                config.max_abstract_length,
            ),
            "hist_category": (None, config.max_history_length, 1),
            "hist_subcategory": (None, config.max_history_length, 1),
            "cand_category": (None, config.max_impressions_length, 1),
            "cand_subcategory": (None, config.max_impressions_length, 1),
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

        # Create view encoders
        self.title_encoder = TitleEncoder(self.config, self.embedding_layer)
        self.abstract_encoder = AbstractEncoder(self.config, self.embedding_layer)
        self.category_encoder = CategoryEncoder(
            self.config, self.processed_news["num_categories"]
        )
        self.subcategory_encoder = SubcategoryEncoder(
            self.config, self.processed_news["num_subcategories"]
        )

        # Create news encoder that combines all views
        self.news_encoder = NewsEncoder(
            self.config,
            self.title_encoder,
            self.abstract_encoder,
            self.category_encoder,
            self.subcategory_encoder,
        )

        # Create user encoder
        self.user_encoder = UserEncoder(self.config, self.news_encoder)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Thin entry point — delegates to :meth:`score_training_batch`."""
        return self.score_training_batch(inputs, training=training)

    def score_training_batch(self, inputs, training=None):
        """Score a training batch. Returns raw logits ``(B, C)``.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        # Concatenate history inputs (title + abstract + category + subcategory)
        history_concat = ops.concatenate(
            [
                inputs["hist_tokens"],
                inputs["hist_abstract_tokens"],
                ops.expand_dims(inputs["hist_category"], axis=-1),
                ops.expand_dims(inputs["hist_subcategory"], axis=-1),
            ],
            axis=-1,
        )
        user_repr = self.user_encoder(history_concat, training=training)

        # Concatenate candidate inputs and encode via reshape trick
        candidate_concat = ops.concatenate(
            [
                inputs["cand_tokens"],
                inputs["cand_abstract_tokens"],
                ops.expand_dims(inputs["cand_category"], axis=-1),
                ops.expand_dims(inputs["cand_subcategory"], axis=-1),
            ],
            axis=-1,
        )
        B = ops.shape(candidate_concat)[0]
        C = ops.shape(candidate_concat)[1]
        F = ops.shape(candidate_concat)[2]
        flat_cand = ops.reshape(candidate_concat, (B * C, F))
        cand_repr = ops.reshape(
            self.news_encoder(flat_cand, training=training), (B, C, -1)
        )

        user_expanded = ops.expand_dims(user_repr, axis=1)
        return ops.sum(cand_repr * user_expanded, axis=-1)

    def get_config(self):
        """Returns the configuration of the NAML model for serialization."""
        base_config = super().get_config()
        base_config.update(
            {
                "max_title_length": self.config.max_title_length,
                "max_abstract_length": self.config.max_abstract_length,
                "embedding_size": self.config.embedding_size,
                "category_embedding_dim": self.config.category_embedding_dim,
                "subcategory_embedding_dim": self.config.subcategory_embedding_dim,
                "cnn_filter_num": self.config.cnn_filter_num,
                "cnn_kernel_size": self.config.cnn_kernel_size,
                "word_attention_query_dim": self.config.word_attention_query_dim,
                "view_attention_query_dim": self.config.view_attention_query_dim,
                "user_attention_query_dim": self.config.user_attention_query_dim,
                "dropout_rate": self.config.dropout_rate,
                "activation": self.config.activation,
                "max_history_length": self.config.max_history_length,
                "max_impressions_length": self.config.max_impressions_length,
                "process_user_id": self.config.process_user_id,
                "seed": self.config.seed,
            }
        )
        return base_config
