from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import NAMLConfig
from src.frameworks.keras.layers import AdditiveAttentionLayer
from src.frameworks.keras.models.base import BaseModel


class TitleEncoder(keras.Model):
    """Title encoder component for NAML.

    Processes news titles through word embeddings, CNN, and additive attention.
    """

    def __init__(self, config: NAMLConfig, embedding_layer: layers.Embedding, name: str = "title_encoder"):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers
        self.dropout1 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed, name="title_embedding_dropout")
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.activation,
            padding="same",
            name="title_cnn",
        )
        self.dropout2 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed, name="title_cnn_dropout")
        self.additive_attention = AdditiveAttentionLayer(
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

    def __init__(self, config: NAMLConfig, embedding_layer: layers.Embedding, name: str = "abstract_encoder"):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer

        # Create layers
        self.dropout1 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed,
                                       name="abstract_embedding_dropout")
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.activation,
            padding="same",
            name="abstract_cnn",
        )
        self.dropout2 = layers.Dropout(self.config.dropout_rate, seed=self.config.seed, name="abstract_cnn_dropout")
        self.additive_attention = AdditiveAttentionLayer(
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

    def __init__(self, config: NAMLConfig, num_categories: int, name: str = "category_encoder"):
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
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
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

    def __init__(self, config: NAMLConfig, num_subcategories: int, name: str = "subcategory_encoder"):
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
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
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

    def __init__(self,
                 config: NAMLConfig,
                 title_encoder: TitleEncoder,
                 abstract_encoder: AbstractEncoder,
                 category_encoder: CategoryEncoder,
                 subcategory_encoder: SubcategoryEncoder,
                 name: str = "news_encoder"):
        super().__init__(name=name)
        self.config = config
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        # View-level attention
        self.view_attention = AdditiveAttentionLayer(
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
        title_tokens = inputs[:, :self.config.max_title_length]
        abstract_tokens = inputs[
            :, self.config.max_title_length:self.config.max_title_length + self.config.max_abstract_length]
        category_id = inputs[
            :, self.config.max_title_length + self.config.max_abstract_length:self.config.max_title_length + self.config.max_abstract_length + 1]
        subcategory_id = inputs[:, self.config.max_title_length + self.config.max_abstract_length + 1:]

        # Encode each view
        title_vec = self.title_encoder(title_tokens, training=training)
        abstract_vec = self.abstract_encoder(abstract_tokens, training=training)
        category_vec = self.category_encoder(category_id, training=training)
        subcategory_vec = self.subcategory_encoder(subcategory_id, training=training)

        # Stack views for attention (batch_size, 4, cnn_filter_num)
        views = ops.stack([title_vec, abstract_vec, category_vec, subcategory_vec], axis=1)

        # Apply view-level attention to combine views
        news_representation = self.view_attention(views)

        return news_representation


class UserEncoder(keras.Model):
    """User encoder component for NAML.

    Processes user browsing history through news encoder and additive attention
    to produce user representations.
    """

    def __init__(self, config: NAMLConfig, news_encoder: NewsEncoder, name: str = "user_encoder"):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

        # TimeDistributed layer for processing history
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_user"
        )

        # User-level attention
        self.user_attention = AdditiveAttentionLayer(
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
        title_tokens = inputs[:, :, :self.config.max_title_length]
        history_mask = ops.any(ops.not_equal(title_tokens, 0), axis=-1)

        # Apply user-level attention
        user_representation = self.user_attention(news_vectors, mask=history_mask)

        return user_representation


class NAMLScorer(keras.Model):
    """Scoring component for NAML.

    Handles different scoring scenarios: training (multiple candidates with softmax),
    single candidate scoring (with sigmoid), and multiple candidate scoring (raw scores).
    """

    def __init__(self, config: NAMLConfig, news_encoder: NewsEncoder, user_encoder: UserEncoder,
                 name: str = "naml_scorer"):
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

    def score_training_batch(self, history_inputs, candidate_inputs, training=None):
        """Score training batch with softmax output.

        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidates tensor (batch_size, num_candidates, feature_size)
            training: Whether in training mode

        Returns:
            Softmax scores (batch_size, num_candidates)
        """
        # Get user representation from concatenated history
        # Always use training=True for training batch
        user_repr = self.user_encoder(history_inputs, training=True)

        # Get representations for all candidates using stored TimeDistributed layer
        candidate_reprs = self.candidate_encoder_train(candidate_inputs, training=True)
        # Result: (batch_size, num_candidates, cnn_filter_num)

        # Expand user representation for broadcasting
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)  # (batch_size, 1, cnn_filter_num)

        # Calculate scores using dot product
        scores = ops.sum(candidate_reprs * user_repr_expanded, axis=-1)  # (batch_size, num_candidates)

        # Apply softmax for training
        output = ops.softmax(scores, axis=-1)
        return output

    def score_single_candidate(self, history_inputs, candidate_inputs, training=None):
        """Score single candidate with sigmoid output.

        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidate tensor (batch_size, feature_size)
            training: Whether in training mode

        Returns:
            Sigmoid scores (batch_size, 1)
        """
        # Get user representation from concatenated history
        user_repr = self.user_encoder(history_inputs, training=False)

        # Get candidate representation from concatenated input
        candidate_repr = self.news_encoder(candidate_inputs, training=False)

        # Calculate score using dot product
        score = ops.sum(candidate_repr * user_repr, axis=-1, keepdims=True)

        # Apply sigmoid for probability
        output = ops.sigmoid(score)
        return output

    def score_multiple_candidates(self, history_inputs, candidate_inputs, training=None):
        """Score multiple candidates with sigmoid activation.

        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidates tensor (batch_size, num_candidates, feature_size)
            training: Whether in training mode

        Returns:
            Sigmoid scores (batch_size, num_candidates)
        """
        # Get user representation from concatenated history
        user_repr = self.user_encoder(history_inputs, training=False)

        # Get representations for all candidates using stored TimeDistributed layer
        candidate_reprs = self.candidate_encoder_eval(candidate_inputs, training=False)
        # Result: (batch_size, num_candidates, cnn_filter_num)

        # Expand user representation for broadcasting
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)

        # Calculate scores using dot product
        scores = ops.sum(candidate_reprs * user_repr_expanded, axis=-1)

        # Apply sigmoid activation for consistency with single candidate scoring
        output = ops.sigmoid(scores)
        return output


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
            max_title_length: int = 30,
            max_abstract_length: int = 50,
            embedding_size: int = 300,
            category_embedding_dim: int = 100,
            subcategory_embedding_dim: int = 100,
            cnn_filter_num: int = 400,
            cnn_kernel_size: int = 3,
            word_attention_query_dim: int = 200,
            view_attention_query_dim: int = 200,
            user_attention_query_dim: int = 200,
            dropout_rate: float = 0.2,
            activation: str = "relu",
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            process_user_id: bool = False,
            seed: int = 42,
            name: str = "naml",
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration object
        self.config = NAMLConfig(
            max_title_length=max_title_length,
            max_abstract_length=max_abstract_length,
            embedding_size=embedding_size,
            category_embedding_dim=category_embedding_dim,
            subcategory_embedding_dim=subcategory_embedding_dim,
            cnn_filter_num=cnn_filter_num,
            cnn_kernel_size=cnn_kernel_size,
            word_attention_query_dim=word_attention_query_dim,
            view_attention_query_dim=view_attention_query_dim,
            user_attention_query_dim=user_attention_query_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            process_user_id=process_user_id,
            seed=seed,
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
            "hist_abstract_tokens": (None, max_history_length, max_abstract_length),
            "cand_abstract_tokens": (None, max_impressions_length, max_abstract_length),
            "hist_category": (None, max_history_length, 1),
            "hist_subcategory": (None, max_history_length, 1),
            "cand_category": (None, max_impressions_length, 1),
            "cand_subcategory": (None, max_impressions_length, 1),
        }
        self.build(dummy_input_shape)

    def build(self, input_shape) -> None:
        """Create all model components."""
        # Create shared embedding layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(self.processed_news["embeddings"]),
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

        # Create scorer
        self.scorer = NAMLScorer(self.config, self.news_encoder, self.user_encoder)

        # Build training and scorer models for compatibility
        self.training_model, self.scorer_model = self._build_compatibility_models()

        super().build(input_shape)

    def _build_compatibility_models(self) -> tuple[keras.Model, keras.Model]:
        """Build training and scorer models for backward compatibility."""
        # ----- Training model -----
        # Create concatenated inputs that match the expected format
        history_concat_input = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length + self.config.max_abstract_length + 2),
            dtype="int32", name="hist_concat_input"
        )
        candidate_concat_input = keras.Input(
            shape=(self.config.max_impressions_length,
                   self.config.max_title_length + self.config.max_abstract_length + 2),
            dtype="int32", name="cand_concat_input"
        )

        # Pass concatenated inputs directly to the scorer
        training_output = self.scorer.score_training_batch(history_concat_input, candidate_concat_input)

        training_model = keras.Model(
            inputs=[history_concat_input, candidate_concat_input],
            outputs=training_output,
            name="naml_training_model"
        )

        # ----- Scorer model -----
        # Create concatenated inputs for scorer model
        score_history_concat_input = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length + self.config.max_abstract_length + 2),
            dtype="int32", name="score_hist_concat_input"
        )
        score_candidate_concat_input = keras.Input(
            shape=(self.config.max_title_length + self.config.max_abstract_length + 2,),
            dtype="int32", name="score_cand_concat_input"
        )

        # Pass concatenated inputs directly to the scorer
        scorer_output = self.scorer.score_single_candidate(score_history_concat_input, score_candidate_concat_input)

        scorer_model = keras.Model(
            inputs=[score_history_concat_input, score_candidate_concat_input],
            outputs=scorer_output,
            name="naml_scorer_model"
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
            # Training mode expects multiple keys for history and candidates
            required_hist_keys = {"hist_tokens", "hist_abstract_tokens", "hist_category", "hist_subcategory"}
            required_cand_keys = {"cand_tokens", "cand_abstract_tokens", "cand_category", "cand_subcategory"}

            if not (required_hist_keys.issubset(input_keys) and required_cand_keys.issubset(input_keys)):
                raise ValueError(
                    f"Training mode requires history keys: {required_hist_keys} and candidate keys: {required_cand_keys}, "
                    f"but got keys: {list(input_keys)}"
                )
        else:
            # Inference mode - check for different valid combinations
            # Add validation logic for inference mode if needed
            pass

    def call(self, inputs, training=None):
        """Main forward pass of the NAML model.

        Routes to appropriate methods based on training mode and input format.

        Args:
            inputs: Dictionary with input tensors
            training: Whether in training mode

        Returns:
            Model predictions based on mode and input format
        """
        self._validate_inputs(inputs, training)

        if training:
            # Training mode: always use training format
            return self._handle_training(inputs)
        else:
            # Inference mode: route based on input format
            if "hist_tokens" in inputs and "cand_tokens" in inputs:
                # Multiple candidates format
                return self._handle_multiple_candidates(inputs)
            else:
                # Add other inference modes as needed
                raise ValueError("Invalid input format for inference mode")

    def _handle_training(self, inputs):
        """Handle training batch scoring with softmax output."""
        # Concatenate history inputs
        history_concat = ops.concatenate([
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            ops.expand_dims(inputs["hist_category"], axis=-1),
            ops.expand_dims(inputs["hist_subcategory"], axis=-1),
        ], axis=-1)

        # Concatenate candidate inputs
        candidate_concat = ops.concatenate([
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            ops.expand_dims(inputs["cand_category"], axis=-1),
            ops.expand_dims(inputs["cand_subcategory"], axis=-1),
        ], axis=-1)

        return self.scorer.score_training_batch(history_concat, candidate_concat)

    def _handle_multiple_candidates(self, inputs):
        """Handle multiple candidate scoring with sigmoid scores."""
        # Concatenate history inputs
        history_concat = ops.concatenate([
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            ops.expand_dims(inputs["hist_category"], axis=-1),
            ops.expand_dims(inputs["hist_subcategory"], axis=-1),
        ], axis=-1)

        # Concatenate candidate inputs
        candidate_concat = ops.concatenate([
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            ops.expand_dims(inputs["cand_category"], axis=-1),
            ops.expand_dims(inputs["cand_subcategory"], axis=-1),
        ], axis=-1)

        return self.scorer.score_multiple_candidates(history_concat, candidate_concat)

    def get_config(self):
        """Returns the configuration of the NAML model for serialization."""
        base_config = super().get_config()
        base_config.update({
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
        })
        return base_config
