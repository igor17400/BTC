from __future__ import annotations

from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import LSTURConfig
from src.frameworks.keras.layers import (
    AdditiveAttention,
    ComputeMasking,
    OverwriteMasking,
)
from src.frameworks.keras.models.base import BaseModel


class NewsEncoder(keras.Model):
    """News encoder component for LSTUR.

    Processes news titles through word embeddings, CNN, and additive attention.
    Can optionally incorporate category and subcategory information for multi-view learning.
    """

    def __init__(
        self,
        config: LSTURConfig,
        embedding_layer: layers.Embedding,
        category_encoder: CategoryEncoder | None = None,
        subcategory_encoder: SubcategoryEncoder | None = None,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        # Create layers for title processing
        self.dropout1 = layers.Dropout(
            self.config.dropout_rate, seed=self.config.seed, name="embedding_dropout"
        )
        self.cnn = layers.Conv1D(
            self.config.cnn_filter_num,
            self.config.cnn_kernel_size,
            activation=self.config.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
            name="title_cnn",
        )
        self.dropout2 = layers.Dropout(
            self.config.dropout_rate, seed=self.config.seed, name="cnn_dropout"
        )
        self.compute_masking = ComputeMasking(name="compute_masking")
        self.overwrite_masking = OverwriteMasking(name="overwrite_masking")
        self.additive_attention = AdditiveAttention(
            self.config.attention_hidden_dim,
            seed=self.config.seed,
            name="title_additive_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.

        Args:
            input_shape: Input shape (batch_size, title_length or title_length + 2)

        Returns:
            Output shape (batch_size, output_dim) where output_dim depends on
            whether category/subcategory encoders are used
        """
        # Base dimension from title CNN
        output_dim = self.config.cnn_filter_num

        # Add category dimension if encoder is present
        if self.category_encoder is not None:
            output_dim += self.config.category_embedding_dim

        # Add subcategory dimension if encoder is present
        if self.subcategory_encoder is not None:
            output_dim += self.config.subcategory_embedding_dim

        return (input_shape[0], output_dim)

    def call(self, inputs, training=None):
        """Forward pass for news encoding.

        Args:
            inputs: News token sequences
                - If concatenated: (batch_size, title_length + 2) with category/subcategory
                - If title only: (batch_size, title_length)
            training: Whether in training mode

        Returns:
            News representations (batch_size, cnn_filter_num)
        """
        # Check if input contains concatenated data (title + category + subcategory)
        input_shape = ops.shape(inputs)
        has_category_data = input_shape[-1] > self.config.max_title_length

        if has_category_data and (
            self.category_encoder is not None or self.subcategory_encoder is not None
        ):
            # Split concatenated input: [title, category, subcategory]
            title_tokens = inputs[:, : self.config.max_title_length]
            category_id = inputs[
                :, self.config.max_title_length : self.config.max_title_length + 1
            ]
            subcategory_id = inputs[:, self.config.max_title_length + 1 :]

            # Process title
            title_vec = self._process_title(title_tokens, training)

            # Collect embeddings for concatenation (following LSTUR paper approach)
            representations = [title_vec]

            # Process category if encoder is available
            if self.category_encoder is not None:
                category_vec = self.category_encoder(category_id, training=training)
                representations.append(category_vec)

            # Process subcategory if encoder is available
            if self.subcategory_encoder is not None:
                subcategory_vec = self.subcategory_encoder(
                    subcategory_id, training=training
                )
                representations.append(subcategory_vec)

            if len(representations) > 1:
                # Concatenate all representations (title + category + subcategory)
                # This follows the LSTUR paper approach of combining embeddings

                news_representation = ops.concatenate(representations, axis=-1)
            else:
                news_representation = title_vec

        else:
            # Title-only input or no category encoders available
            if has_category_data:
                title_tokens = inputs[:, : self.config.max_title_length]
            else:
                title_tokens = inputs

            news_representation = self._process_title(title_tokens, training)

        return news_representation

    def _process_title(self, title_tokens, training=None):
        """Process title tokens through CNN and attention.

        Args:
            title_tokens: Title token sequences (batch_size, title_length)
            training: Whether in training mode

        Returns:
            Title representations (batch_size, cnn_filter_num)
        """
        # Word Embedding
        embedded_sequences = self.embedding_layer(title_tokens)

        # Dropout after embedding
        y = self.dropout1(embedded_sequences, training=training)

        # CNN
        y = self.cnn(y)

        # Dropout after CNN
        y = self.dropout2(y, training=training)

        # Create mask and apply it
        mask = self.compute_masking(title_tokens)
        y = self.overwrite_masking([y, mask])

        # Apply masking for attention
        padding_mask = ops.not_equal(title_tokens, 0)

        # Additive Attention to get single news vector
        title_representation = self.additive_attention(y, mask=padding_mask)

        return title_representation


class CategoryEncoder(keras.Model):
    """Category encoder component for LSTUR.

    Simple embedding layer for category IDs as described in the LSTUR paper.
    """

    def __init__(
        self, config: LSTURConfig, num_categories: int, name: str = "category_encoder"
    ):
        super().__init__(name=name)
        self.config = config
        self.num_categories = num_categories

        # Create embedding layer only
        self.embedding = layers.Embedding(
            self.num_categories + 1,
            self.config.category_embedding_dim,
            trainable=True,
            name="category_embedding",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the category encoder.

        Args:
            input_shape: Input shape (batch_size, 1)

        Returns:
            Output shape (batch_size, category_embedding_dim)
        """
        return (input_shape[0], self.config.category_embedding_dim)

    def call(self, inputs, training=None):
        """Forward pass for category encoding.

        Args:
            inputs: Category IDs (batch_size, 1)
            training: Whether in training mode

        Returns:
            Category representations (batch_size, category_embedding_dim)
        """
        # Embedding
        embedded = self.embedding(inputs)

        # Reshape from (batch_size, 1, category_embedding_dim) to (batch_size, category_embedding_dim)
        category_representation = ops.squeeze(embedded, axis=1)

        return category_representation


class SubcategoryEncoder(keras.Model):
    """Subcategory encoder component for LSTUR.

    Simple embedding layer for subcategory IDs as described in the LSTUR paper.
    """

    def __init__(
        self,
        config: LSTURConfig,
        num_subcategories: int,
        name: str = "subcategory_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.num_subcategories = num_subcategories

        # Create embedding layer only
        self.embedding = layers.Embedding(
            self.num_subcategories + 1,
            self.config.subcategory_embedding_dim,
            trainable=True,
            name="subcategory_embedding",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the subcategory encoder.

        Args:
            input_shape: Input shape (batch_size, 1)

        Returns:
            Output shape (batch_size, subcategory_embedding_dim)
        """
        return (input_shape[0], self.config.subcategory_embedding_dim)

    def call(self, inputs, training=None):
        """Forward pass for subcategory encoding.

        Args:
            inputs: Subcategory IDs (batch_size, 1)
            training: Whether in training mode

        Returns:
            Subcategory representations (batch_size, subcategory_embedding_dim)
        """
        # Embedding
        embedded = self.embedding(inputs)

        # Reshape from (batch_size, 1, subcategory_embedding_dim) to (batch_size, subcategory_embedding_dim)
        subcategory_representation = ops.squeeze(embedded, axis=1)

        return subcategory_representation


class UserEncoder(keras.Model):
    """User encoder component for LSTUR.

    Processes user history through news encoder and GRU with user embeddings
    to produce user representations.
    """

    def __init__(
        self,
        config: LSTURConfig,
        news_encoder: NewsEncoder,
        num_users: int,
        name: str = "user_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.num_users = num_users

        # User embedding layer
        self.user_embedding = layers.Embedding(
            self.num_users,
            self.config.gru_unit,
            trainable=True,
            embeddings_initializer="zeros",
            name="user_embedding",
        )

        # Bernoulli masking on user embeddings during training (paper §3.2)
        self.user_embedding_dropout = layers.Dropout(0.5, name="user_emb_dropout")

        # TimeDistributed layer for processing history
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_user"
        )

        # GRU layer
        self.gru = layers.GRU(
            self.config.gru_unit,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.config.seed),
            recurrent_initializer=keras.initializers.GlorotUniform(
                seed=self.config.seed
            ),
            bias_initializer=keras.initializers.Zeros(),
            return_sequences=False,
            name="user_gru",
        )

        # Masking layer for GRU input
        self.masking = layers.Masking(mask_value=0.0, name="gru_masking")

        # Dense layer for "con" type
        if self.config.type == "con":
            self.concat_dense = layers.Dense(
                self.config.gru_unit,
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.GlorotUniform(
                    seed=self.config.seed
                ),
                name="concat_dense",
            )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the user encoder.

        Args:
            input_shape: Tuple representing input shapes

        Returns:
            Output shape (batch_size, gru_unit)
        """
        return (input_shape[0][0], self.config.gru_unit)

    def call(self, inputs, training=None):
        """Forward pass for user encoding.

        Args:
            inputs: List of [history_tokens, user_indices]
                - history_tokens: (batch_size, history_length, title_length)
                - user_indices: (batch_size,) or (batch_size, 1)
            training: Whether in training mode

        Returns:
            User representations (batch_size, gru_unit)
        """
        history_tokens, user_indices = inputs

        # Get user embeddings
        # Handle both (batch_size,) and (batch_size, 1) shapes
        if len(ops.shape(user_indices)) == 1:
            user_indices = ops.expand_dims(user_indices, axis=-1)

        long_u_emb = self.user_embedding(user_indices)
        long_u_emb = ops.squeeze(long_u_emb, axis=1)  # (batch_size, gru_unit)
        # Bernoulli masking of long-term user repr during training (paper §3.2)
        long_u_emb = self.user_embedding_dropout(long_u_emb, training=training)

        # Process all history items using TimeDistributed layer
        click_title_presents = self.time_distributed(history_tokens, training=training)
        # Result: (batch_size, history_length, cnn_filter_num)

        # Apply masking for GRU
        masked_presents = self.masking(click_title_presents)

        if self.config.type == "ini":
            # Use user embedding as initial state
            user_present = self.gru(masked_presents, initial_state=[long_u_emb])
        elif self.config.type == "con":
            # Concatenate short-term and long-term representations
            short_uemb = self.gru(masked_presents)
            concat_emb = ops.concatenate([short_uemb, long_u_emb], axis=-1)
            user_present = self.concat_dense(concat_emb)
        else:
            raise ValueError(f"Invalid user encoder type: {self.config.type}")

        return user_present


class LSTURScorer(keras.Model):
    """Scoring component for LSTUR.

    Handles different scoring scenarios: training (multiple candidates with softmax),
    single candidate scoring (with sigmoid), and multiple candidate scoring (raw scores).
    """

    def __init__(
        self,
        config: LSTURConfig,
        news_encoder: NewsEncoder,
        user_encoder: UserEncoder,
        name: str = "lstur_scorer",
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

    def score_training_batch(
        self, history_inputs, user_indices, candidate_inputs, training=None
    ):
        """Score training batch (raw logits — loss handles softmax).

        Args:
            history_inputs: History tensor (batch_size, history_length, title_length)
            user_indices: User indices (batch_size, 1)
            candidate_inputs: Candidates tensor (batch_size, num_candidates, title_length)
            training: Whether in training mode

        Returns:
            Raw logit scores (batch_size, num_candidates)
        """
        # Get user representation
        user_repr = self.user_encoder([history_inputs, user_indices], training=training)

        # Get representations for all candidates
        candidate_reprs = self.candidate_encoder_train(
            candidate_inputs, training=training
        )
        # Result: (batch_size, num_candidates, cnn_filter_num)

        # Calculate scores using dot product (raw logits)
        scores = layers.Dot(axes=-1, name="dot_product_train")(
            [candidate_reprs, user_repr]
        )

        return scores

    def score_single_candidate(
        self, history_inputs, user_indices, candidate_inputs, training=None
    ):
        """Score single candidate with sigmoid output.

        Args:
            history_inputs: History tensor (batch_size, history_length, title_length)
            user_indices: User indices (batch_size, 1)
            candidate_inputs: Candidate tensor (batch_size, title_length)
            training: Whether in training mode

        Returns:
            Sigmoid scores (batch_size, 1)
        """
        # Get user representation
        user_repr = self.user_encoder([history_inputs, user_indices], training=training)

        # Get candidate representation
        candidate_repr = self.news_encoder(candidate_inputs, training=training)

        # Calculate score using dot product
        score = layers.Dot(axes=-1, name="dot_product_single")(
            [candidate_repr, user_repr]
        )

        # Apply sigmoid for probability
        return layers.Activation("sigmoid", name="sigmoid_activation")(score)

    def score_multiple_candidates(
        self, history_inputs, user_indices, candidate_inputs, training=False
    ):
        """Score multiple candidates with sigmoid activation.

        Args:
            history_inputs: History tensor (batch_size, history_length, title_length)
            user_indices: User indices (batch_size, 1)
            candidate_inputs: Candidates tensor (batch_size, num_candidates, title_length)
            training: Whether in training mode

        Returns:
            Sigmoid scores (batch_size, num_candidates)
        """
        # Get user representation
        user_repr = self.user_encoder([history_inputs, user_indices], training=training)

        # Get representations for all candidates
        candidate_reprs = self.candidate_encoder_eval(
            candidate_inputs, training=training
        )
        # Result: (batch_size, num_candidates, cnn_filter_num)

        # Expand user representation for broadcasting
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)

        # Calculate scores using dot product
        scores = ops.sum(candidate_reprs * user_repr_expanded, axis=-1)

        # Apply sigmoid activation for consistency with single candidate scoring
        return ops.sigmoid(scores)


class LSTUR(BaseModel):
    """Neural News Recommendation with Long- and Short-term User Representations (LSTUR) model.

    This model is based on the paper: "Neural News Recommendation with Long- and Short-term
    User Representations" by M. An et al., ACL 2019. It learns news representations through
    CNN and attention, and user representations through GRU with user embeddings.

    Key features:
    - CNN-based news encoding with additive attention
    - GRU-based user encoding with long-term user embeddings
    - Two types of user encoders: "ini" (initial state) and "con" (concatenation)

    Refactored with clean architecture using separate components for better
    maintainability, testability, and code organization.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        num_users: int,
        embedding_size: int = 300,
        cnn_filter_num: int = 300,
        cnn_kernel_size: int = 3,
        cnn_activation: str = "relu",
        attention_hidden_dim: int = 200,
        gru_unit: int = 300,
        type: str = "ini",
        dropout_rate: float = 0.2,
        seed: int = 42,
        max_title_length: int = 50,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        process_user_id: bool = True,
        use_category: bool = False,
        use_subcategory: bool = False,
        category_embedding_dim: int = 100,
        subcategory_embedding_dim: int = 100,
        name: str = "lstur",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration object
        self.config = LSTURConfig(
            embedding_size=embedding_size,
            cnn_filter_num=cnn_filter_num,
            cnn_kernel_size=cnn_kernel_size,
            cnn_activation=cnn_activation,
            attention_hidden_dim=attention_hidden_dim,
            gru_unit=gru_unit,
            type=type,
            dropout_rate=dropout_rate,
            seed=seed,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            process_user_id=process_user_id,
            use_category=use_category,
            use_subcategory=use_subcategory,
            category_embedding_dim=category_embedding_dim,
            subcategory_embedding_dim=subcategory_embedding_dim,
        )

        # Store processed news data and num_users
        self.processed_news = processed_news
        self.num_users = num_users
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
            embeddings_initializer=keras.initializers.Constant(
                self.processed_news["embeddings"]
            ),
            trainable=True,
            mask_zero=False,  # LSTUR uses custom masking
            name="word_embedding",
        )

        # Create category/subcategory encoders if enabled
        category_encoder = None
        subcategory_encoder = None

        if self.config.use_category:
            category_encoder = CategoryEncoder(
                self.config, self.processed_news["num_categories"]
            )

        if self.config.use_subcategory:
            subcategory_encoder = SubcategoryEncoder(
                self.config, self.processed_news["num_subcategories"]
            )

        # Create component encoders
        self.news_encoder = NewsEncoder(
            self.config,
            self.embedding_layer,
            category_encoder=category_encoder,
            subcategory_encoder=subcategory_encoder,
        )
        self.user_encoder = UserEncoder(self.config, self.news_encoder, self.num_users)
        self.scorer = LSTURScorer(self.config, self.news_encoder, self.user_encoder)

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
        user_indices_input = keras.Input(shape=(1,), dtype="int32", name="user_indices")
        candidates_input = keras.Input(
            shape=(self.config.max_impressions_length, self.config.max_title_length),
            dtype="int32",
            name="cand_tokens",
        )

        training_output = self.scorer.score_training_batch(
            history_input, user_indices_input, candidates_input
        )
        training_model = keras.Model(
            inputs=[history_input, user_indices_input, candidates_input],
            outputs=training_output,
            name="lstur_training_model",
        )

        # ----- Scorer model -----
        history_input_score = keras.Input(
            shape=(self.config.max_history_length, self.config.max_title_length),
            dtype="int32",
            name="history_tokens_score",
        )
        user_indices_input_score = keras.Input(
            shape=(1,), dtype="int32", name="user_indices_score"
        )
        single_candidate_input = keras.Input(
            shape=(self.config.max_title_length,),
            dtype="int32",
            name="single_candidate_tokens_score",
        )

        scorer_output = self.scorer.score_single_candidate(
            history_input_score, user_indices_input_score, single_candidate_input
        )
        scorer_model = keras.Model(
            inputs=[
                history_input_score,
                user_indices_input_score,
                single_candidate_input,
            ],
            outputs=scorer_output,
            name="lstur_scorer_model",
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
            # Training mode expects hist_tokens, user_ids, and cand_tokens
            # Note: user_ids is the key used by the dataloader
            required_keys = {"hist_tokens", "user_ids", "cand_tokens"}
            if not required_keys.issubset(input_keys):
                raise ValueError(
                    f"Training mode requires keys: {required_keys}, "
                    f"but got keys: {list(input_keys)}"
                )
        else:
            # Inference mode - check for different valid combinations
            # Add validation logic for inference mode if needed
            pass

    def call(self, inputs, training=None):
        """Main forward pass of the LSTUR model.

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
            return self._handle_training(inputs, training)
        else:
            # Inference mode: route based on input format
            if "hist_tokens" in inputs and "cand_tokens" in inputs:
                # Multiple candidates format
                return self._handle_multiple_candidates(inputs)
            elif "hist_tokens" in inputs and "single_candidate_tokens" in inputs:
                # Single candidate format
                return self._handle_single_candidate(inputs)
            else:
                # Add other inference modes as needed
                raise ValueError("Invalid input format for inference mode")

    def _handle_training(self, inputs, training=None):
        """Handle training batch scoring (raw logits)."""
        # Extract the inputs - note that dataloader uses "user_ids" not "user_indices"
        history_tokens = inputs["hist_tokens"]
        user_ids = inputs.get(
            "user_ids", inputs.get("user_indices")
        )  # Handle both keys
        candidate_tokens = inputs["cand_tokens"]

        # Check if we have category/subcategory data
        if "hist_category" in inputs and "hist_subcategory" in inputs:
            # Concatenate history inputs: title + category + subcategory
            hist_category = ops.expand_dims(inputs["hist_category"], axis=-1)
            hist_subcategory = ops.expand_dims(inputs["hist_subcategory"], axis=-1)
            history_tokens = ops.concatenate(
                [history_tokens, hist_category, hist_subcategory], axis=-1
            )

            # Concatenate candidate inputs: title + category + subcategory
            cand_category = ops.expand_dims(inputs["cand_category"], axis=-1)
            cand_subcategory = ops.expand_dims(inputs["cand_subcategory"], axis=-1)
            candidate_tokens = ops.concatenate(
                [candidate_tokens, cand_category, cand_subcategory], axis=-1
            )

        return self.scorer.score_training_batch(
            history_tokens, user_ids, candidate_tokens, training=training
        )

    def _handle_multiple_candidates(self, inputs):
        """Handle multiple candidate scoring with sigmoid scores."""
        # Extract the inputs - handle both user_ids and user_indices keys
        history_tokens = inputs["hist_tokens"]
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))
        candidate_tokens = inputs["cand_tokens"]

        # Check if we have category/subcategory data
        if "hist_category" in inputs and "hist_subcategory" in inputs:
            # Concatenate history inputs: title + category + subcategory
            hist_category = ops.expand_dims(inputs["hist_category"], axis=-1)
            hist_subcategory = ops.expand_dims(inputs["hist_subcategory"], axis=-1)
            history_tokens = ops.concatenate(
                [history_tokens, hist_category, hist_subcategory], axis=-1
            )

            # Concatenate candidate inputs: title + category + subcategory
            cand_category = ops.expand_dims(inputs["cand_category"], axis=-1)
            cand_subcategory = ops.expand_dims(inputs["cand_subcategory"], axis=-1)
            candidate_tokens = ops.concatenate(
                [candidate_tokens, cand_category, cand_subcategory], axis=-1
            )

        return self.scorer.score_multiple_candidates(
            history_tokens, user_ids, candidate_tokens, training=False
        )

    def _handle_single_candidate(self, inputs):
        """Handle single candidate scoring with sigmoid scores."""
        # Extract the inputs - handle both user_ids and user_indices keys
        history_tokens = inputs["hist_tokens"]
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))
        candidate_tokens = inputs["single_candidate_tokens"]

        # Check if we have category/subcategory data
        if "hist_category" in inputs and "hist_subcategory" in inputs:
            # Concatenate history inputs: title + category + subcategory
            hist_category = ops.expand_dims(inputs["hist_category"], axis=-1)
            hist_subcategory = ops.expand_dims(inputs["hist_subcategory"], axis=-1)
            history_tokens = ops.concatenate(
                [history_tokens, hist_category, hist_subcategory], axis=-1
            )

            # Concatenate single candidate inputs: title + category + subcategory
            if "single_cand_category" in inputs and "single_cand_subcategory" in inputs:
                single_cand_category = ops.expand_dims(
                    inputs["single_cand_category"], axis=-1
                )
                single_cand_subcategory = ops.expand_dims(
                    inputs["single_cand_subcategory"], axis=-1
                )
                candidate_tokens = ops.concatenate(
                    [candidate_tokens, single_cand_category, single_cand_subcategory],
                    axis=-1,
                )

        return self.scorer.score_single_candidate(
            history_tokens, user_ids, candidate_tokens, training=False
        )

    def get_config(self):
        """Returns the configuration of the LSTUR model for serialization."""
        base_config = super().get_config()
        base_config.update(
            {
                "num_users": self.num_users,
                "embedding_size": self.config.embedding_size,
                "cnn_filter_num": self.config.cnn_filter_num,
                "cnn_kernel_size": self.config.cnn_kernel_size,
                "cnn_activation": self.config.cnn_activation,
                "attention_hidden_dim": self.config.attention_hidden_dim,
                "gru_unit": self.config.gru_unit,
                "type": self.config.type,
                "dropout_rate": self.config.dropout_rate,
                "seed": self.config.seed,
                "max_title_length": self.config.max_title_length,
                "max_history_length": self.config.max_history_length,
                "max_impressions_length": self.config.max_impressions_length,
                "process_user_id": self.config.process_user_id,
                "use_category": self.config.use_category,
                "use_subcategory": self.config.use_subcategory,
                "category_embedding_dim": self.config.category_embedding_dim,
                "subcategory_embedding_dim": self.config.subcategory_embedding_dim,
            }
        )
        return base_config
