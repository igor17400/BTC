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
            kernel_initializer=keras.initializers.GlorotUniform(),
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
        self.user_embedding_dropout = layers.Dropout(
            self.config.user_embedding_dropout_rate, name="user_emb_dropout"
        )

        # TimeDistributed layer for processing history
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_user"
        )

        # GRU layer
        self.gru = layers.GRU(
            self.config.gru_unit,
            kernel_initializer=keras.initializers.GlorotUniform(),
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
        config: LSTURConfig | None = None,
        name: str = "lstur",
        **config_overrides,
    ):
        """Build an LSTUR model.

        Args:
            processed_news: Dataset's processed news dict (vocab_size,
                embeddings, num_categories, num_subcategories, ...).
            num_users: Number of unique users in the dataset (used for
                the long-term user embedding table).
            config: Optional pre-built LSTURConfig. If ``None``, one is
                constructed from ``config_overrides``.
            name: Keras layer name.
            **config_overrides: Field overrides forwarded to LSTURConfig
                when ``config`` is None. Used by ``build_model_from_spec``
                which dumps every config field as a kwarg.
        """
        super().__init__(name=name)

        if config is None:
            config = LSTURConfig(**config_overrides)
        self.config = config

        # Store processed news data and num_users
        self.processed_news = processed_news
        self.num_users = num_users
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
        history_tokens = inputs["hist_tokens"]
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))
        candidate_tokens = inputs["cand_tokens"]

        # Optionally concatenate category/subcategory into the token tensors
        if "hist_category" in inputs and "hist_subcategory" in inputs:
            hist_category = ops.expand_dims(inputs["hist_category"], axis=-1)
            hist_subcategory = ops.expand_dims(inputs["hist_subcategory"], axis=-1)
            history_tokens = ops.concatenate(
                [history_tokens, hist_category, hist_subcategory], axis=-1
            )
            cand_category = ops.expand_dims(inputs["cand_category"], axis=-1)
            cand_subcategory = ops.expand_dims(inputs["cand_subcategory"], axis=-1)
            candidate_tokens = ops.concatenate(
                [candidate_tokens, cand_category, cand_subcategory], axis=-1
            )

        user_repr = self.user_encoder(
            [history_tokens, user_ids], training=training
        )

        B = ops.shape(candidate_tokens)[0]
        C = ops.shape(candidate_tokens)[1]
        T = ops.shape(candidate_tokens)[2]
        flat_cand = ops.reshape(candidate_tokens, (B * C, T))
        cand_repr = ops.reshape(
            self.news_encoder(flat_cand, training=training), (B, C, -1)
        )

        user_expanded = ops.expand_dims(user_repr, axis=1)
        return ops.sum(cand_repr * user_expanded, axis=-1)

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
