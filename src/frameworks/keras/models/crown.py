"""
Code Reference: https://github.com/seongeunryu/crown-www25
"""

from typing import Any

import keras
import numpy as np
from keras import layers, ops

from src.core.models.configs import CROWNConfig
from src.frameworks.keras.attention_layers import AdditiveAttention
from src.frameworks.keras.graph_layers import (
    GraphAttentionLayer,
    GraphSAGELayer,
    MultiHeadAttentionBlock,
)
from src.frameworks.keras.models.base import BaseModel


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for Multi-head Attention Block (MAB) models.

    Why Positional Encoding is Essential:
    --------------------------------------
    Transformers and attention mechanisms are inherently permutation-invariant, meaning they
    cannot distinguish between different positions in a sequence. Without positional information,
    the model treats "The cat sat on the mat" the same as "Mat the on sat the cat".

    In CROWN's Context:
    -------------------
    CROWN uses Multi-head Attention Blocks (MABs) to process news titles and abstracts.
    Positional encoding is crucial because:

    1. **Word Order Matters**: In news articles, word order carries semantic meaning.
       "Company acquires startup" vs "Startup acquires company" have opposite meanings.

    2. **Syntactic Structure**: Position helps identify subject, verb, object relationships.
       Early positions often contain subjects, while later positions contain objects/details.

    3. **Title vs Body Processing**: Titles are typically shorter and front-loaded with key
       information, while abstracts have more complex positional patterns.

    How It Works:
    -------------
    Uses sinusoidal functions to generate unique position encodings:
    - Even dimensions: sin(pos / 10000^(2i/d_model))
    - Odd dimensions: cos(pos / 10000^(2i/d_model))

    These are added to word embeddings before attention processing:
    embedded_words + positional_encoding = position-aware embeddings

    Benefits of Sinusoidal Encoding:
    --------------------------------
    1. **Deterministic**: No learned parameters, consistent across models
    2. **Extrapolation**: Can handle sequences longer than training data
    3. **Relative Position**: The model can learn to attend based on relative positions
       (sin(pos+k) can be expressed as a function of sin(pos) and cos(pos))
    4. **Smooth Decay**: Positions far apart have increasingly different encodings

    Args:
        dropout_rate: Dropout applied after adding positional encoding
        max_len: Maximum sequence length to generate encodings for
    """

    def __init__(self, dropout_rate=0.1, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate)
        self.max_len = max_len
        self.supports_masking = True

    def build(self, input_shape):
        model_dimension = input_shape[-1]

        # Create positional encoding matrix using numpy for initialization
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, model_dimension, 2, dtype=np.float32)
            * (-np.log(10000.0) / model_dimension)
        )

        positional_encoding = np.zeros(
            (self.max_len, model_dimension), dtype=np.float32
        )
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)

        # Add batch dimension
        positional_encoding = positional_encoding.reshape(
            1, self.max_len, model_dimension
        )

        self.positional_encoding = self.add_weight(
            name="positional_encoding",
            shape=(1, self.max_len, model_dimension),
            initializer=keras.initializers.Constant(positional_encoding),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x, training=None):
        """Add positional encoding to input embeddings."""
        # Use broadcast_to to avoid dynamic slicing issues with JAX
        input_shape = ops.shape(x)
        batch_size, seq_len = input_shape[0], input_shape[1]

        # Select the appropriate slice of positional encoding and broadcast to match input
        pe_for_input = self.positional_encoding[:, :seq_len, :]
        pe_broadcasted = ops.broadcast_to(
            pe_for_input, (batch_size, seq_len, pe_for_input.shape[-1])
        )

        x = x + pe_broadcasted
        return self.dropout(x, training=training)

    def compute_mask(self, inputs, mask=None):
        """Pass through the mask unchanged."""
        return mask


class CategoryPredictor(layers.Layer):
    """Category prediction layer for auxiliary task."""

    def __init__(self, num_categories, **kwargs):
        super().__init__(**kwargs)
        self.num_categories = num_categories

    def build(self, input_shape):
        self.fc = layers.Dense(
            self.num_categories,
            kernel_initializer="glorot_uniform",
            name="category_predictor_dense",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict category from intent embeddings.

        Args:
            inputs: Intent embeddings (batch_size * news_num, intent_embedding_dim)

        Returns:
            Category logits (batch_size * news_num, num_categories)
        """
        return self.fc(inputs)


class NewsEncoder(keras.Model):
    """News encoder component for CROWN model.

    Implements category-guided intent disentanglement and consistency-based
    news representation.
    """

    def __init__(
        self,
        config: CROWNConfig,
        embedding_layer: layers.Embedding,
        category_embedding: layers.Embedding,
        subcategory_embedding: layers.Embedding,
        category_predictor: CategoryPredictor,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.embedding_layer = embedding_layer
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        # Category predictor for auxiliary task - initialized with default, will be overridden by parent
        self.category_predictor = category_predictor

        # News embedding dimension calculation
        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        # Dropout layer
        self.dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

        # Positional encoding
        self.title_pos_encoder = PositionalEncoding(
            config.dropout_rate, config.max_title_length
        )
        self.body_pos_encoder = PositionalEncoding(
            config.dropout_rate, config.max_abstract_length
        )

        # MAB encoders
        self.title_mab = self._build_mab_encoder("title")
        self.body_mab = self._build_mab_encoder("body")

        # Category affine transformation
        self.category_affine = layers.Dense(
            config.category_embedding_dim,
            kernel_initializer="glorot_uniform",
            name="category_affine",
        )

        # Intent layers for k-intent disentanglement
        self.intent_layers = []
        for i in range(config.intent_num):
            self.intent_layers.append(
                layers.Dense(
                    config.intent_embedding_dim,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    name=f"intent_layer_{i}",
                )
            )

        # Intent attention layers
        self.title_intent_attention = AdditiveAttention(
            query_vec_dim=config.attention_dim,
            seed=config.seed,
            name="title_intent_attention",
        )
        self.body_intent_attention = AdditiveAttention(
            query_vec_dim=config.attention_dim,
            seed=config.seed,
            name="body_intent_attention",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the news encoder.

        Args:
            input_shape: Input shape (batch_size, title_length + abstract_length + 2)

        Returns:
            Output shape (batch_size, news_embedding_dim)
        """
        return (input_shape[0], self.news_embedding_dim)

    def _build_mab_encoder(self, name_prefix):
        """Build an encoder using single MultiHeadAttentionBlock from layers.py.

        According to the CROWN paper, we use a single MAB (Multi-Head Attention Block):
        MAB(X,Y) = LayerNorm(H + FeedForward(H)) where H = LayerNorm(X + Multihead(X,Y,Y))
        """

        # Create single MAB block as per CROWN paper
        mab_block = MultiHeadAttentionBlock(
            dim_out=self.config.embedding_size,
            num_heads=self.config.num_heads,
            use_layer_norm=True,
            seed=self.config.seed,
            name=f"{name_prefix}_mab_block",
        )

        return mab_block

    def k_intent_disentangle(self, news_embedding):
        """Apply k-FC layers for k-intent disentanglement.

        Args:
            news_embedding: (batch_size * news_num, embedding_dim)

        Returns:
            k_intent_embeddings: (batch_size * news_num, k, intent_embedding_dim)
        """
        k_intent_embeddings = []

        for i in range(self.config.intent_num):
            # Apply different linear transformation for each intent
            intent_embedding = self.intent_layers[i](news_embedding)
            # Expand dimension
            intent_embedding_exp = ops.expand_dims(intent_embedding, axis=1)
            k_intent_embeddings.append(intent_embedding_exp)

        # Concatenate along the intent dimension
        k_intent_embeddings = ops.concatenate(k_intent_embeddings, axis=1)

        return k_intent_embeddings

    def similarity_compute(self, title, body):
        """Compute title-body similarity using cosine similarity.

        Args:
            title: (batch_size * news_num, intent_embedding_dim)
            body: (batch_size * news_num, intent_embedding_dim)

        Returns:
            similarity: (batch_size * news_num,)
        """
        # Normalize vectors
        title_norm = ops.sqrt(ops.sum(ops.square(title), axis=1, keepdims=True))
        body_norm = ops.sqrt(ops.sum(ops.square(body), axis=1, keepdims=True))

        title_normalized = title / (title_norm + keras.backend.epsilon())
        body_normalized = body / (body_norm + keras.backend.epsilon())

        # Compute cosine similarity
        cosine_similarity = ops.sum(title_normalized * body_normalized, axis=1)

        # Scale to [0, 1]
        title_body_similarity = (cosine_similarity + 1) / 2.0

        return title_body_similarity

    def call(self, inputs, training=None, return_category_logits=False):
        """Forward pass for news encoding.

        Args:
            inputs: Concatenated tensor containing [title_tokens, abstract_tokens, category_id, subcategory_id]
                   Shape: (batch_size, title_length + abstract_length + 2)
            training: Whether in training mode
            return_category_logits: Whether to return category logits for auxiliary loss

        Returns:
            If return_category_logits is False:
                News representations (batch_size, news_embedding_dim)
            If return_category_logits is True:
                Tuple of (news_representations, category_logits, category_ids)
        """
        # Split the concatenated input: [title, abstract, category, subcategory]
        # Use fixed dimensions to avoid dynamic conditions that break JAX tracing
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

        # Squeeze dimension
        category_id = ops.squeeze(category_id, axis=-1)
        subcategory_id = ops.squeeze(subcategory_id, axis=-1)

        # Word embeddings
        title_w = self.dropout(self.embedding_layer(title_tokens), training=training)
        body_w = self.dropout(self.embedding_layer(abstract_tokens), training=training)

        # Positional encoding
        title_p = self.title_pos_encoder(title_w, training=training)
        body_p = self.body_pos_encoder(body_w, training=training)

        # MAB encoding
        title_t = self.title_mab(title_p, training=training)
        body_t = self.body_mab(body_p, training=training)

        # Mean pooling
        title_embedding = ops.mean(title_t, axis=1)
        body_embedding = ops.mean(body_t, axis=1)

        # Category-aware intent disentanglement
        cat_embed = self.category_embedding(category_id)
        subcat_embed = self.subcategory_embedding(subcategory_id)
        category_concat = ops.concatenate([cat_embed, subcat_embed], axis=1)
        category_representation = self.category_affine(category_concat)

        # Concatenate with category information
        category_aware_title = ops.concatenate(
            [title_embedding, category_representation], axis=1
        )
        category_aware_body = ops.concatenate(
            [body_embedding, category_representation], axis=1
        )

        # K-intent disentanglement
        title_k_intents = self.k_intent_disentangle(category_aware_title)
        body_k_intents = self.k_intent_disentangle(category_aware_body)

        # Intent-based attention
        title_intent_embedding = self.title_intent_attention(title_k_intents)
        body_intent_embedding = self.body_intent_attention(body_k_intents)

        # Category predictor (auxiliary task)
        category_logits = None
        if return_category_logits:
            category_logits = self.category_predictor(title_intent_embedding)

        # Title-body similarity computation
        similarity = self.similarity_compute(
            title_intent_embedding, body_intent_embedding
        )
        similarity_expanded = ops.expand_dims(similarity, axis=1)

        # Consistency-based representation
        news_representation = ops.concatenate(
            [title_intent_embedding, similarity_expanded * body_intent_embedding],
            axis=1,
        )

        # Add category and subcategory embeddings
        news_representation = ops.concatenate(
            [
                news_representation,
                self.dropout(cat_embed, training=training),
                self.dropout(subcat_embed, training=training),
            ],
            axis=1,
        )

        if return_category_logits:
            # Return tuple for auxiliary loss computation
            return news_representation, category_logits, category_id
        else:
            # Return only news representation for regular forward pass
            return news_representation


class UserEncoder(keras.Model):
    """User encoder component for CROWN model.

    Implements GNN-enhanced hybrid user representation using GraphSAGE.
    """

    def __init__(
        self, config: CROWNConfig, news_encoder: NewsEncoder, name: str = "user_encoder"
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

        # TimeDistributed layer for processing history
        self.time_distributed = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_user"
        )

        # TimeDistributed layer for processing history with auxiliary outputs
        self.time_distributed_aux = TimeDistributedAux(
            self.news_encoder, name="td_news_encoder_user_aux"
        )

        # User-level attention (public for direct access when needed)
        self.user_attention = AdditiveAttention(
            self.config.attention_dim,
            seed=self.config.seed,
            name="user_additive_attention",
        )

        # GNN layer selection based on config
        if self.config.gnn_type == "gat":
            self.gnn_layer = GraphAttentionLayer(
                units=self.config.graph_hidden_dim,
                num_heads=self.config.gat_num_heads,
                dropout_rate=self.config.dropout_rate,
                activation="relu",
                seed=self.config.seed,
                alpha=self.config.gat_alpha,
                concat_heads=self.config.gat_concat_heads,
                name="gnn_gat_layer",
            )
        elif self.config.gnn_type == "graphsage":
            self.gnn_layer = GraphSAGELayer(
                units=self.config.graph_hidden_dim,
                aggregator=self.config.sage_aggregator,
                dropout_rate=self.config.dropout_rate,
                activation="relu",
                seed=self.config.seed,
                normalize=self.config.sage_normalize,
                name="gnn_sage_layer",
            )
        else:
            raise ValueError(
                f"Unknown GNN type: {self.config.gnn_type}. Choose 'graphsage' or 'gat'."
            )

        # User node initialization layer for creating initial user representations
        self.user_node_init = layers.Dense(
            self.config.graph_hidden_dim,
            kernel_initializer="glorot_uniform",
            name="user_node_init",
        )

        # News projection layer for graph processing
        self.news_graph_projection = layers.Dense(
            self.config.graph_hidden_dim,
            kernel_initializer="glorot_uniform",
            name="news_graph_projection",
        )

        # Final user representation projection
        # Combines GNN (graph_hidden_dim) + attention (news_embedding_dim) features
        news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )
        self.final_user_projection = layers.Dense(
            news_embedding_dim,
            kernel_initializer="glorot_uniform",
            name="final_user_projection",
        )

    def build(self, input_shape):
        super().build(input_shape)

    def create_bipartite_graph(self, user_history_mask):
        """Create adjacency matrix for user-news bipartite graph.

        Creates a simpler bipartite graph where each user (single node per batch)
        connects to their valid history nodes. This is more memory efficient.

        Args:
            user_history_mask: (batch_size, max_history_num)

        Returns:
            adjacency_matrix: (batch_size, max_history_num + 1, max_history_num + 1)
        """
        # Create adjacency matrix parts
        # Top-left: history-to-history (no connections)
        hist_to_hist = ops.zeros_like(
            ops.expand_dims(user_history_mask, axis=-1)
            @ ops.expand_dims(user_history_mask, axis=-2)
        )

        # Top-right: history-to-user connections based on mask
        hist_to_user = ops.expand_dims(user_history_mask, axis=-1)

        # Bottom-left: user-to-history connections based on mask
        user_to_hist = ops.expand_dims(user_history_mask, axis=1)

        # Bottom-right: user-to-user (no self-loop)
        # Create a zeros tensor with shape (batch_size, 1, 1) using broadcasting
        user_to_user = ops.zeros_like(
            ops.expand_dims(user_history_mask[:, :1], axis=-1)
        )

        # Combine all parts
        top_half = ops.concatenate([hist_to_hist, hist_to_user], axis=2)
        bottom_half = ops.concatenate([user_to_hist, user_to_user], axis=2)
        adjacency = ops.concatenate([top_half, bottom_half], axis=1)

        return adjacency

    def gnn_enhanced_user_representation(
        self, news_embeddings, history_mask, training=None
    ):
        """Compute GNN-enhanced hybrid user representation using proper GraphSAGE/GAT.

        According to CROWN paper, this implements:
        1. Bipartite graph construction between users and news
        2. Graph neural network processing with mutual updates using GraphSAGE or GAT
        3. Hybrid representation combining attention and graph features

        Args:
            news_embeddings: News embeddings from history (batch_size, history_length, embedding_dim)
            history_mask: Mask for valid history items (batch_size, history_length)
            training: Whether in training mode

        Returns:
            Tuple of (enhanced_user_embedding, enhanced_news_embeddings)
        """
        # Step 1: Initialize user representation from news mean pooling
        # This follows the paper's approach of initializing user nodes
        mask_expanded = ops.expand_dims(ops.cast(history_mask, "float32"), axis=-1)
        masked_news = news_embeddings * mask_expanded

        # Compute mean over valid news items
        sum_news = ops.sum(masked_news, axis=1)
        count_news = ops.sum(mask_expanded, axis=1)
        count_news = ops.maximum(count_news, 1.0)  # Avoid division by zero

        user_initial = sum_news / count_news  # (batch_size, embedding_dim)

        # Step 2: Project to graph space
        user_graph_features = self.user_node_init(
            user_initial
        )  # (batch_size, graph_hidden_dim)
        user_graph_features = ops.expand_dims(
            user_graph_features, axis=1
        )  # (batch_size, 1, graph_hidden_dim)

        news_graph_features = self.news_graph_projection(news_embeddings)
        # (batch_size, history_length, graph_hidden_dim)

        # Step 3: Create bipartite adjacency matrix
        # For CROWN, we create connections between each user and their clicked news items
        adjacency = self.create_bipartite_graph(history_mask)

        # Step 4: Apply GNN layer (GraphSAGE or GAT)
        # The GNN layer expects (user_features, news_features, adjacency_matrix)
        enhanced_user_emb, enhanced_news_embs = self.gnn_layer(
            (user_graph_features, news_graph_features, adjacency), training=training
        )

        # Remove the extra dimension from user embeddings
        enhanced_user_emb = ops.squeeze(
            enhanced_user_emb, axis=1
        )  # (batch_size, graph_hidden_dim)

        return enhanced_user_emb, enhanced_news_embs

    def call(self, inputs, training=None, return_aux=False):
        """Forward pass for user encoding.

        Args:
            inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            training: Whether in training mode
            return_aux: Whether to return auxiliary outputs for loss computation

        Returns:
            If return_aux is False:
                User representations (batch_size, news_embedding_dim)
            If return_aux is True:
                Tuple of (user_representations, category_logits, category_ids)
        """
        if return_aux:
            # Apply news encoder with auxiliary outputs to each news item in history
            hist_outputs = self.time_distributed_aux(inputs, training=training)

            if isinstance(hist_outputs, tuple):
                clicked_news, hist_cat_logits, hist_cat_ids = hist_outputs
            else:
                clicked_news = hist_outputs
                hist_cat_logits = None
                hist_cat_ids = None

            # Create mask for history items (2D: batch_size, history_len)
            history_mask = ops.any(ops.not_equal(inputs, 0), axis=-1)

            # GNN-Enhanced Hybrid User Representation (CROWN paper)
            # Uses efficient bipartite graph approach
            try:
                enhanced_user_emb, enhanced_news_embs = (
                    self.gnn_enhanced_user_representation(
                        clicked_news, history_mask, training=training
                    )
                )

                # Attention-based user representation on enhanced news
                attention_user_repr = self.user_attention(
                    enhanced_news_embs, mask=history_mask
                )

                # Hybrid representation: combine GNN and attention features
                user_representation = ops.concatenate(
                    [enhanced_user_emb, attention_user_repr], axis=-1
                )

                # Project to final dimension
                user_representation = self.final_user_projection(user_representation)
            except Exception as e:
                # Fallback to standard attention if GNN fails
                print(f"GNN fallback activated: {e}")
                user_representation = self.user_attention(
                    clicked_news, mask=history_mask
                )

            return user_representation, hist_cat_logits, hist_cat_ids
        else:
            # Apply news encoder to each news item in history
            clicked_news = self.time_distributed(inputs, training=training)

            # Create mask for history items (2D: batch_size, history_len)
            history_mask = ops.any(ops.not_equal(inputs, 0), axis=-1)

            # GNN-Enhanced Hybrid User Representation (CROWN paper)
            try:
                enhanced_user_emb, enhanced_news_embs = (
                    self.gnn_enhanced_user_representation(
                        clicked_news, history_mask, training=training
                    )
                )

                # Attention-based user representation on enhanced news
                attention_user_repr = self.user_attention(
                    enhanced_news_embs, mask=history_mask
                )

                # Hybrid representation: combine GNN and attention features
                user_representation = ops.concatenate(
                    [enhanced_user_emb, attention_user_repr], axis=-1
                )

                # Project to final dimension
                user_representation = self.final_user_projection(user_representation)
            except Exception as e:
                # Fallback to standard attention if GNN fails
                print(f"GNN fallback activated: {e}")
                user_representation = self.user_attention(
                    clicked_news, mask=history_mask
                )

            return user_representation


class TimeDistributedAux(layers.Layer):
    """TimeDistributed layer that can return auxiliary outputs."""

    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, training=None):
        """Apply layer to each timestep, returning auxiliary outputs if available."""
        batch_size = ops.shape(inputs)[0]
        num_timesteps = ops.shape(inputs)[1]

        # Reshape input to process all timesteps at once
        input_shape = ops.shape(inputs)
        reshaped_inputs = ops.reshape(inputs, (-1, input_shape[-1]))

        # Process with auxiliary outputs
        outputs = self.layer(
            reshaped_inputs, training=training, return_category_logits=True
        )

        if isinstance(outputs, tuple):
            news_reprs, category_logits, category_ids = outputs

            # Reshape news representations back to time-distributed format
            news_reprs = ops.reshape(news_reprs, (batch_size, num_timesteps, -1))

            return news_reprs, category_logits, category_ids
        else:
            # Reshape back for single output
            return ops.reshape(outputs, (batch_size, num_timesteps, -1))


class CROWNScorer(keras.Model):
    """Scoring component for CROWN model."""

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: NewsEncoder,
        user_encoder: UserEncoder,
        name: str = "crown_scorer",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

        # Store TimeDistributed layers for candidate processing
        self.candidate_encoder_train = TimeDistributedAux(
            self.news_encoder, name="td_news_encoder_candidates"
        )
        self.candidate_encoder_eval = layers.TimeDistributed(
            self.news_encoder, name="td_news_encoder_eval"
        )

    def score_training_batch(
        self, history_inputs, candidate_inputs, training=None, return_aux=False
    ):
        """Score training batch with softmax output.

        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidates tensor (batch_size, num_candidates, feature_size)
            training: Whether in training mode
            return_aux: Whether to return auxiliary outputs for loss computation

        Returns:
            If return_aux is False:
                Softmax scores (batch_size, num_candidates)
            If return_aux is True:
                Tuple of (softmax_scores, all_category_logits, all_category_ids)
        """
        if return_aux:
            # Get user representation and history auxiliary outputs
            user_outputs = self.user_encoder(
                history_inputs, training=True, return_aux=True
            )

            if isinstance(user_outputs, tuple):
                user_repr, hist_cat_logits, hist_cat_ids = user_outputs
            else:
                user_repr = user_outputs
                hist_cat_logits = None
                hist_cat_ids = None

            # Get representations and auxiliary outputs for all candidates
            candidate_outputs = self.candidate_encoder_train(
                candidate_inputs, training=True, return_category_logits=True
            )

            if isinstance(candidate_outputs, tuple):
                cand_repr, cand_cat_logits, cand_cat_ids = candidate_outputs
            else:
                # Fallback if auxiliary outputs not available
                cand_repr = candidate_outputs
                cand_cat_logits = None
                cand_cat_ids = None

            # Expand user representation for broadcasting
            user_repr_expanded = ops.expand_dims(user_repr, axis=1)

            # Calculate scores using dot product
            scores = ops.sum(cand_repr * user_repr_expanded, axis=-1)

            # Apply softmax for training
            output = ops.softmax(scores, axis=-1)

            # Combine auxiliary outputs
            if hist_cat_logits is not None and cand_cat_logits is not None:
                all_cat_logits = ops.concatenate(
                    [hist_cat_logits, cand_cat_logits], axis=0
                )
                all_cat_ids = ops.concatenate([hist_cat_ids, cand_cat_ids], axis=0)
                return output, all_cat_logits, all_cat_ids
            else:
                return output
        else:
            # Regular training without auxiliary outputs
            user_repr = self.user_encoder(history_inputs, training=True)

            # Get candidate representations (without auxiliary outputs)
            cand_outputs = self.candidate_encoder_train(candidate_inputs, training=True)
            if isinstance(cand_outputs, tuple):
                cand_repr = cand_outputs[0]  # Just take the representations
            else:
                cand_repr = cand_outputs

            # Expand user representation for broadcasting
            user_repr_expanded = ops.expand_dims(user_repr, axis=1)

            # Calculate scores using dot product
            scores = ops.sum(cand_repr * user_repr_expanded, axis=-1)

            # Apply softmax for training
            return ops.softmax(scores, axis=-1)

    def score_single_candidate(self, history_inputs, candidate_inputs, training=None):
        """Score single candidate with sigmoid output.

        Args:
            history_inputs: Concatenated history tensor (batch_size, history_length, feature_size)
            candidate_inputs: Concatenated candidate tensor (batch_size, feature_size)
            training: Whether in training mode

        Returns:
            Sigmoid scores (batch_size, 1)
        """
        # Encode candidate
        candidate_repr = self.news_encoder(candidate_inputs, training=False)
        candidate_repr = ops.expand_dims(candidate_repr, axis=1)

        # Encode user
        user_repr = self.user_encoder(history_inputs, training=False)

        # Calculate score
        score = ops.sum(user_repr * ops.squeeze(candidate_repr, axis=1), axis=-1)

        # Apply sigmoid
        return ops.sigmoid(score)

    def score_multiple_candidates(
        self, history_inputs, candidate_inputs, training=None
    ):
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
        candidate_reprs = self.candidate_encoder_eval(
            candidate_inputs, training=False, return_category_logits=False
        )
        # Result: (batch_size, num_candidates, cnn_filter_num)

        # Expand user representation for broadcasting
        user_repr_expanded = ops.expand_dims(user_repr, axis=1)

        # Calculate scores using dot product
        scores = ops.sum(candidate_reprs * user_repr_expanded, axis=-1)

        # Apply sigmoid activation for consistency with single candidate scoring
        output = ops.sigmoid(scores)
        return output


class CROWN(BaseModel):
    """CROWN: A Novel Approach to Comprehending Users' Preferences
    for Accurate Personalized News Recommendation.

    This model implements:
    1. Category-guided intent disentanglement
    2. Consistency-based news representation
    3. GNN-enhanced hybrid user representation
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        # Model dimensions
        embedding_size: int = 300,
        intent_embedding_dim: int = 200,
        category_embedding_dim: int = 100,
        subcategory_embedding_dim: int = 100,
        attention_dim: int = 200,
        # Intent disentanglement
        intent_num: int = 4,
        alpha: float = 0.5,
        # MAB parameters
        num_heads: int = 12,
        head_dim: int = 25,
        feedforward_dim: int = 512,
        num_layers: int = 2,
        # GNN parameters
        gnn_type: str = "graphsage",
        graph_hidden_dim: int = 300,
        graph_num_layers: int = 1,
        # GAT-specific parameters
        gat_num_heads: int = 4,
        gat_alpha: float = 0.2,
        gat_concat_heads: bool = True,
        # GraphSAGE-specific parameters
        sage_aggregator: str = "mean",
        sage_normalize: bool = True,
        # Common parameters
        dropout_rate: float = 0.2,
        seed: int = 42,
        # Input parameters
        max_title_length: int = 50,
        max_abstract_length: int = 100,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        process_user_id: bool = False,
        name: str = "crown",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Create configuration
        self.config = CROWNConfig(
            embedding_size=embedding_size,
            intent_embedding_dim=intent_embedding_dim,
            category_embedding_dim=category_embedding_dim,
            subcategory_embedding_dim=subcategory_embedding_dim,
            attention_dim=attention_dim,
            intent_num=intent_num,
            alpha=alpha,
            num_heads=num_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            graph_hidden_dim=graph_hidden_dim,
            graph_num_layers=graph_num_layers,
            gat_num_heads=gat_num_heads,
            gat_alpha=gat_alpha,
            gat_concat_heads=gat_concat_heads,
            sage_aggregator=sage_aggregator,
            sage_normalize=sage_normalize,
            dropout_rate=dropout_rate,
            seed=seed,
            max_title_length=max_title_length,
            max_abstract_length=max_abstract_length,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            process_user_id=process_user_id,
        )

        # Store processed news data
        self.processed_news = processed_news
        self._validate_processed_news()

        # Set BaseModel attributes for fast evaluation
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

    def build(self, input_shape):
        """Create all model components."""
        # Create embedding layers
        self.embedding_layer = layers.Embedding(
            input_dim=self.processed_news["vocab_size"],
            output_dim=self.config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(
                self.processed_news["embeddings"]
            ),
            trainable=True,
            mask_zero=False,
            name="word_embedding",
        )

        self.category_embedding = layers.Embedding(
            input_dim=self.processed_news["num_categories"],
            output_dim=self.config.category_embedding_dim,
            embeddings_initializer="glorot_uniform",
            trainable=True,
            name="category_embedding",
        )

        self.subcategory_embedding = layers.Embedding(
            input_dim=self.processed_news["num_subcategories"],
            output_dim=self.config.subcategory_embedding_dim,
            embeddings_initializer="glorot_uniform",
            trainable=True,
            name="subcategory_embedding",
        )

        # Set category predictor with correct number of categories
        num_categories = self.processed_news.get("num_categories", 18)
        self.category_predictor = CategoryPredictor(num_categories)

        # Create encoders
        self.news_encoder = NewsEncoder(
            self.config,
            self.embedding_layer,
            self.category_embedding,
            self.subcategory_embedding,
            self.category_predictor,
        )

        self.user_encoder = UserEncoder(self.config, self.news_encoder)
        self.scorer = CROWNScorer(self.config, self.news_encoder, self.user_encoder)

        # Build compatibility models
        self.training_model, self.scorer_model = self._build_compatibility_models()

        super().build(input_shape)

    def _build_compatibility_models(self) -> tuple[keras.Model, keras.Model]:
        """Build training and scorer models for compatibility."""
        # ----- Training model -----
        # Create concatenated inputs that match the expected format
        history_concat_input = keras.Input(
            shape=(
                self.config.max_history_length,
                self.config.max_title_length + self.config.max_abstract_length + 2,
            ),
            dtype="int32",
            name="hist_concat_input",
        )
        candidate_concat_input = keras.Input(
            shape=(
                self.config.max_impressions_length,
                self.config.max_title_length + self.config.max_abstract_length + 2,
            ),
            dtype="int32",
            name="cand_concat_input",
        )

        # Pass concatenated inputs directly to the scorer
        training_output = self.scorer.score_training_batch(
            history_concat_input, candidate_concat_input
        )

        training_model = keras.Model(
            inputs=[history_concat_input, candidate_concat_input],
            outputs=training_output,
            name="crown_training_model",
        )

        # ----- Scorer model -----
        # Create concatenated inputs for scorer model
        score_history_concat_input = keras.Input(
            shape=(
                self.config.max_history_length,
                self.config.max_title_length + self.config.max_abstract_length + 2,
            ),
            dtype="int32",
            name="score_hist_concat_input",
        )
        score_candidate_concat_input = keras.Input(
            shape=(self.config.max_title_length + self.config.max_abstract_length + 2,),
            dtype="int32",
            name="score_cand_concat_input",
        )

        # Pass concatenated inputs directly to the scorer
        scorer_output = self.scorer.score_single_candidate(
            score_history_concat_input, score_candidate_concat_input
        )

        scorer_model = keras.Model(
            inputs=[score_history_concat_input, score_candidate_concat_input],
            outputs=scorer_output,
            name="crown_scorer_model",
        )

        return training_model, scorer_model

    def _validate_inputs(self, inputs: dict, training: bool = None) -> None:
        """Validate input format and shapes based on mode."""
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary")

        input_keys = set(inputs.keys())

        if training:
            # Training mode expects multiple keys for history and candidates
            required_hist_keys = {
                "hist_tokens",
                "hist_abstract_tokens",
                "hist_category",
                "hist_subcategory",
            }
            required_cand_keys = {
                "cand_tokens",
                "cand_abstract_tokens",
                "cand_category",
                "cand_subcategory",
            }

            if not (
                required_hist_keys.issubset(input_keys)
                and required_cand_keys.issubset(input_keys)
            ):
                raise ValueError(
                    f"Training mode requires history keys: {required_hist_keys} and candidate keys: {required_cand_keys}, "
                    f"but got keys: {list(input_keys)}"
                )
        else:
            # Inference mode - check for different valid combinations
            # Add validation logic for inference mode if needed
            pass

    def call(self, inputs, training=None):
        """Main forward pass of CROWN model."""
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
        """Handle training batch scoring with softmax output only."""
        # Concatenate history inputs
        history_concat = ops.concatenate(
            [
                inputs["hist_tokens"],
                inputs["hist_abstract_tokens"],
                ops.expand_dims(inputs["hist_category"], axis=-1),
                ops.expand_dims(inputs["hist_subcategory"], axis=-1),
            ],
            axis=-1,
        )

        # Concatenate candidate inputs
        candidate_concat = ops.concatenate(
            [
                inputs["cand_tokens"],
                inputs["cand_abstract_tokens"],
                ops.expand_dims(inputs["cand_category"], axis=-1),
                ops.expand_dims(inputs["cand_subcategory"], axis=-1),
            ],
            axis=-1,
        )

        # Return only click scores for standard Keras training
        # Auxiliary loss will be handled through a custom training procedure
        return self.scorer.score_training_batch(
            history_concat, candidate_concat, return_aux=False
        )

    def _handle_multiple_candidates(self, inputs):
        """Handle multiple candidate scoring with sigmoid scores."""
        # Concatenate history inputs
        history_concat = ops.concatenate(
            [
                inputs["hist_tokens"],
                inputs["hist_abstract_tokens"],
                ops.expand_dims(inputs["hist_category"], axis=-1),
                ops.expand_dims(inputs["hist_subcategory"], axis=-1),
            ],
            axis=-1,
        )

        # Concatenate candidate inputs
        candidate_concat = ops.concatenate(
            [
                inputs["cand_tokens"],
                inputs["cand_abstract_tokens"],
                ops.expand_dims(inputs["cand_category"], axis=-1),
                ops.expand_dims(inputs["cand_subcategory"], axis=-1),
            ],
            axis=-1,
        )

        return self.scorer.score_multiple_candidates(
            history_concat, candidate_concat, training=False
        )

    def get_config(self):
        """Return configuration for serialization."""
        base_config = super().get_config()
        base_config.update(
            {
                "embedding_size": self.config.embedding_size,
                "intent_embedding_dim": self.config.intent_embedding_dim,
                "category_embedding_dim": self.config.category_embedding_dim,
                "subcategory_embedding_dim": self.config.subcategory_embedding_dim,
                "attention_dim": self.config.attention_dim,
                "intent_num": self.config.intent_num,
                "alpha": self.config.alpha,
                "num_heads": self.config.num_heads,
                "head_dim": self.config.head_dim,
                "feedforward_dim": self.config.feedforward_dim,
                "num_layers": self.config.num_layers,
                "gnn_type": self.config.gnn_type,
                "graph_hidden_dim": self.config.graph_hidden_dim,
                "graph_num_layers": self.config.graph_num_layers,
                "gat_num_heads": self.config.gat_num_heads,
                "gat_alpha": self.config.gat_alpha,
                "gat_concat_heads": self.config.gat_concat_heads,
                "sage_aggregator": self.config.sage_aggregator,
                "sage_normalize": self.config.sage_normalize,
                "dropout_rate": self.config.dropout_rate,
                "seed": self.config.seed,
                "max_title_length": self.config.max_title_length,
                "max_abstract_length": self.config.max_abstract_length,
                "max_history_length": self.config.max_history_length,
                "max_impressions_length": self.config.max_impressions_length,
                "process_user_id": self.config.process_user_id,
            }
        )
        return base_config
