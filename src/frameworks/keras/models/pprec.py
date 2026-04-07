"""PP-Rec: News Recommendation with Personalized User Interest and
Time-aware News Popularity (ACL 2021).

Keras implementation following NewsReX patterns.

Components
----------
- PPRecNewsEncoder: Knowledge-aware news encoder (title MHSA + entity MHSA + category)
- PopularityPredictor: Content + recency + CTR popularity scoring
- PPRecUserEncoder: Popularity-aware user encoder with CPJA
- ActivityGater: Per-user gate balancing relevance vs. popularity
- PPRec: Main model orchestrating all components
"""

from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import PPRecConfig
from src.frameworks.keras.layers import (
    AdditiveAttention,
    AttentivePoolingQKY,
    GlorotUniformMHA,
)
from src.frameworks.keras.models.base import BaseModel

# ---------------------------------------------------------------------------
# News Encoder
# ---------------------------------------------------------------------------


class PPRecNewsEncoder(keras.Model):
    """Knowledge-aware news encoder for PP-Rec (paper ``co1`` variant).

    Encodes a news article from its title tokens, entity indices, and
    category index. Implements bidirectional cross-attention (MHCA) between
    word and entity embeddings, matching ``get_news_encoder_co1`` in the
    official PP-Rec code (Encoders.py:168-217).

    Architecture::

        word_emb  = WordEmb(title_tokens)        Dropout(0.2)
        entity_emb = EntityEmb(entities)         (trainable)
        cat_emb   = CategoryEmb(category)         Reshape -> Dropout

        # Bidirectional MHCA on RAW embeddings (before MHSA)
        title_co  = MHCA(5,40)([title_emb, entity_emb, entity_emb])   # 200d
        entity_co = MHCA(5,40)([entity_emb, title_emb, title_emb])    # 200d

        # MHSA + concat with co-attention output, project, pool
        title_vecs  = MHSA(20,20)([title_emb])                        # 400d
        title_vecs  = Concat([title_vecs, title_co])                  # 600d
        title_vecs  = Dense(400)(title_vecs); Dropout(0.2)            # 400d
        title_vec   = AdditiveAttention(title_vecs)                   # 400d

        entity_vecs = MHSA(20,20)([entity_emb])                       # 400d
        entity_vecs = Concat([entity_vecs, entity_co])                # 600d
        entity_vecs = Dense(400)(entity_vecs); Dropout(0.2)           # 400d
        entity_vec  = AdditiveAttention(entity_vecs)                  # 400d

        news_vec    = Dense(400)(Concat([title_vec, entity_vec, cat_vec]))
    """

    # Number of heads / per-head dim for the cross-attention. Paper uses 5x40.
    CO_NUM_HEADS = 5
    CO_HEAD_DIM = 40

    def __init__(
        self,
        config: PPRecConfig,
        word_embedding_layer: layers.Embedding,
        entity_embedding_layer: layers.Embedding | None = None,
        category_embedding_layer: layers.Embedding | None = None,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config

        self.word_embedding = word_embedding_layer
        self.entity_embedding = entity_embedding_layer
        self.category_embedding = category_embedding_layer

        # --- Word (title) branch ---
        self.word_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.word_mhsa = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name=f"{name}_word_mhsa",
        )
        self.word_proj = layers.Dense(
            config.news_dim,
            kernel_initializer=keras.initializers.GlorotUniform(),
            name=f"{name}_word_proj",
        )
        self.word_dropout2 = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.word_attention = AdditiveAttention(
            query_vec_dim=config.attention_hidden_dim,
            seed=config.seed,
            name=f"{name}_word_additive",
        )

        # --- Entity branch (optional) + bidirectional MHCA ---
        if entity_embedding_layer is not None:
            self.entity_mhsa = layers.MultiHeadAttention(
                num_heads=config.num_heads,
                key_dim=config.head_dim,
                dropout=config.dropout_rate,
                kernel_initializer=GlorotUniformMHA(),
                name=f"{name}_entity_mhsa",
            )
            self.entity_proj = layers.Dense(
                config.news_dim,
                kernel_initializer=keras.initializers.GlorotUniform(),
                name=f"{name}_entity_proj",
            )
            self.entity_dropout = layers.Dropout(
                config.dropout_rate, seed=config.seed
            )
            self.entity_attention = AdditiveAttention(
                query_vec_dim=config.attention_hidden_dim,
                seed=config.seed,
                name=f"{name}_entity_additive",
            )
            # Word-conditioned-on-entity cross-attention (Q=title, KV=entities)
            self.title_mhca = layers.MultiHeadAttention(
                num_heads=self.CO_NUM_HEADS,
                key_dim=self.CO_HEAD_DIM,
                dropout=config.dropout_rate,
                kernel_initializer=GlorotUniformMHA(),
                name=f"{name}_title_mhca",
            )
            # Entity-conditioned-on-word cross-attention (Q=entities, KV=title)
            self.entity_mhca = layers.MultiHeadAttention(
                num_heads=self.CO_NUM_HEADS,
                key_dim=self.CO_HEAD_DIM,
                dropout=config.dropout_rate,
                kernel_initializer=GlorotUniformMHA(),
                name=f"{name}_entity_mhca",
            )

        # --- Category branch (optional) ---
        if category_embedding_layer is not None:
            self.category_dropout = layers.Dropout(
                config.dropout_rate, seed=config.seed
            )

        # --- Fusion: [title_vec, entity_vec, category_vec] -> Dense(news_dim) ---
        self.fusion_dense = layers.Dense(
            config.news_dim,
            kernel_initializer=keras.initializers.GlorotUniform(),
            name=f"{name}_fusion",
        )

    def call(self, inputs, training=None):
        """Encode news features into a news vector.

        Args:
            inputs: Tensor of shape ``(batch, feature_dim)`` where the
                feature dim is the concatenation of
                ``[title_tokens, entity_indices, category_index]`` (entity
                and category are optional, controlled by the config).

        Returns:
            News vector of shape ``(batch, news_dim)``.
        """
        offset = 0
        title_len = self.config.max_title_length

        # --- Slice features ---
        title_tokens = inputs[:, offset : offset + title_len]
        offset += title_len
        title_mask = ops.not_equal(title_tokens, 0)

        has_entity = self.entity_embedding is not None and self.config.use_entity
        if has_entity:
            entity_len = self.config.max_entities
            entity_indices = inputs[:, offset : offset + entity_len]
            offset += entity_len
            entity_mask = ops.not_equal(entity_indices, 0)
        else:
            entity_indices = None
            entity_mask = None

        has_category = self.category_embedding is not None
        if has_category:
            category_idx = inputs[:, offset : offset + 1]
            offset += 1
        else:
            category_idx = None

        # --- Raw embeddings ---
        title_emb = self.word_dropout(self.word_embedding(title_tokens), training=training)
        if has_entity:
            entity_emb = self.entity_embedding(entity_indices)

        # --- Bidirectional MHCA on RAW embeddings (paper co1) ---
        if has_entity:
            title_co = self.title_mhca(
                title_emb, entity_emb, entity_emb,
                key_mask=entity_mask, value_mask=entity_mask,
                training=training,
            )
            entity_co = self.entity_mhca(
                entity_emb, title_emb, title_emb,
                key_mask=title_mask, value_mask=title_mask,
                training=training,
            )

        # --- Title self-attention + concat with co-attention ---
        title_vecs = self.word_mhsa(
            title_emb, title_emb, title_emb,
            key_mask=title_mask, value_mask=title_mask,
            training=training,
        )
        if has_entity:
            title_vecs = ops.concatenate([title_vecs, title_co], axis=-1)
        title_vecs = self.word_proj(title_vecs)
        title_vecs = self.word_dropout2(title_vecs, training=training)
        title_vec = self.word_attention(title_vecs, mask=title_mask)

        vecs = [title_vec]

        # --- Entity self-attention + concat with co-attention ---
        if has_entity:
            entity_vecs = self.entity_mhsa(
                entity_emb, entity_emb, entity_emb,
                key_mask=entity_mask, value_mask=entity_mask,
                training=training,
            )
            entity_vecs = ops.concatenate([entity_vecs, entity_co], axis=-1)
            entity_vecs = self.entity_proj(entity_vecs)
            entity_vecs = self.entity_dropout(entity_vecs, training=training)
            entity_vec = self.entity_attention(entity_vecs, mask=entity_mask)
            vecs.append(entity_vec)

        # --- Category branch ---
        if has_category:
            cat_vec = self.category_embedding(category_idx)
            cat_vec = ops.squeeze(cat_vec, axis=1)
            cat_vec = self.category_dropout(cat_vec, training=training)
            vecs.append(cat_vec)

        # --- Fusion ---
        if len(vecs) > 1:
            fused = ops.concatenate(vecs, axis=-1)
        else:
            fused = vecs[0]
        return self.fusion_dense(fused)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.config.news_dim)


# ---------------------------------------------------------------------------
# Popularity Predictor
# ---------------------------------------------------------------------------


class PopularityPredictor(keras.Model):
    """Time-aware news popularity predictor.

    Combines content-based popularity, recency-aware popularity (via a learned
    gate), and a CTR scaler.
    """

    def __init__(self, config: PPRecConfig, name: str = "popularity_predictor"):
        super().__init__(name=name)
        self.config = config
        seed = config.seed

        # Content scorer: news_dim -> 256 -> 256 -> 128 -> 1
        self.content_dense1 = layers.Dense(
            256,
            activation="tanh",
            name="content_d1",
            kernel_initializer=keras.initializers.GlorotUniform(),
        )
        self.content_dense2 = layers.Dense(
            256,
            activation="tanh",
            name="content_d2",
            kernel_initializer=keras.initializers.GlorotUniform(),
        )
        self.content_dense3 = layers.Dense(
            128,
            name="content_d3",
            kernel_initializer=keras.initializers.GlorotUniform(),
        )
        self.content_out = layers.Dense(
            1,
            use_bias=False,
            name="content_out",
            kernel_initializer=keras.initializers.GlorotUniform(),
        )

        # Recency scorer: recency_emb_dim -> 64 -> 64 -> 1
        if config.use_recency:
            self.recency_embedding = layers.Embedding(
                config.recency_embedding_bins,
                config.recency_embedding_dim,
                name="recency_embedding",
            )
            self.recency_dense1 = layers.Dense(
                64,
                activation="tanh",
                name="recency_d1",
                kernel_initializer=keras.initializers.GlorotUniform(),
            )
            self.recency_dense2 = layers.Dense(
                64,
                activation="tanh",
                name="recency_d2",
                kernel_initializer=keras.initializers.GlorotUniform(),
            )
            self.recency_out = layers.Dense(
                1,
                use_bias=False,
                name="recency_out",
                kernel_initializer=keras.initializers.GlorotUniform(),
            )
            # Gate: concat(news_vec, recency_emb) -> 128 -> 64 -> 1(sigmoid)
            self.gate_dense1 = layers.Dense(
                128,
                activation="tanh",
                name="gate_d1",
                kernel_initializer=keras.initializers.GlorotUniform(),
            )
            self.gate_dense2 = layers.Dense(
                64,
                activation="tanh",
                name="gate_d2",
                kernel_initializer=keras.initializers.GlorotUniform(),
            )
            self.gate_out = layers.Dense(
                1,
                activation="sigmoid",
                name="gate_out",
                kernel_initializer=keras.initializers.GlorotUniform(),
            )

        # CTR scaler: single learned weight initialized to ctr_scaler_init
        if config.use_ctr:
            self.ctr_scaler = layers.Dense(
                1,
                use_bias=False,
                name="ctr_scaler",
                kernel_initializer=keras.initializers.Constant(config.ctr_scaler_init),
            )

    def call(self, bias_news_vec, recency_indices=None, ctr_values=None, training=None):
        """Compute popularity score for candidate news.

        Args:
            bias_news_vec: (batch, news_dim) from bias news encoder.
            recency_indices: (batch,) int recency bucket indices.
            ctr_values: (batch,) float CTR values.

        Returns:
            Popularity score: (batch,) float.
        """
        # Content-based popularity
        x = self.content_dense1(bias_news_vec)
        x = self.content_dense2(x)
        x = self.content_dense3(x)
        content_score = ops.squeeze(self.content_out(x), axis=-1)

        pop_score = content_score

        # Recency-aware gating
        if self.config.use_recency and recency_indices is not None:
            recency_emb = self.recency_embedding(recency_indices)
            r = self.recency_dense1(recency_emb)
            r = self.recency_dense2(r)
            recency_score = ops.squeeze(self.recency_out(r), axis=-1)

            gate_input = ops.concatenate([bias_news_vec, recency_emb], axis=-1)
            g = self.gate_dense1(gate_input)
            g = self.gate_dense2(g)
            gate = ops.squeeze(self.gate_out(g), axis=-1)

            pop_score = (1.0 - gate) * content_score + gate * recency_score

        # CTR component
        ctr_score = 0.0
        if self.config.use_ctr and ctr_values is not None:
            ctr_input = ops.expand_dims(ctr_values, axis=-1)
            ctr_score = ops.squeeze(self.ctr_scaler(ctr_input), axis=-1)

        return pop_score + ctr_score


# ---------------------------------------------------------------------------
# User Encoder
# ---------------------------------------------------------------------------


class PPRecUserEncoder(keras.Model):
    """Popularity-aware user encoder with Content-Popularity Joint Attention.

    Encodes user click history using the relevance news encoder, then applies
    MHSA and CPJA (where attention keys are the concatenation of contextual
    news vectors and popularity embeddings, but the weighted sum only
    aggregates the news vectors).
    """

    def __init__(
        self,
        config: PPRecConfig,
        news_encoder: PPRecNewsEncoder,
        name: str = "user_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder

        self.td_news_encoder = layers.TimeDistributed(
            news_encoder, name="td_user_news_encoder"
        )
        self.user_mhsa = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="user_mhsa",
        )

        # Popularity embedding for CPJA
        self.popularity_embedding = layers.Embedding(
            config.popularity_embedding_bins,
            config.popularity_embedding_dim,
            name="popularity_embedding",
        )
        # CPJA: attention on concat(news_vec, pop_emb), weighted sum of news_vec
        self.cpja = AttentivePoolingQKY(
            query_vec_dim=config.attention_hidden_dim,
            seed=config.seed,
            name="cpja_attention",
        )

    def call(self, inputs, training=None):
        """Encode user history into a user vector.

        Args:
            inputs: List of [history_features, history_ctr].
                history_features: (batch, history_len, feature_dim) news features.
                history_ctr: (batch, history_len) discretized int CTR values.

        Returns:
            User vector: (batch, news_dim).
        """
        if isinstance(inputs, (list, tuple)):
            history_features, history_ctr = inputs
        else:
            # Fallback: no CTR data, use zeros
            history_features = inputs
            history_ctr = ops.zeros(
                (ops.shape(history_features)[0], ops.shape(history_features)[1]),
                dtype="int32",
            )

        # Encode each news in history
        user_vecs = self.td_news_encoder(history_features, training=training)

        # History mask (detect padded items)
        history_mask = ops.any(ops.not_equal(history_features, 0), axis=-1)

        # Self-attention over clicked news
        user_vecs = self.user_mhsa(
            user_vecs,
            user_vecs,
            user_vecs,
            key_mask=history_mask,
            value_mask=history_mask,
            training=training,
        )

        # Popularity embeddings from discretized CTR
        pop_emb = self.popularity_embedding(history_ctr)

        # CPJA: concat [user_vecs, pop_emb] as key, user_vecs as value
        key_input = ops.concatenate([user_vecs, pop_emb], axis=-1)
        user_vec = self.cpja([key_input, user_vecs], mask=history_mask)

        return user_vec

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.config.news_dim)


# ---------------------------------------------------------------------------
# Activity Gater
# ---------------------------------------------------------------------------


class ActivityGater(keras.Model):
    """Per-user gate that balances relevance vs. popularity.

    Maps user vector to a scalar gate eta in [0, 1]:
        eta close to 1 -> rely on personalized relevance
        eta close to 0 -> rely on popularity
    """

    def __init__(self, config: PPRecConfig, name: str = "activity_gater"):
        super().__init__(name=name)
        self.dense1 = layers.Dense(
            64,
            activation="tanh",
            name="gate_d1",
            kernel_initializer=keras.initializers.GlorotUniform(),
        )
        self.dense2 = layers.Dense(
            1,
            activation="sigmoid",
            name="gate_d2",
            kernel_initializer=keras.initializers.GlorotUniform(),
        )

    def call(self, user_vec, training=None):
        """Compute activity gate.

        Args:
            user_vec: (batch, news_dim) user representation.

        Returns:
            Gate value eta: (batch,) in [0, 1].
        """
        x = self.dense1(user_vec)
        return ops.squeeze(self.dense2(x), axis=-1)


# ---------------------------------------------------------------------------
# Main PP-Rec Model
# ---------------------------------------------------------------------------


class PPRec(BaseModel):
    """PP-Rec: News Recommendation with Personalized User Interest and
    Time-aware News Popularity.

    Two separate news encoders (relevance + bias/popularity), a popularity
    predictor, a popularity-aware user encoder, and an activity gate that
    balances personalized matching with popularity scoring.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        embedding_size: int = 300,
        news_dim: int = 400,
        entity_embedding_dim: int = 100,
        category_embedding_dim: int = 200,
        num_heads: int = 20,
        head_dim: int = 20,
        attention_hidden_dim: int = 200,
        popularity_embedding_bins: int = 200,
        popularity_embedding_dim: int = 400,
        recency_embedding_bins: int = 1500,
        recency_embedding_dim: int = 100,
        ctr_scaler_init: float = 19.0,
        use_entity: bool = True,
        use_recency: bool = True,
        use_ctr: bool = True,
        use_activity_gate: bool = True,
        dropout_rate: float = 0.2,
        seed: int = 42,
        max_title_length: int = 30,
        max_history_length: int = 50,
        max_impressions_length: int = 2,
        max_entities: int = 5,
        process_user_id: bool = False,
        name: str = "pprec",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.config = PPRecConfig(
            embedding_size=embedding_size,
            news_dim=news_dim,
            entity_embedding_dim=entity_embedding_dim,
            category_embedding_dim=category_embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_hidden_dim=attention_hidden_dim,
            popularity_embedding_bins=popularity_embedding_bins,
            popularity_embedding_dim=popularity_embedding_dim,
            recency_embedding_bins=recency_embedding_bins,
            recency_embedding_dim=recency_embedding_dim,
            ctr_scaler_init=ctr_scaler_init,
            use_entity=use_entity,
            use_recency=use_recency,
            use_ctr=use_ctr,
            use_activity_gate=use_activity_gate,
            dropout_rate=dropout_rate,
            seed=seed,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            max_entities=max_entities,
            process_user_id=process_user_id,
        )

        self.processed_news = processed_news
        self._validate_processed_news()
        self.process_user_id = process_user_id

        # Compute news feature dimension for concatenated inputs
        self._news_feature_dim = max_title_length
        if use_entity and "entity_indices" in processed_news:
            self._news_feature_dim += max_entities
        if "num_categories" in processed_news:
            self._news_feature_dim += 1  # category index

        # Will be set in build
        self.news_encoder = None
        self.bias_news_encoder = None
        self.user_encoder = None
        self.popularity_predictor = None
        self.activity_gater = None

        self.build(None)

    def build(self, input_shape) -> None:
        cfg = self.config
        pn = self.processed_news

        # Shared word embedding (used by both news encoders)
        word_emb = layers.Embedding(
            input_dim=pn["vocab_size"],
            output_dim=cfg.embedding_size,
            embeddings_initializer=keras.initializers.Constant(pn["embeddings"]),
            trainable=True,
            name="word_embedding",
        )

        # Entity embedding (shared across both encoders)
        entity_emb = None
        if cfg.use_entity and "entity_embeddings" in pn:
            entity_emb = layers.Embedding(
                input_dim=pn["entity_vocab_size"],
                output_dim=cfg.entity_embedding_dim,
                embeddings_initializer=keras.initializers.Constant(
                    pn["entity_embeddings"]
                ),
                # trainable per the paper's co1 variant: entity embeddings
                # are fine-tuned during PP-Rec training (Encoders.py:194).
                trainable=True,
                name="entity_embedding",
            )

        # Category embedding (shared)
        cat_emb = None
        if "num_categories" in pn:
            cat_emb = layers.Embedding(
                input_dim=pn["num_categories"] + 1,
                output_dim=cfg.category_embedding_dim,
                name="category_embedding",
            )

        # Relevance news encoder
        self.news_encoder = PPRecNewsEncoder(
            cfg, word_emb, entity_emb, cat_emb, name="relevance_news_encoder"
        )

        # Bias news encoder (separate word embedding, shared entity/category)
        bias_word_emb = layers.Embedding(
            input_dim=pn["vocab_size"],
            output_dim=cfg.embedding_size,
            embeddings_initializer=keras.initializers.Constant(pn["embeddings"]),
            trainable=True,
            name="bias_word_embedding",
        )
        self.bias_news_encoder = PPRecNewsEncoder(
            cfg, bias_word_emb, entity_emb, cat_emb, name="bias_news_encoder"
        )

        # User encoder (uses relevance news encoder)
        self.user_encoder = PPRecUserEncoder(cfg, self.news_encoder)

        # Popularity predictor
        self.popularity_predictor = PopularityPredictor(cfg)

        # Activity gater
        if cfg.use_activity_gate:
            self.activity_gater = ActivityGater(cfg)

        # TimeDistributed for candidates
        self._td_rel = layers.TimeDistributed(
            self.news_encoder, name="td_rel_candidates"
        )
        self._td_bias = layers.TimeDistributed(
            self.bias_news_encoder, name="td_bias_candidates"
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        if training:
            return self.score_training_batch(inputs, training=training)
        else:
            return self.score_candidates(inputs, training=False)

    def score_training_batch(self, inputs, training=None):
        """Score a training batch.

        Args:
            inputs: Dict with keys:
                hist_tokens: (B, H, feat_dim)
                cand_tokens: (B, C, feat_dim)
                hist_ctr: (B, H) int discretized CTR
                cand_ctr: (B, C) float CTR
                cand_recency: (B, C) int recency indices

        Returns:
            Logits: (B, C)
        """
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_ctr = inputs.get("cand_ctr")
        cand_recency = inputs.get("cand_recency")

        # User vector from relevance encoder
        if hist_ctr is not None:
            user_vec = self.user_encoder([hist_features, hist_ctr], training=training)
        else:
            user_vec = self.user_encoder(hist_features, training=training)

        # Candidate vectors (relevance)
        rel_cand_vecs = self._td_rel(cand_features, training=training)

        # Relevance scores (dot product)
        user_expanded = ops.expand_dims(user_vec, axis=1)
        rel_scores = ops.sum(rel_cand_vecs * user_expanded, axis=-1)

        # Bias candidate vectors (popularity)
        bias_cand_vecs = self._td_bias(cand_features, training=training)

        # Popularity scores per candidate
        B = ops.shape(cand_features)[0]
        C = ops.shape(cand_features)[1]
        bias_flat = ops.reshape(bias_cand_vecs, (B * C, self.config.news_dim))

        recency_flat = None
        if cand_recency is not None:
            recency_flat = ops.reshape(cand_recency, (B * C,))

        ctr_flat = None
        if cand_ctr is not None:
            ctr_flat = ops.reshape(cand_ctr, (B * C,))

        pop_scores_flat = self.popularity_predictor(
            bias_flat,
            recency_indices=recency_flat,
            ctr_values=ctr_flat,
            training=training,
        )
        pop_scores = ops.reshape(pop_scores_flat, (B, C))

        # Activity gate
        if self.config.use_activity_gate and self.activity_gater is not None:
            eta = self.activity_gater(user_vec, training=training)
            eta = ops.expand_dims(eta, axis=-1)
            scores = 2.0 * eta * rel_scores + 2.0 * (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores

        return scores

    def score_candidates(self, inputs, training=False):
        """Score multiple candidates (evaluation)."""
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")

        if hist_ctr is not None:
            user_vec = self.user_encoder([hist_features, hist_ctr], training=training)
        else:
            user_vec = self.user_encoder(hist_features, training=training)

        rel_cand_vecs = self._td_rel(cand_features, training=training)

        user_expanded = ops.expand_dims(user_vec, axis=1)
        scores = ops.sum(rel_cand_vecs * user_expanded, axis=-1)

        return ops.sigmoid(scores)

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {f: getattr(self.config, f) for f in self.config.__dataclass_fields__}
        )
        return base_config
