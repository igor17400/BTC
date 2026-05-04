"""CROWN (WWW 2025)

- **News encoder**: Transformer + positional encoding → mean pool →
  category-aware k-intent disentanglement → additive attention →
  title-body cosine similarity → concatenation with category embeddings.
- **User encoder**: Bipartite user↔news graph → multi-layer GAT/GraphSAGE
  → paper eq. 9 user-query attention (candidate-independent).
- **Auxiliary loss**: Category prediction from title intent embeddings.

Reference: Ko et al., "CROWN: A Novel Approach to Comprehending Users'
Preferences for Accurate Personalized News Recommendation", WWW 2025.
"""

from __future__ import annotations

import math
from typing import Any

import keras
import numpy as np
from keras import layers, ops

from src.core.models.configs import CROWNConfig
from src.frameworks.keras.layers import AdditiveAttention, GlorotUniformMHA
from src.frameworks.keras.models.base import BaseModel

# ======================================================================
# Layers
# ======================================================================


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding (standard transformer)."""

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.1,
        max_len: int = 512,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate, seed=seed)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe_value = pe[None, :, :]  # (1, max_len, d_model)

    def build(self, input_shape):
        self.pe = self.add_weight(
            name="pe",
            shape=self.pe_value.shape,
            initializer=keras.initializers.Constant(self.pe_value),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        return self.dropout(x + self.pe[:, : ops.shape(x)[1]], training=training)


class UserQueryAttention(layers.Layer):
    """Additive attention with GNN-updated user proxy as query (paper eq. 9)."""

    def __init__(self, feature_dim: int, attention_dim: int, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.W_key = layers.Dense(attention_dim, use_bias=True, name="W_key")
        self.W_query = layers.Dense(attention_dim, use_bias=False, name="W_query")

    def call(self, news, user_node, mask=None):
        """news: (B,H,D), user_node: (B,D), mask: (B,H) -> (B,D)."""
        keys = ops.tanh(self.W_key(news))  # (B, H, A)
        query = ops.expand_dims(self.W_query(user_node), axis=1)  # (B, 1, A)
        scores = ops.sum(keys * query, axis=-1)  # (B, H)
        if mask is not None:
            scores = ops.where(mask, scores, ops.full_like(scores, -1e9))
        weights = ops.expand_dims(keras.activations.softmax(scores, axis=-1), axis=-1)
        return ops.sum(news * weights, axis=1)


# ======================================================================
# Bipartite GNN layers (paper §3.3, eq. 8)
# ======================================================================


class BipartiteGATLayer(layers.Layer):
    """GAT on a user↔news bipartite graph with self-loops."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        alpha: float = 0.2,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.alpha = alpha
        self.W = layers.Dense(dim, use_bias=False, name="gat_W")
        self.dropout = layers.Dropout(dropout_rate, seed=seed)

    def build(self, input_shape):
        glorot = keras.initializers.GlorotUniform()
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.head_dim),
            initializer=glorot,
            trainable=True,
        )
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.num_heads, self.head_dim),
            initializer=glorot,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, user, news, news_mask, training=None):
        """user: (B,D), news: (B,H,D), news_mask: (B,H) -> (user_new, news_new)."""
        B = ops.shape(news)[0]
        H = ops.shape(news)[1]
        D = self.dim
        NH, HD = self.num_heads, self.head_dim

        Wu = ops.reshape(self.W(user), (B, NH, HD))
        Wn = ops.reshape(self.W(news), (B, H, NH, HD))

        src_u = ops.sum(Wu * self.a_src, axis=-1)  # (B, NH)
        dst_u = ops.sum(Wu * self.a_dst, axis=-1)
        src_n = ops.sum(Wn * self.a_src, axis=-1)  # (B, H, NH)
        dst_n = ops.sum(Wn * self.a_dst, axis=-1)

        neg_inf = -1e9

        # User update: attend over {self, all news}
        score_u_n = ops.where(
            ops.expand_dims(news_mask, axis=-1),
            keras.activations.relu(ops.expand_dims(src_u, axis=1) + dst_n)
            - self.alpha
            * ops.abs(
                ops.expand_dims(src_u, axis=1)
                + dst_n
                - ops.abs(ops.expand_dims(src_u, axis=1) + dst_n)
            ),
            ops.full((B, H, NH), neg_inf),
        )
        # Simpler leaky_relu via ops
        raw_u_n = ops.expand_dims(src_u, axis=1) + dst_n  # (B, H, NH)
        score_u_n = ops.where(raw_u_n >= 0, raw_u_n, self.alpha * raw_u_n)
        score_u_n = ops.where(
            ops.expand_dims(news_mask, axis=-1),
            score_u_n,
            ops.full_like(score_u_n, neg_inf),
        )

        raw_u_u = src_u + dst_u  # (B, NH)
        score_u_u = ops.where(raw_u_u >= 0, raw_u_u, self.alpha * raw_u_u)

        all_u = ops.concatenate(
            [score_u_n, ops.expand_dims(score_u_u, axis=1)], axis=1
        )  # (B, H+1, NH)
        attn_u = self.dropout(
            keras.activations.softmax(all_u, axis=1), training=training
        )
        u_from_n = ops.sum(
            ops.expand_dims(attn_u[:, :H], axis=-1) * Wn, axis=1
        )  # (B, NH, HD)
        u_self = ops.expand_dims(attn_u[:, H], axis=-1) * Wu
        user_new = ops.reshape(keras.activations.elu(u_from_n + u_self), (B, D))

        # News update: each news attends over {self, user}
        raw_n_u = src_n + ops.expand_dims(dst_u, axis=1)  # (B, H, NH)
        score_n_u = ops.where(raw_n_u >= 0, raw_n_u, self.alpha * raw_n_u)
        raw_n_n = src_n + dst_n
        score_n_n = ops.where(raw_n_n >= 0, raw_n_n, self.alpha * raw_n_n)
        stacked = ops.stack([score_n_u, score_n_n], axis=-2)  # (B, H, 2, NH)
        attn_n = self.dropout(
            keras.activations.softmax(stacked, axis=-2), training=training
        )
        n_from_u = ops.expand_dims(attn_n[:, :, 0], axis=-1) * ops.expand_dims(
            Wu, axis=1
        )
        n_self = ops.expand_dims(attn_n[:, :, 1], axis=-1) * Wn
        news_new = ops.reshape(keras.activations.elu(n_from_u + n_self), (B, H, D))

        return user_new, news_new


class BipartiteSAGELayer(layers.Layer):
    """GraphSAGE on a user↔news bipartite graph."""

    def __init__(self, dim: int, dropout_rate: float, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.W_self = layers.Dense(dim, name="sage_self")
        self.W_neigh = layers.Dense(dim, name="sage_neigh")
        self.dropout = layers.Dropout(dropout_rate, seed=seed)

    def call(self, user, news, news_mask, training=None):
        m = ops.expand_dims(ops.cast(news_mask, dtype=news.dtype), axis=-1)
        news_count = ops.maximum(ops.sum(m, axis=1), 1.0)
        news_mean = ops.sum(news * m, axis=1) / news_count

        user_new = ops.relu(self.W_self(user) + self.W_neigh(news_mean))
        user_expanded = ops.repeat(
            ops.expand_dims(user, axis=1), ops.shape(news)[1], axis=1
        )
        news_new = ops.relu(self.W_self(news) + self.W_neigh(user_expanded))

        # L2 normalize
        user_new = user_new / (
            ops.sqrt(ops.sum(ops.square(user_new), axis=-1, keepdims=True)) + 1e-8
        )
        news_new = news_new / (
            ops.sqrt(ops.sum(ops.square(news_new), axis=-1, keepdims=True)) + 1e-8
        )
        return self.dropout(user_new, training=training), self.dropout(
            news_new, training=training
        )


# ======================================================================
# News encoder (paper §3.1)
# ======================================================================


class CROWNNewsEncoder(keras.Model):
    """CROWN news encoder with intent disentanglement.

    Pipeline:
        1. Word embedding + dropout + positional encoding
        2. TransformerEncoder (configurable layers)
        3. Mean pooling
        4. Category-aware concat
        5. k-intent disentanglement (k separate Dense → ReLU)
        6. Additive attention over k intents
        7. Title-body cosine similarity
        8. Concat [title_intent, sim * body_intent, cat_emb, subcat_emb]
    """

    def __init__(
        self,
        config: CROWNConfig,
        word_embedding: layers.Embedding,
        category_embedding: layers.Embedding,
        subcategory_embedding: layers.Embedding,
        num_categories: int = 1,
        name: str = "news_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.word_embedding = word_embedding
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        self.emb_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)
        self.title_pos = PositionalEncoding(
            config.embedding_size,
            config.dropout_rate,
            config.max_title_length,
            seed=config.seed,
        )
        self.body_pos = PositionalEncoding(
            config.embedding_size,
            config.dropout_rate,
            config.max_abstract_length,
            seed=config.seed,
        )

        # Transformer layers (title and body share architecture but not weights)
        self.title_mha = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="title_mha",
        )
        self.title_ffn1 = layers.Dense(
            config.feedforward_dim, activation="relu", name="title_ffn1"
        )
        self.title_ffn2 = layers.Dense(config.embedding_size, name="title_ffn2")
        self.title_ln1 = layers.LayerNormalization(name="title_ln1")
        self.title_ln2 = layers.LayerNormalization(name="title_ln2")
        self.title_drop = layers.Dropout(config.dropout_rate, seed=config.seed)

        self.body_mha = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            dropout=config.dropout_rate,
            kernel_initializer=GlorotUniformMHA(),
            name="body_mha",
        )
        self.body_ffn1 = layers.Dense(
            config.feedforward_dim, activation="relu", name="body_ffn1"
        )
        self.body_ffn2 = layers.Dense(config.embedding_size, name="body_ffn2")
        self.body_ln1 = layers.LayerNormalization(name="body_ln1")
        self.body_ln2 = layers.LayerNormalization(name="body_ln2")
        self.body_drop = layers.Dropout(config.dropout_rate, seed=config.seed)

        # Category affine
        self.category_affine = layers.Dense(
            config.category_embedding_dim, name="category_affine"
        )

        # k-intent disentanglement
        self.intent_layers = [
            layers.Dense(config.intent_embedding_dim, name=f"intent_{i}")
            for i in range(config.intent_num)
        ]

        # Intent attention (additive)
        self.title_intent_attn = AdditiveAttention(
            query_vec_dim=config.attention_dim,
            seed=config.seed,
            name="title_intent_attn",
        )
        self.body_intent_attn = AdditiveAttention(
            query_vec_dim=config.attention_dim,
            seed=config.seed,
            name="body_intent_attn",
        )

        # Category predictor — sized at construction time (no replacement needed)
        self.category_predictor = layers.Dense(
            num_categories, name="category_predictor"
        )

        self.cat_dropout = layers.Dropout(config.dropout_rate, seed=config.seed)

    def _transformer_block(self, x, mha, ffn1, ffn2, ln1, ln2, drop, training):
        attn_out = mha(x, x, x, training=training)
        x = ln1(x + drop(attn_out, training=training))
        ff_out = ffn2(ffn1(x))
        x = ln2(x + drop(ff_out, training=training))
        return x

    def call(
        self,
        title_tokens_or_concat,
        abstract_tokens=None,
        category_ids=None,
        subcategory_ids=None,
        compute_aux_loss=False,
        training=None,
    ):
        """Encode news. Accepts explicit args or a concatenated tensor."""
        cfg = self.config
        if abstract_tokens is None:
            concat = title_tokens_or_concat
            title_tokens = concat[:, : cfg.max_title_length]
            abstract_tokens = concat[
                :, cfg.max_title_length : cfg.max_title_length + cfg.max_abstract_length
            ]
            category_ids = concat[:, cfg.max_title_length + cfg.max_abstract_length]
            subcategory_ids = concat[
                :, cfg.max_title_length + cfg.max_abstract_length + 1
            ]
        else:
            title_tokens = title_tokens_or_concat

        # 1. Word embedding + positional encoding
        title_emb = self.emb_dropout(
            self.word_embedding(title_tokens), training=training
        )
        body_emb = self.emb_dropout(
            self.word_embedding(abstract_tokens), training=training
        )
        title_emb = self.title_pos(title_emb, training=training)
        body_emb = self.body_pos(body_emb, training=training)

        # 2. Transformer
        title_enc = self._transformer_block(
            title_emb,
            self.title_mha,
            self.title_ffn1,
            self.title_ffn2,
            self.title_ln1,
            self.title_ln2,
            self.title_drop,
            training,
        )
        body_enc = self._transformer_block(
            body_emb,
            self.body_mha,
            self.body_ffn1,
            self.body_ffn2,
            self.body_ln1,
            self.body_ln2,
            self.body_drop,
            training,
        )

        # 3. Mean pooling
        title_pool = ops.mean(title_enc, axis=1)
        body_pool = ops.mean(body_enc, axis=1)

        # 4. Category-aware
        cat_emb = self.category_embedding(category_ids)
        subcat_emb = self.subcategory_embedding(subcategory_ids)
        cat_repr = self.category_affine(ops.concatenate([cat_emb, subcat_emb], axis=-1))

        cat_aware_title = ops.concatenate([title_pool, cat_repr], axis=-1)
        cat_aware_body = ops.concatenate([body_pool, cat_repr], axis=-1)

        # 5. k-intent disentanglement
        title_k = ops.stack(
            [ops.relu(layer(cat_aware_title)) for layer in self.intent_layers], axis=1
        )
        body_k = ops.stack(
            [ops.relu(layer(cat_aware_body)) for layer in self.intent_layers], axis=1
        )

        # 6. Intent attention
        title_intent = self.title_intent_attn(title_k)
        body_intent = self.body_intent_attn(body_k)

        # 7. Auxiliary loss — computed inline and returned via
        # get_auxiliary_loss() which the training loop calls after forward.
        if compute_aux_loss and self.category_predictor is not None:
            logits = self.category_predictor(title_intent)
            aux = ops.mean(
                keras.losses.sparse_categorical_crossentropy(
                    category_ids,
                    logits,
                    from_logits=True,
                )
            )
        else:
            aux = ops.zeros(())
        # Store for retrieval — use a list to avoid JAX tracing issues
        # with attribute assignment during traced functions.
        self._aux_loss_holder = [aux]

        # 8. Title-body cosine similarity
        title_norm = ops.sqrt(ops.sum(ops.square(title_intent), axis=-1) + 1e-8)
        body_norm = ops.sqrt(ops.sum(ops.square(body_intent), axis=-1) + 1e-8)
        sim = ops.sum(title_intent * body_intent, axis=-1) / (title_norm * body_norm)
        sim = (sim + 1.0) / 2.0

        # 9. Final news representation
        news_repr = ops.concatenate(
            [
                title_intent,
                ops.expand_dims(sim, axis=-1) * body_intent,
                self.cat_dropout(cat_emb, training=training),
                self.cat_dropout(subcat_emb, training=training),
            ],
            axis=-1,
        )

        return news_repr


# ======================================================================
# User encoder (paper §3.3)
# ======================================================================


class CROWNUserEncoder(keras.Model):
    """CROWN user encoder: bipartite GNN + paper eq. 9 user-query attention."""

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: CROWNNewsEncoder,
        name: str = "user_encoder",
    ):
        super().__init__(name=name)
        self.config = config
        self.news_encoder = news_encoder
        news_emb_dim = news_encoder.news_embedding_dim

        # GNN layers
        if config.gnn_type == "gat":
            self.gnn_layers = [
                BipartiteGATLayer(
                    dim=news_emb_dim,
                    num_heads=config.gat_num_heads,
                    dropout_rate=config.dropout_rate,
                    alpha=config.gat_alpha,
                    seed=config.seed,
                    name=f"gat_{i}",
                )
                for i in range(config.graph_num_layers)
            ]
        elif config.gnn_type == "graphsage":
            self.gnn_layers = [
                BipartiteSAGELayer(
                    dim=news_emb_dim,
                    dropout_rate=config.dropout_rate,
                    seed=config.seed,
                    name=f"sage_{i}",
                )
                for i in range(config.graph_num_layers)
            ]
        else:
            raise ValueError(f"Unknown gnn_type: {config.gnn_type!r}")

        self.user_attention = UserQueryAttention(
            feature_dim=news_emb_dim,
            attention_dim=config.user_attention_dim,
            seed=config.seed,
            name="user_query_attention",
        )

        self._news_emb_dim = news_emb_dim

    def build(self, input_shape):
        # Learnable user proxy node
        self.user_node = self.add_weight(
            name="user_proxy",
            shape=(self._news_emb_dim,),
            initializer=keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
        )
        super().build(input_shape)

    def _encode_history_graph(
        self,
        history_title,
        history_abstract,
        history_category,
        history_subcategory,
        history_mask,
        training=None,
    ):
        B = ops.shape(history_title)[0]
        H = ops.shape(history_title)[1]

        flat_news = self.news_encoder(
            ops.reshape(history_title, (B * H, -1)),
            ops.reshape(history_abstract, (B * H, -1)),
            ops.reshape(history_category, (B * H,)),
            ops.reshape(history_subcategory, (B * H,)),
            compute_aux_loss=False,
            training=training,
        )
        news = ops.reshape(flat_news, (B, H, -1))

        user = ops.broadcast_to(self.user_node[None, :], (B, self._news_emb_dim))

        for gnn in self.gnn_layers:
            user, news = gnn(user, news, history_mask, training=training)
        return user, news

    def forward_with_candidates(
        self,
        history_title,
        history_abstract,
        history_category,
        history_subcategory,
        history_mask,
        candidate_repr,
        training=None,
    ):
        """Training: one user representation per behavior."""
        user_node, news = self._encode_history_graph(
            history_title,
            history_abstract,
            history_category,
            history_subcategory,
            history_mask,
            training=training,
        )
        return self.user_attention(news, user_node, mask=history_mask)

    def call(self, inputs, training=None):
        """Evaluation: concatenated history -> single user vector."""
        cfg = self.config
        tl = cfg.max_title_length
        al = cfg.max_abstract_length

        history_title = inputs[:, :, :tl]
        history_abstract = inputs[:, :, tl : tl + al]
        history_category = inputs[:, :, tl + al]
        history_subcategory = inputs[:, :, tl + al + 1]
        history_mask = ops.any(ops.not_equal(inputs, 0), axis=-1)

        user_node, news = self._encode_history_graph(
            history_title,
            history_abstract,
            history_category,
            history_subcategory,
            history_mask,
            training=training,
        )
        return self.user_attention(news, user_node, mask=history_mask)


# ======================================================================
# Full CROWN model
# ======================================================================


class CROWN(BaseModel):
    """CROWN: intent disentanglement + bipartite GNN user encoder.

    Training: encode candidates (with aux loss), encode user via GNN,
    dot-product scoring. Evaluation uses news_encoder and user_encoder
    separately via the shared evaluation pipeline.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CROWNConfig | None = None,
        name: str = "crown",
        **config_overrides,
    ):
        super().__init__(name=name)

        if config is None:
            valid_fields = set(CROWNConfig.__dataclass_fields__)
            filtered = {k: v for k, v in config_overrides.items() if k in valid_fields}
            config = CROWNConfig(**filtered)
        self.config = config
        self.processed_news = processed_news
        self._validate_processed_news()
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        num_subcategories = int(processed_news.get("num_subcategories", 100))

        # Shared word embedding
        vocab_size = processed_news["vocab_size"]
        self.embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_size,
            embeddings_initializer=keras.initializers.Constant(
                processed_news["embeddings"]
            ),
            trainable=True,
            name="word_embedding",
        )

        # Category / subcategory embeddings
        self.category_emb = layers.Embedding(
            input_dim=num_categories + 1,
            output_dim=config.category_embedding_dim,
            embeddings_initializer=keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
            name="category_embedding",
        )
        self.subcategory_emb = layers.Embedding(
            input_dim=num_subcategories + 1,
            output_dim=config.subcategory_embedding_dim,
            embeddings_initializer=keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
            name="subcategory_embedding",
        )

        # News encoder
        self.news_encoder = CROWNNewsEncoder(
            config,
            self.embedding_layer,
            self.category_emb,
            self.subcategory_emb,
            num_categories=num_categories + 1,
        )

        # User encoder
        self.user_encoder = CROWNUserEncoder(config, self.news_encoder)

        self.alpha = config.alpha

        # Force-build with dummy shapes
        tl = config.max_title_length
        al = config.max_abstract_length
        H = config.max_history_length
        feat = tl + al + 2  # title + abstract + category + subcategory

        dummy_news = np.zeros((1, feat), dtype="int32")
        # Use compute_aux_loss=True so the category predictor gets built
        self.news_encoder(dummy_news, compute_aux_loss=True, training=False)

        dummy_hist = np.zeros((1, H, feat), dtype="int32")
        self.user_encoder(dummy_hist, training=False)

    def call(self, inputs, training=None):
        """Training forward pass. Returns raw logits (B, C)."""
        B = ops.shape(inputs["cand_tokens"])[0]
        C = ops.shape(inputs["cand_tokens"])[1]

        # Encode candidates (with aux loss during training)
        cand_full = self.news_encoder(
            ops.reshape(inputs["cand_tokens"], (B * C, -1)),
            ops.reshape(inputs["cand_abstract_tokens"], (B * C, -1)),
            ops.reshape(inputs["cand_category"], (B * C,)),
            ops.reshape(inputs["cand_subcategory"], (B * C,)),
            compute_aux_loss=bool(training),
            training=training,
        )
        cand_full = ops.reshape(cand_full, (B, C, -1))

        # Snapshot aux loss before history encoding overwrites it
        self._aux_loss_snapshot = self.news_encoder._aux_loss_holder[0]

        history_mask = ops.any(ops.not_equal(inputs["hist_tokens"], 0), axis=-1)

        user_repr = self.user_encoder.forward_with_candidates(
            history_title=inputs["hist_tokens"],
            history_abstract=inputs["hist_abstract_tokens"],
            history_category=inputs["hist_category"],
            history_subcategory=inputs["hist_subcategory"],
            history_mask=history_mask,
            candidate_repr=cand_full,
            training=training,
        )

        # Dot-product scoring
        scores = ops.sum(ops.expand_dims(user_repr, axis=1) * cand_full, axis=-1)
        return scores

    def get_auxiliary_loss(self):
        """Return weighted auxiliary loss from the news encoder."""
        return self.alpha * self._aux_loss_snapshot

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                field: getattr(self.config, field)
                for field in CROWNConfig.__dataclass_fields__
            }
        )
        return base_config
