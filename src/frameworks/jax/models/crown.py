"""CROWN (WWW 2025) — JAX/Flax NNX implementation.

Faithful port of the PyTorch CROWN model. Architecture:
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

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from src.core.models.configs import CROWNConfig

from .base import BaseModel

# ======================================================================
# Layers
# ======================================================================


class PositionalEncoding(nnx.Module):
    """Sinusoidal positional encoding (standard transformer)."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int, *, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe[None, :, :])  # (1, max_len, d_model)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        return self.dropout(x + self.pe[:, : x.shape[1]], deterministic=deterministic)


class TransformerEncoderLayer(nnx.Module):
    """Standard transformer encoder layer (MHA + FFN + LayerNorm)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, *, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            qkv_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        attn_out = self.self_attn(x, x, deterministic=deterministic)
        x = self.norm1(x + self.dropout(attn_out, deterministic=deterministic))
        ff_out = self.linear2(jax.nn.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out, deterministic=deterministic))
        return x


class UserQueryAttention(nnx.Module):
    """Additive attention with GNN-updated user proxy as query (paper eq. 9)."""

    def __init__(self, feature_dim: int, attention_dim: int, *, rngs: nnx.Rngs):
        self.W_key = nnx.Linear(feature_dim, attention_dim, use_bias=True, rngs=rngs)
        self.W_query = nnx.Linear(feature_dim, attention_dim, use_bias=False, rngs=rngs)

    def __call__(
        self, news: jax.Array, user_node: jax.Array, mask: jax.Array | None = None
    ) -> jax.Array:
        keys = jnp.tanh(self.W_key(news))  # (B, H, A)
        query = self.W_query(user_node)[:, None, :]  # (B, 1, A)
        scores = jnp.sum(keys * query, axis=-1)  # (B, H)
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1)[:, :, None]  # (B, H, 1)
        return jnp.sum(news * weights, axis=1)  # (B, D)


# ======================================================================
# Bipartite GNN layers (paper §3.3, eq. 8)
# ======================================================================


class BipartiteGATLayer(nnx.Module):
    """GAT on a user↔news bipartite graph with self-loops."""

    def __init__(self, dim: int, num_heads: int, dropout_rate: float, alpha: float = 0.2, *, rngs: nnx.Rngs):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.alpha = alpha

        self.W = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        glorot = nnx.initializers.glorot_uniform()
        self.a_src = nnx.Param(glorot(rngs.params(), (num_heads, self.head_dim)))
        self.a_dst = nnx.Param(glorot(rngs.params(), (num_heads, self.head_dim)))
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self, user: jax.Array, news: jax.Array, news_mask: jax.Array, *, deterministic: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        B, H, D = news.shape
        NH, HD = self.num_heads, self.head_dim
        neg_inf = jnp.finfo(news.dtype).min

        Wu = self.W(user).reshape(B, NH, HD)
        Wn = self.W(news).reshape(B, H, NH, HD)

        src_u = jnp.sum(Wu * self.a_src.value, axis=-1)
        dst_u = jnp.sum(Wu * self.a_dst.value, axis=-1)
        src_n = jnp.sum(Wn * self.a_src.value, axis=-1)
        dst_n = jnp.sum(Wn * self.a_dst.value, axis=-1)

        # User update: attend over {self, all news}
        score_u_n = jax.nn.leaky_relu(src_u[:, None, :] + dst_n, negative_slope=self.alpha)
        score_u_u = jax.nn.leaky_relu(src_u + dst_u, negative_slope=self.alpha)
        score_u_n = jnp.where(news_mask[:, :, None], score_u_n, neg_inf)

        all_u = jnp.concatenate([score_u_n, score_u_u[:, None, :]], axis=1)
        attn_u = self.dropout(jax.nn.softmax(all_u, axis=1), deterministic=deterministic)
        u_from_n = jnp.sum(attn_u[:, :H, :, None] * Wn, axis=1)
        u_self = attn_u[:, H, :, None] * Wu
        user_new = jax.nn.elu(u_from_n + u_self).reshape(B, D)

        # News update: each news attends over {self, user}
        score_n_u = jax.nn.leaky_relu(src_n + dst_u[:, None, :], negative_slope=self.alpha)
        score_n_n = jax.nn.leaky_relu(src_n + dst_n, negative_slope=self.alpha)
        stacked = jnp.stack([score_n_u, score_n_n], axis=-2)
        attn_n = self.dropout(jax.nn.softmax(stacked, axis=-2), deterministic=deterministic)
        n_from_u = attn_n[:, :, 0, :, None] * Wu[:, None, :, :]
        n_self = attn_n[:, :, 1, :, None] * Wn
        news_new = jax.nn.elu(n_from_u + n_self).reshape(B, H, D)

        return user_new, news_new


class BipartiteSAGELayer(nnx.Module):
    """GraphSAGE on a user↔news bipartite graph."""

    def __init__(self, dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.W_self = nnx.Linear(dim, dim, rngs=rngs)
        self.W_neigh = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self, user: jax.Array, news: jax.Array, news_mask: jax.Array, *, deterministic: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        m = news_mask[:, :, None].astype(news.dtype)
        news_count = jnp.maximum(jnp.sum(m, axis=1), 1.0)
        news_mean = jnp.sum(news * m, axis=1) / news_count

        user_new = jax.nn.relu(self.W_self(user) + self.W_neigh(news_mean))
        news_new = jax.nn.relu(self.W_self(news) + self.W_neigh(user[:, None, :] * jnp.ones_like(news)))

        user_new = user_new / (jnp.linalg.norm(user_new, axis=-1, keepdims=True) + 1e-8)
        news_new = news_new / (jnp.linalg.norm(news_new, axis=-1, keepdims=True) + 1e-8)
        return (
            self.dropout(user_new, deterministic=deterministic),
            self.dropout(news_new, deterministic=deterministic),
        )


# ======================================================================
# News encoder (paper §3.1)
# ======================================================================


class CROWNNewsEncoder(nnx.Module):
    """CROWN news encoder with intent disentanglement."""

    def __init__(
        self,
        config: CROWNConfig,
        word_embedding: nnx.Embed,
        category_embedding: nnx.Embed,
        subcategory_embedding: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.word_embedding = word_embedding
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.title_pos = PositionalEncoding(config.embedding_size, config.dropout_rate, config.max_title_length, rngs=rngs)
        self.body_pos = PositionalEncoding(config.embedding_size, config.dropout_rate, config.max_abstract_length, rngs=rngs)

        self.title_transformer = TransformerEncoderLayer(
            config.embedding_size, config.num_heads, config.feedforward_dim, config.dropout_rate, rngs=rngs,
        )
        self.body_transformer = TransformerEncoderLayer(
            config.embedding_size, config.num_heads, config.feedforward_dim, config.dropout_rate, rngs=rngs,
        )

        cat_concat_dim = config.category_embedding_dim + config.subcategory_embedding_dim
        self.category_affine = nnx.Linear(cat_concat_dim, config.category_embedding_dim, rngs=rngs)

        intent_input_dim = config.embedding_size + config.category_embedding_dim
        self.intent_layers = nnx.List([
            nnx.Linear(intent_input_dim, config.intent_embedding_dim, rngs=rngs)
            for _ in range(config.intent_num)
        ])

        self.title_intent_W = nnx.Param(
            nnx.initializers.glorot_uniform()(rngs.params(), (config.intent_embedding_dim, config.attention_dim))
        )
        self.title_intent_b = nnx.Param(jnp.zeros((config.attention_dim,)))
        self.title_intent_q = nnx.Param(
            nnx.initializers.glorot_uniform()(rngs.params(), (config.attention_dim, 1))
        )
        self.body_intent_W = nnx.Param(
            nnx.initializers.glorot_uniform()(rngs.params(), (config.intent_embedding_dim, config.attention_dim))
        )
        self.body_intent_b = nnx.Param(jnp.zeros((config.attention_dim,)))
        self.body_intent_q = nnx.Param(
            nnx.initializers.glorot_uniform()(rngs.params(), (config.attention_dim, 1))
        )

        self.category_predictor = nnx.Linear(config.intent_embedding_dim, 1, rngs=rngs)

        self._aux_loss = nnx.Variable(jnp.float32(0.0))

    def _additive_attn(self, features: jax.Array, W, b, q) -> jax.Array:
        hidden = jnp.tanh(jnp.matmul(features, W.value) + b.value)
        scores = jnp.squeeze(jnp.matmul(hidden, q.value), axis=-1)
        weights = jax.nn.softmax(scores, axis=-1)[:, :, None]
        return jnp.sum(features * weights, axis=1)

    def _k_intent_disentangle(self, x: jax.Array) -> jax.Array:
        intents = [jax.nn.relu(layer(x))[:, None, :] for layer in self.intent_layers]
        return jnp.concatenate(intents, axis=1)

    def __call__(
        self,
        title_tokens_or_concat: jax.Array,
        abstract_tokens: jax.Array | None = None,
        category_ids: jax.Array | None = None,
        subcategory_ids: jax.Array | None = None,
        *,
        compute_aux_loss: bool = False,
        training: bool = False,
    ) -> jax.Array:
        cfg = self.config
        det = not training

        if abstract_tokens is None:
            concat = title_tokens_or_concat
            title_tokens = concat[:, :cfg.max_title_length]
            abstract_tokens = concat[:, cfg.max_title_length : cfg.max_title_length + cfg.max_abstract_length]
            category_ids = concat[:, cfg.max_title_length + cfg.max_abstract_length]
            subcategory_ids = concat[:, cfg.max_title_length + cfg.max_abstract_length + 1]
        else:
            title_tokens = title_tokens_or_concat

        title_emb = self.dropout(self.word_embedding(title_tokens), deterministic=det)
        body_emb = self.dropout(self.word_embedding(abstract_tokens), deterministic=det)
        title_emb = self.title_pos(title_emb, deterministic=det)
        body_emb = self.body_pos(body_emb, deterministic=det)

        title_enc = self.title_transformer(title_emb, deterministic=det)
        body_enc = self.body_transformer(body_emb, deterministic=det)

        title_pool = jnp.mean(title_enc, axis=1)
        body_pool = jnp.mean(body_enc, axis=1)

        cat_emb = self.category_embedding(category_ids)
        subcat_emb = self.subcategory_embedding(subcategory_ids)
        cat_repr = self.category_affine(jnp.concatenate([cat_emb, subcat_emb], axis=-1))

        cat_aware_title = jnp.concatenate([title_pool, cat_repr], axis=-1)
        cat_aware_body = jnp.concatenate([body_pool, cat_repr], axis=-1)

        title_k = self._k_intent_disentangle(cat_aware_title)
        body_k = self._k_intent_disentangle(cat_aware_body)

        title_intent = self._additive_attn(title_k, self.title_intent_W, self.title_intent_b, self.title_intent_q)
        body_intent = self._additive_attn(body_k, self.body_intent_W, self.body_intent_b, self.body_intent_q)

        if compute_aux_loss:
            logits = self.category_predictor(title_intent)
            self._aux_loss.value = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, category_ids)
            )

        sim = jnp.sum(title_intent * body_intent, axis=-1) / (
            jnp.linalg.norm(title_intent, axis=-1) * jnp.linalg.norm(body_intent, axis=-1) + 1e-8
        )
        sim = (sim + 1.0) / 2.0

        news_repr = jnp.concatenate(
            [
                title_intent,
                sim[:, None] * body_intent,
                self.dropout(cat_emb, deterministic=det),
                self.dropout(subcat_emb, deterministic=det),
            ],
            axis=-1,
        )
        return news_repr


# ======================================================================
# User encoder (paper §3.3)
# ======================================================================


class CROWNUserEncoder(nnx.Module):
    """CROWN user encoder: bipartite GNN + paper eq. 9 user-query attention."""

    def __init__(self, config: CROWNConfig, news_encoder: CROWNNewsEncoder, *, rngs: nnx.Rngs):
        self.config = config
        self.news_encoder = news_encoder
        news_emb_dim = news_encoder.news_embedding_dim

        glorot = nnx.initializers.glorot_uniform()
        self.user_node = nnx.Param(
            jax.random.uniform(rngs.params(), (news_emb_dim,), minval=-0.1, maxval=0.1)
        )

        if config.gnn_type == "gat":
            self.gnn_layers = nnx.List([
                BipartiteGATLayer(
                    dim=news_emb_dim, num_heads=config.gat_num_heads,
                    dropout_rate=config.dropout_rate, alpha=config.gat_alpha, rngs=rngs,
                )
                for _ in range(config.graph_num_layers)
            ])
        elif config.gnn_type == "graphsage":
            self.gnn_layers = nnx.List([
                BipartiteSAGELayer(dim=news_emb_dim, dropout_rate=config.dropout_rate, rngs=rngs)
                for _ in range(config.graph_num_layers)
            ])
        else:
            raise ValueError(f"Unknown gnn_type: {config.gnn_type!r}")

        self.user_attention = UserQueryAttention(
            feature_dim=news_emb_dim, attention_dim=config.user_attention_dim, rngs=rngs,
        )

    def _encode_history_graph(
        self,
        history_title: jax.Array,
        history_abstract: jax.Array,
        history_category: jax.Array,
        history_subcategory: jax.Array,
        history_mask: jax.Array,
        *,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        B, H = history_title.shape[:2]
        det = not training

        flat_news = self.news_encoder(
            history_title.reshape(B * H, -1),
            history_abstract.reshape(B * H, -1),
            history_category.reshape(B * H),
            history_subcategory.reshape(B * H),
            compute_aux_loss=False,
            training=training,
        )
        news = flat_news.reshape(B, H, -1)

        user = jnp.broadcast_to(self.user_node.value[None, :], (B, self.user_node.value.shape[0]))
        for gnn in self.gnn_layers:
            user, news = gnn(user, news, history_mask, deterministic=det)
        return user, news

    def forward_with_candidates(
        self,
        history_title: jax.Array,
        history_abstract: jax.Array,
        history_category: jax.Array,
        history_subcategory: jax.Array,
        history_mask: jax.Array,
        candidate_repr: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        user_node, news = self._encode_history_graph(
            history_title, history_abstract, history_category,
            history_subcategory, history_mask, training=training,
        )
        return self.user_attention(news, user_node, mask=history_mask)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        cfg = self.config
        tl = cfg.max_title_length
        al = cfg.max_abstract_length

        history_title = inputs[:, :, :tl]
        history_abstract = inputs[:, :, tl : tl + al]
        history_category = inputs[:, :, tl + al]
        history_subcategory = inputs[:, :, tl + al + 1]
        history_mask = jnp.any(inputs != 0, axis=-1)

        user_node, news = self._encode_history_graph(
            history_title, history_abstract, history_category,
            history_subcategory, history_mask, training=training,
        )
        return self.user_attention(news, user_node, mask=history_mask)


# ======================================================================
# Full CROWN model
# ======================================================================


class CROWN(BaseModel):
    """CROWN: intent disentanglement + bipartite GNN user encoder."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CROWNConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if config is None:
            config = CROWNConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        num_subcategories = int(processed_news.get("num_subcategories", 100))

        vocab_size = int(processed_news["vocab_size"])
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        self.word_embedding = nnx.Embed(num_embeddings=vocab_size, features=config.embedding_size, rngs=rngs)
        self.word_embedding.embedding.value = jnp.asarray(embeddings_matrix)

        self.category_embedding = nnx.Embed(num_embeddings=num_categories + 1, features=config.category_embedding_dim, rngs=rngs)
        self.category_embedding.embedding.value = jax.random.uniform(
            rngs.params(), (num_categories + 1, config.category_embedding_dim), minval=-0.1, maxval=0.1,
        )
        self.subcategory_embedding = nnx.Embed(num_embeddings=num_subcategories + 1, features=config.subcategory_embedding_dim, rngs=rngs)
        self.subcategory_embedding.embedding.value = jax.random.uniform(
            rngs.params(), (num_subcategories + 1, config.subcategory_embedding_dim), minval=-0.1, maxval=0.1,
        )

        self.news_encoder = CROWNNewsEncoder(
            config, self.word_embedding, self.category_embedding, self.subcategory_embedding, rngs=rngs,
        )
        self.news_encoder.category_predictor = nnx.Linear(
            config.intent_embedding_dim, num_categories + 1, rngs=rngs,
        )

        self.user_encoder = CROWNUserEncoder(config, self.news_encoder, rngs=rngs)
        self.alpha = config.alpha

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        B, C = inputs["cand_tokens"].shape[:2]

        cand_full = self.news_encoder(
            inputs["cand_tokens"].reshape(B * C, -1),
            inputs["cand_abstract_tokens"].reshape(B * C, -1),
            inputs["cand_category"].reshape(B * C),
            inputs["cand_subcategory"].reshape(B * C),
            compute_aux_loss=training,
            training=training,
        ).reshape(B, C, -1)

        history_mask = jnp.any(inputs["hist_tokens"] != 0, axis=-1)

        user_repr = self.user_encoder.forward_with_candidates(
            history_title=inputs["hist_tokens"],
            history_abstract=inputs["hist_abstract_tokens"],
            history_category=inputs["hist_category"],
            history_subcategory=inputs["hist_subcategory"],
            history_mask=history_mask,
            candidate_repr=cand_full,
            training=training,
        )

        scores = jnp.sum(user_repr[:, None, :] * cand_full, axis=-1)
        return scores

    def get_auxiliary_loss(self) -> jax.Array:
        return self.alpha * self.news_encoder._aux_loss.value
