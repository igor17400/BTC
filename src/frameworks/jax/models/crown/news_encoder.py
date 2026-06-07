"""CROWN news encoder (Flax NNX) — encoder-agnostic, two text views.

Mirror of the PyTorch CROWN news encoder. Pipeline (paper §3.1):
    1. text_encoder(news_idx) -> tokens, mask  (separate TextEncoder per view)
    2. Optional Linear(text_dim -> embedding_size) per view
    3. Positional encoding + dropout
    4. Transformer encoder layer
    5. Mask-weighted mean pool
    6. Category-aware concatenation with [cat | subcat] -> Linear
    7. k-intent disentanglement
    8. Additive attention over k intents
    9. Title-body cosine similarity scaling
    10. Concatenate ``[title_intent, sim * body_intent, cat_emb, subcat_emb]``

Auxiliary category-prediction loss is stored on the module via an
``nnx.Variable`` and surfaced through :meth:`CROWN.get_auxiliary_loss`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from src.core.models.configs import CROWNConfig

from ...layers import TextEncoder


class PositionalEncoding(nnx.Module):
    """Sinusoidal positional encoding."""

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int, *, rngs: nnx.Rngs
    ):
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
    """Standard transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
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


class NewsEncoder(nnx.Module):
    """CROWN news encoder. Outputs ``(*leading, news_embedding_dim)``."""

    def __init__(
        self,
        config: CROWNConfig,
        title_text_encoder: TextEncoder,
        abstract_text_encoder: TextEncoder,
        category_embedding: nnx.Embed,
        subcategory_embedding: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.title_text_encoder = title_text_encoder
        self.abstract_text_encoder = abstract_text_encoder
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        # Option-2 PLM: per-news Transformer + intent disentanglement
        # run at the text encoder's native dim (300 for GloVe, 768 for
        # BERT-base). The only post-pool projection is in intent_layers
        # (text_dim+cat_dim -> intent_dim). Preserves the BERT signal
        # through the per-news Transformer + mean pool.
        t_dim = title_text_encoder.output_dim
        a_dim = abstract_text_encoder.output_dim
        # CROWN shares Transformer blocks across views, so dims must match.
        if t_dim != a_dim:
            raise ValueError(
                f"CROWN expects title_text_encoder.output_dim ({t_dim}) "
                f"to match abstract_text_encoder.output_dim ({a_dim})."
            )
        if t_dim % config.num_heads != 0:
            raise ValueError(
                f"CROWN Transformer: text_dim={t_dim} not divisible by "
                f"num_heads={config.num_heads}. Override "
                "spec.model.architecture.news_encoder.num_heads in the "
                f"experiment yaml to a divisor of {t_dim} (e.g. 12 for "
                "BERT-base 768d -> head_dim 64)."
            )
        self.text_dim = int(t_dim)

        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.title_pos = PositionalEncoding(
            t_dim, config.dropout_rate, config.max_title_length, rngs=rngs
        )
        self.body_pos = PositionalEncoding(
            t_dim, config.dropout_rate, config.max_abstract_length, rngs=rngs
        )

        self.title_transformer = TransformerEncoderLayer(
            t_dim,
            config.num_heads,
            config.feedforward_dim,
            config.dropout_rate,
            rngs=rngs,
        )
        self.body_transformer = TransformerEncoderLayer(
            t_dim,
            config.num_heads,
            config.feedforward_dim,
            config.dropout_rate,
            rngs=rngs,
        )

        cat_concat_dim = (
            config.category_embedding_dim + config.subcategory_embedding_dim
        )
        self.category_affine = nnx.Linear(
            cat_concat_dim, config.category_embedding_dim, rngs=rngs
        )

        # Intent layers map text_dim+cat_dim -> intent_dim — the only
        # post-pool projection for CROWN.
        intent_input_dim = t_dim + config.category_embedding_dim
        self.intent_layers = nnx.List(
            [
                nnx.Linear(intent_input_dim, config.intent_embedding_dim, rngs=rngs)
                for _ in range(config.intent_num)
            ]
        )

        glorot = nnx.initializers.glorot_uniform()
        self.title_intent_W = nnx.Param(
            glorot(rngs.params(), (config.intent_embedding_dim, config.attention_dim))
        )
        self.title_intent_b = nnx.Param(jnp.zeros((config.attention_dim,)))
        self.title_intent_q = nnx.Param(
            glorot(rngs.params(), (config.attention_dim, 1))
        )
        self.body_intent_W = nnx.Param(
            glorot(rngs.params(), (config.intent_embedding_dim, config.attention_dim))
        )
        self.body_intent_b = nnx.Param(jnp.zeros((config.attention_dim,)))
        self.body_intent_q = nnx.Param(glorot(rngs.params(), (config.attention_dim, 1)))

        # Placeholder — ``CROWN.__init__`` overrides this once
        # ``num_categories`` is known.
        self.category_predictor = nnx.Linear(config.intent_embedding_dim, 1, rngs=rngs)

        self._aux_loss = nnx.Variable(jnp.float32(0.0))

    @staticmethod
    def valid_mask(packed_features: jax.Array) -> jax.Array:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero."""
        if packed_features.ndim >= 2 and packed_features.shape[-1] > 1:
            return jnp.not_equal(packed_features[..., 0], 0)
        return jnp.not_equal(packed_features, 0)

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
        packed_features: jax.Array,
        *,
        compute_aux_loss: bool = False,
        training: bool = False,
    ) -> jax.Array:
        """Encode a batch of news articles.

        Args:
            packed_features: ``(*leading, 3)`` int with column order
                ``[news_idx | category | subcategory]``.
        """
        det = not training

        leading_shape = packed_features.shape[:-1]
        flat = packed_features.reshape(-1, packed_features.shape[-1])
        news_idx = flat[:, 0].astype(jnp.int32)
        category_ids = flat[:, 1].astype(jnp.int32)
        subcategory_ids = flat[:, 2].astype(jnp.int32)

        title_tokens, title_mask = self.title_text_encoder(news_idx)
        body_tokens, body_mask = self.abstract_text_encoder(news_idx)
        title_emb = self.dropout(title_tokens, deterministic=det)  # (N, T, text_dim)
        body_emb = self.dropout(body_tokens, deterministic=det)
        title_mask = title_mask.astype(jnp.bool_)
        body_mask = body_mask.astype(jnp.bool_)

        title_emb = self.title_pos(title_emb, deterministic=det)
        body_emb = self.body_pos(body_emb, deterministic=det)

        title_enc = self.title_transformer(title_emb, deterministic=det)
        body_enc = self.body_transformer(body_emb, deterministic=det)

        # Unmasked mean pool — matches reference
        # ``crown-www25/newsEncoders.py:164,168`` which uses plain
        # ``.mean(dim=1)`` over the transformer output. The masked
        # variant flattened the intent signal in our ablations.
        title_pool = jnp.mean(title_enc, axis=1)
        body_pool = jnp.mean(body_enc, axis=1)

        cat_emb = self.category_embedding(category_ids)
        subcat_emb = self.subcategory_embedding(subcategory_ids)
        cat_repr = self.category_affine(jnp.concatenate([cat_emb, subcat_emb], axis=-1))

        cat_aware_title = jnp.concatenate([title_pool, cat_repr], axis=-1)
        cat_aware_body = jnp.concatenate([body_pool, cat_repr], axis=-1)

        title_k = self._k_intent_disentangle(cat_aware_title)
        body_k = self._k_intent_disentangle(cat_aware_body)

        title_intent = self._additive_attn(
            title_k, self.title_intent_W, self.title_intent_b, self.title_intent_q
        )
        body_intent = self._additive_attn(
            body_k, self.body_intent_W, self.body_intent_b, self.body_intent_q
        )

        if compute_aux_loss:
            logits = self.category_predictor(title_intent)
            self._aux_loss.value = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, category_ids)
            )

        sim = jnp.sum(title_intent * body_intent, axis=-1) / (
            jnp.linalg.norm(title_intent, axis=-1)
            * jnp.linalg.norm(body_intent, axis=-1)
            + 1e-8
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
        return news_repr.reshape(*leading_shape, self.news_embedding_dim)
