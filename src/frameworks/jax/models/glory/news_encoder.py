"""GLORY local news encoder + attention primitives (Flax NNX).

Mirror of the PyTorch ``news_encoder`` module. The attention primitives
(``ScaledDotProductAttention``, ``MultiHeadAttention``, ``AttentionPooling``)
follow GLORY's reference ``src/models/base/layers.py`` exactly —
including the non-standard masking scheme (``exp`` then mask) — and are
shared by every other encoder in the package.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import GLORYConfig


class ScaledDotProductAttention(nnx.Module):
    """Scaled dot-product attention used inside ``MultiHeadAttention``.

    Note: GLORY's reference uses a non-standard masking scheme — scores
    are exponentiated before masking, then re-normalised.  We replicate
    that behaviour exactly to match the paper's initialization and
    training dynamics.
    """

    def __init__(self, d_k: int):
        self.d_k = d_k

    def __call__(
        self,
        Q: jax.Array,
        K: jax.Array,
        V: jax.Array,
        attn_mask: jax.Array | None = None,
    ) -> jax.Array:
        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / math.sqrt(self.d_k)
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        scores = jnp.exp(scores)
        if attn_mask is not None:
            scores = scores * jnp.expand_dims(attn_mask, axis=-2)
        attn = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-8)
        return jnp.matmul(attn, V)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention matching GLORY's Q/K/V projection choice."""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        head_num: int,
        head_dim: int,
        residual: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.head_num = head_num
        self.head_dim = head_dim
        self.residual = residual
        out_dim = head_num * head_dim

        xavier = nnx.initializers.xavier_uniform()
        self.W_Q = nnx.Linear(
            key_size,
            out_dim,
            use_bias=True,
            kernel_init=xavier,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.W_K = nnx.Linear(
            query_size,
            out_dim,
            use_bias=False,
            kernel_init=xavier,
            rngs=rngs,
        )
        self.W_V = nnx.Linear(
            value_size,
            out_dim,
            use_bias=True,
            kernel_init=xavier,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.attn = ScaledDotProductAttention(head_dim)

    def __call__(
        self,
        Q: jax.Array,
        K: jax.Array | None = None,
        V: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.shape[0]
        H, D = self.head_num, self.head_dim
        if mask is not None:
            mask = jnp.broadcast_to(
                jnp.expand_dims(mask, axis=1), (batch_size, H, mask.shape[-1])
            )
        q = self.W_Q(Q).reshape(batch_size, -1, H, D).transpose(0, 2, 1, 3)
        k = self.W_K(K).reshape(batch_size, -1, H, D).transpose(0, 2, 1, 3)
        v = self.W_V(V).reshape(batch_size, -1, H, D).transpose(0, 2, 1, 3)
        ctx = self.attn(q, k, v, mask)
        out = ctx.transpose(0, 2, 1, 3).reshape(batch_size, -1, H * D)
        return out + Q if self.residual else out


class AttentionPooling(nnx.Module):
    """Additive attention pooling over a sequence of vectors.

    Follows GLORY's implementation exactly:
    ``alpha = softmax(v^T tanh(W x)) / (sum + eps)`` with multiplicative
    masking (consistent with :class:`ScaledDotProductAttention`).
    """

    def __init__(self, emb_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        xavier_tanh = nnx.initializers.variance_scaling(
            (5.0 / 3.0) ** 2,
            "fan_avg",
            "uniform",
        )
        self.att_fc1 = nnx.Linear(
            emb_size,
            hidden_size,
            use_bias=True,
            kernel_init=xavier_tanh,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.att_fc2 = nnx.Linear(
            hidden_size,
            1,
            use_bias=True,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, attn_mask: jax.Array | None = None) -> jax.Array:
        # x: (B, N, E)
        e = jnp.tanh(self.att_fc1(x))
        raw = self.att_fc2(e)  # (B, N, 1)
        alpha = jnp.exp(raw - jnp.max(raw, axis=1, keepdims=True))
        if attn_mask is not None:
            alpha = alpha * jnp.expand_dims(attn_mask, axis=2)
        alpha = alpha / (jnp.sum(alpha, axis=1, keepdims=True) + 1e-8)
        # (B, E, N) @ (B, N, 1) → (B, E, 1) → (B, E)
        return jnp.squeeze(jnp.matmul(jnp.swapaxes(x, 1, 2), alpha), axis=-1)


class NewsEncoder(nnx.Module):
    """Local news encoder: word emb → dropout → MHA → LN → drop → pool → LN.

    Consumes a ``(*, T + E + 1 + 1 + 1)`` feature tensor where the columns
    are ``[title tokens (T), entity ids (E), category, subcategory,
    news_index]``.  Only the title tokens are used here.
    """

    def __init__(
        self,
        config: GLORYConfig,
        word_embedding: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.word_embedding = word_embedding
        self.news_dim = config.head_num * config.head_dim
        self.title_size = config.title_size
        self.entity_size = config.entity_size

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.msa = MultiHeadAttention(
            config.word_emb_dim,
            config.word_emb_dim,
            config.word_emb_dim,
            config.head_num,
            config.head_dim,
            rngs=rngs,
        )
        self.layernorm1 = nnx.LayerNorm(self.news_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.attn_pool = AttentionPooling(
            self.news_dim,
            config.attention_hidden_dim,
            rngs=rngs,
        )
        self.layernorm2 = nnx.LayerNorm(self.news_dim, rngs=rngs)

    def __call__(
        self,
        news_input: jax.Array,
        mask: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        det = not training
        B, N = news_input.shape[:2]
        title_tokens = news_input[..., : self.title_size].astype(jnp.int32)
        flat_title = title_tokens.reshape(B * N, self.title_size)

        word_emb = self.dropout1(
            self.word_embedding(flat_title),
            deterministic=det,
        )

        attn_out = self.msa(word_emb, word_emb, word_emb, mask)
        attn_out = self.layernorm1(attn_out)
        attn_out = self.dropout2(attn_out, deterministic=det)

        pooled = self.attn_pool(attn_out, mask)
        pooled = self.layernorm2(pooled)

        return pooled.reshape(B, N, self.news_dim)
