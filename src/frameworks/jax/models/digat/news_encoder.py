"""DIGAT news encoder (Flax NNX).

Mirror of the PyTorch ``NewsEncoder``: embedding → MHA → additive
attention.  ``MultiHeadSelfAttention`` is a custom NNX module (bias-free
K, biased Q/V) so the parameter layout matches reference DIGAT exactly.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import DIGATConfig


class AdditiveAttention(nnx.Module):
    """Additive (Bahdanau) attention for sequence pooling."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.affine = nnx.Linear(feature_dim, attention_dim, use_bias=True, rngs=rngs)
        self.project = nnx.Linear(attention_dim, 1, use_bias=False, rngs=rngs)

    def __call__(self, features: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # features: (B, N, F)
        scores = jnp.squeeze(
            self.project(jnp.tanh(self.affine(features))), axis=-1
        )  # (B, N)
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        weights = jax.nn.softmax(scores, axis=1)[:, None, :]  # (B, 1, N)
        return jnp.squeeze(jnp.matmul(weights, features), axis=1)  # (B, F)


class MultiHeadSelfAttention(nnx.Module):
    """Multi-head self-attention (news encoder MSA).

    Custom implementation (not ``nnx.MultiHeadAttention``) to match the
    PyTorch DIGAT reference exactly: a bias-free K projection and a
    biased Q/V projection, plus manual head reshape/transpose.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(float(head_dim))
        self.W_Q = nnx.Linear(d_model, self.out_dim, use_bias=True, rngs=rngs)
        self.W_K = nnx.Linear(d_model, self.out_dim, use_bias=False, rngs=rngs)
        self.W_V = nnx.Linear(d_model, self.out_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        # (B, L, H*D) → (B, L, H, D) → (B, H, L, D)
        Q = self.W_Q(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, D).transpose(0, 2, 1, 3)
        attn = jax.nn.softmax(
            jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale, axis=-1
        )
        out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, self.out_dim)
        return out


class NewsEncoder(nnx.Module):
    """MSA-based news encoder: embedding → MHA → additive attention.

    GloVe mode reads token ids ``(B, num_news, T)`` and embeds them.
    PLM mode reads parsed news_idx ``(B, num_news)`` and looks up the
    cached PLM token features + attention mask via :class:`PLMTokenLookup`.
    """

    def __init__(
        self,
        config: DIGATConfig,
        word_embedding: nnx.Module,
        *,
        rngs: nnx.Rngs,
        encoder_type: str = "glove",
    ):
        self.encoder_type = encoder_type
        self.word_embedding = word_embedding
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.msa = MultiHeadSelfAttention(
            config.embedding_size,
            config.msa_head_num,
            config.msa_head_dim,
            rngs=rngs,
        )
        self.news_embedding_dim = config.news_embedding_dim
        self.attention = AdditiveAttention(
            self.news_embedding_dim,
            config.attention_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        title_text: jax.Array,
        title_mask: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode a batch of news with SAG neighbor structure.

        Args:
            title_text: GloVe — ``(batch_size, num_news, title_len)`` ids.
                PLM — ``(batch_size, num_news)`` parsed news_idx.
            title_mask: GloVe — optional ``(B, num_news, title_len)`` mask
                (1 = valid). Ignored under PLM.

        Returns:
            ``(batch_size, num_news, news_embedding_dim)`` news reprs.
        """
        det = not training
        if self.encoder_type == "glove":
            batch_size, num_news, title_len = title_text.shape
            flat_text = title_text.reshape(batch_size * num_news, title_len)
            flat_mask = (
                title_mask.reshape(batch_size * num_news, title_len)
                if title_mask is not None
                else None
            )
            w = self.dropout(self.word_embedding(flat_text), deterministic=det)
        else:
            batch_size, num_news = title_text.shape
            flat_idx = title_text.reshape(batch_size * num_news).astype(jnp.int32)
            tokens, mask = self.word_embedding(flat_idx)
            w = self.dropout(tokens, deterministic=det)
            flat_mask = mask.astype(jnp.float32)

        h = jax.nn.relu(self.msa(w))
        news_repr = self.attention(h, mask=flat_mask)
        return news_repr.reshape(batch_size, num_news, self.news_embedding_dim)
