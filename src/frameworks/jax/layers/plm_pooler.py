"""Pluggable poolers over per-news token sequences (Flax NNX).

Mirror of :mod:`src.frameworks.pytorch.layers.plm_pooler`. Collapses a
``(B, T, D)`` token sequence into a per-news ``(B, D)`` vector.

- ``mean``  — attention-mask-weighted mean over tokens
- ``cls``   — first token's vector
- ``avg_first_last`` — treated as ``mean`` here (we only cache the last layer)
- ``attention`` — learnable MHA + AdditiveAttention pool (IP2 style)
- ``gate``  — gated pool
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from .attention_layers import AdditiveAttention

logger = logging.getLogger(__name__)


class Pooler(nnx.Module):
    """Pools ``(B, T, D)`` token features into ``(B, D)`` per-news vectors."""

    SUPPORTED = ("mean", "cls", "attention", "gate", "avg_first_last")

    def __init__(
        self,
        plm_dim: int,
        pooler_type: str = "mean",
        *,
        attention_query_dim: int = 200,
        num_heads: int = 6,
        head_dim: int = 128,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ):
        if pooler_type not in self.SUPPORTED:
            raise ValueError(
                f"Unknown pooler_type: {pooler_type!r}. Supported: {self.SUPPORTED}"
            )
        self.pooler_type = pooler_type
        self.plm_dim = plm_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        if pooler_type == "attention":
            inner_dim = num_heads * head_dim
            self.norm = nnx.LayerNorm(num_features=plm_dim, rngs=rngs)
            self.attn_in = nnx.Linear(plm_dim, inner_dim, rngs=rngs)
            self.attn_mha = nnx.MultiHeadAttention(
                num_heads=num_heads,
                in_features=inner_dim,
                qkv_features=inner_dim,
                decode=False,
                rngs=rngs,
            )
            self.attn_out = nnx.Linear(inner_dim, plm_dim, rngs=rngs)
            self.additive = AdditiveAttention(
                input_dim=plm_dim,
                query_vec_dim=attention_query_dim,
                rngs=rngs,
            )
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        elif pooler_type == "gate":
            inner_dim = num_heads * head_dim
            self.norm = nnx.LayerNorm(num_features=plm_dim, rngs=rngs)
            self.attn_in = nnx.Linear(plm_dim, inner_dim, rngs=rngs)
            self.attn_mha = nnx.MultiHeadAttention(
                num_heads=num_heads,
                in_features=inner_dim,
                qkv_features=inner_dim,
                decode=False,
                rngs=rngs,
            )
            self.attn_out = nnx.Linear(inner_dim, plm_dim, rngs=rngs)
            self.fc1 = nnx.Linear(plm_dim, 300, rngs=rngs)
            self.fc2 = nnx.Linear(300, 1, use_bias=False, rngs=rngs)
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        elif pooler_type == "avg_first_last":
            logger.warning(
                "avg_first_last pooler requires caching the first layer too; "
                "falling back to 'mean' pooling on the last layer."
            )

    def __call__(
        self,
        token_features: jax.Array,
        attention_mask: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Pool token features.

        Args:
            token_features: ``(B, T, plm_dim)`` per-token vectors.
            attention_mask: ``(B, T)`` 0/1 mask (1 = real token).
            training: Enable dropout.

        Returns:
            ``(B, plm_dim)`` per-news vectors.
        """
        mask_f = attention_mask.astype(token_features.dtype)[..., None]

        if self.pooler_type in ("mean", "avg_first_last"):
            denom = jnp.clip(
                attention_mask.astype(token_features.dtype).sum(axis=1, keepdims=True),
                a_min=1.0,
            )
            return (token_features * mask_f).sum(axis=1) / denom

        if self.pooler_type == "cls":
            return token_features[:, 0, :]

        if self.pooler_type == "attention":
            mha_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            x = self.attn_in(token_features)
            attn_out = self.attn_mha(x, x, mask=mha_mask, deterministic=not training)
            attn_out = self.attn_out(attn_out)
            attn_out = self.dropout(attn_out, deterministic=not training)
            normed = self.norm(token_features + attn_out)
            return self.additive(normed, mask=attention_mask.astype(jnp.bool_))

        if self.pooler_type == "gate":
            mha_mask = attention_mask[:, None, None, :]
            x = self.attn_in(token_features)
            attn_out = self.attn_mha(x, x, mask=mha_mask, deterministic=not training)
            attn_out = self.attn_out(attn_out)
            attn_out = self.dropout(attn_out, deterministic=not training)
            h_tilde = self.norm(token_features + attn_out)
            r = jax.nn.softmax(self.fc2(jnp.tanh(self.fc1(h_tilde))), axis=1)
            r = r * mask_f
            r = r / jnp.clip(r.sum(axis=1, keepdims=True), a_min=1e-6)
            return (r * h_tilde).sum(axis=1)

        raise RuntimeError(f"Unreachable: pooler_type={self.pooler_type}")
