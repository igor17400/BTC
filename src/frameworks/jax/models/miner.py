"""MINER (Multi-Interest Matching Network) -- Flax NNX.

Reference: Li et al., "MINER: Multi-Interest Matching Network for News
Recommendation", Findings of ACL 2022.

Key ideas:
- Poly attention: K learnable context codes attend over clicked news
  embeddings to extract K interest vectors per user.
- Disagreement regularization: minimizes cosine similarity between
  interest vector pairs to encourage diversity.
- MINER-weighted: target-aware aggregation of per-interest scores.

Training uses MINER-weighted with disagreement regularization.
Evaluation uses multi-interest scoring via the MINER evaluator.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import MINERConfig

from ..layers import AdditiveAttention
from .base import BaseModel

# ---------------------------------------------------------------------------
# News encoder
# ---------------------------------------------------------------------------


class NewsEncoder(nnx.Module):
    """Encode a news article from its title token sequence.

    Pipeline: Embedding -> Dropout -> MultiHeadSelfAttention -> Dropout
    -> AdditiveAttention -> news vector.
    """

    def __init__(
        self,
        config: MINERConfig,
        embedding_layer: nnx.Embed,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embedding_layer = embedding_layer

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.multi_head_attention = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.embedding_size,
            qkv_features=config.num_heads * config.head_dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.additive_attention = AdditiveAttention(
            input_dim=config.embedding_size,
            query_vec_dim=config.attention_hidden_dim,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """inputs: (batch, title_length) -> (batch, embedding_size)."""
        embedded = self.embedding_layer(inputs)
        y = self.dropout1(embedded, deterministic=not training)

        padding_mask = jnp.not_equal(inputs, 0)
        attn_mask = padding_mask[:, None, None, :]

        y = self.multi_head_attention(y, y, mask=attn_mask, deterministic=not training)
        y = self.dropout2(y, deterministic=not training)

        return self.additive_attention(y, mask=padding_mask)


# ---------------------------------------------------------------------------
# Poly Attention — multi-interest extraction
# ---------------------------------------------------------------------------


class PolyAttention(nnx.Module):
    """Extract K interest vectors from clicked news via learnable context codes.

    Each context code c_i attends over the M clicked news embeddings:
        e_i = sum_j w_j^{c_i} * h_j
        w_j^{c_i} = softmax(c_i^T tanh(W^h h_j))
    """

    def __init__(self, config: MINERConfig, *, rngs: nnx.Rngs):
        K = config.num_interest_vectors
        E = config.embedding_size
        D = config.context_code_dim

        # Reference uses xavier_uniform_ with gain=tanh (~1.667)
        glorot = nnx.initializers.glorot_uniform()
        tanh_gain = 5.0 / 3.0
        key_cc = rngs.params()

        self.context_codes = nnx.Param(glorot(key_cc, (K, D)) * tanh_gain)
        # Reference uses bias=False for the projection
        self.W_h = nnx.Param(glorot(rngs.params(), (E, D)))

    def __call__(
        self,
        news_embeddings: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Extract K interest vectors.

        Args:
            news_embeddings: ``(B, M, E)`` clicked news embeddings.
            mask: ``(B, M)`` bool mask, True where news is present.

        Returns:
            ``(B, K, E)`` interest vectors.
        """
        # No bias (matching reference)
        projected = jnp.tanh(jnp.matmul(news_embeddings, self.W_h.value))  # (B, M, D)

        logits = jnp.einsum("bmd,kd->bkm", projected, self.context_codes.value)

        if mask is not None:
            logits = jnp.where(mask[:, None, :], logits, -1e9)

        weights = jax.nn.softmax(logits, axis=-1)  # (B, K, M)

        return jnp.einsum("bkm,bme->bke", weights, news_embeddings)  # (B, K, E)


# ---------------------------------------------------------------------------
# User encoder
# ---------------------------------------------------------------------------


class UserEncoder(nnx.Module):
    """Encode user history into interest vectors.

    Returns mean of K interest vectors for eval compatibility with the
    default evaluator. The MINER evaluator calls ``encode_interests()``
    to get the full K vectors.
    """

    def __init__(
        self,
        config: MINERConfig,
        news_encoder: NewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder
        self.poly_attention = PolyAttention(config, rngs=rngs)

    def _encode_news_history(
        self, inputs: jax.Array, *, training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """Encode history and return (news_embeds, mask)."""
        B, H, T = inputs.shape
        flat = inputs.reshape(B * H, T)
        flat_vecs = self.news_encoder(flat, training=training)
        news_embeds = flat_vecs.reshape(B, H, -1)
        mask = jnp.any(jnp.not_equal(inputs, 0), axis=-1)
        return news_embeds, mask

    def encode_interests(
        self, inputs: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Encode history into K interest vectors.

        Returns:
            ``(B, K, E)`` interest vectors.
        """
        news_embeds, mask = self._encode_news_history(inputs, training=training)
        return self.poly_attention(news_embeds, mask=mask)

    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        """Encode history into a single user vector (mean of K interests)."""
        interest_vecs = self.encode_interests(inputs, training=training)
        return jnp.mean(interest_vecs, axis=1)


# ---------------------------------------------------------------------------
# Disagreement regularization (pure function, no stop_gradient)
# ---------------------------------------------------------------------------


def _disagreement_loss(interest_vecs: jax.Array) -> jax.Array:
    """Compute L_D (Eq. 3) — mean off-diagonal cosine similarity.

    Uses stop_gradient on norms for numerical stability in JAX.
    The gradient still flows through the numerator (interest_vecs)
    which pushes vectors to be orthogonal.

    Args:
        interest_vecs: ``(B, K, E)`` interest vectors.

    Returns:
        Scalar loss (no beta scaling — caller handles that).
    """
    K = interest_vecs.shape[1]

    norms = jnp.linalg.norm(interest_vecs, axis=-1, keepdims=True) + 1e-8
    normalized = interest_vecs / jax.lax.stop_gradient(norms)

    cos_sim = jnp.matmul(normalized, normalized.transpose(0, 2, 1))  # (B, K, K)

    # Zero out diagonal
    diag_mask = 1.0 - jnp.eye(K)
    cos_sim = cos_sim * diag_mask[None, :, :]

    return jnp.mean(cos_sim)


# ---------------------------------------------------------------------------
# Full MINER model
# ---------------------------------------------------------------------------


class MINER(BaseModel):
    """Multi-Interest Matching Network for News Recommendation.

    Training: MINER-weighted scoring. Disagreement regularization is
    computed and returned via ``get_auxiliary_loss()``.
    Evaluation: multi-interest scoring via the MINER evaluator.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: MINERConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = MINERConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        # Shared word embedding
        embeddings_matrix = np.asarray(processed_news["embeddings"])
        vocab_size = int(processed_news["vocab_size"])
        self.embedding_layer = nnx.Embed(
            num_embeddings=vocab_size,
            features=config.embedding_size,
            rngs=rngs,
        )
        self.embedding_layer.embedding.value = jnp.asarray(embeddings_matrix)

        # Sub-modules
        self.news_encoder = NewsEncoder(config, self.embedding_layer, rngs=rngs)
        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)

        # Target-aware attention: projects INTERESTS (not candidates)
        # Reference: proj = gelu(linear(interests)), then candidates attend
        E = config.embedding_size
        self.W_e = nnx.Linear(E, E, rngs=rngs)

        # Cached disagreement loss for get_auxiliary_loss()
        # Using nnx.Variable so NNX can track it through JIT, but the
        # optimizer (wrt=nnx.Param) won't update it.
        self._cached_d_loss = nnx.Variable(jnp.float32(0.0))

    def get_auxiliary_loss(self) -> jax.Array:
        """Return beta * L_D computed during the last forward pass."""
        return self.config.disagreement_beta * self._cached_d_loss.value

    def score_training_batch(
        self,
        hist_tokens: jax.Array,
        cand_tokens: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """MINER-weighted training scoring.

        Disagreement loss is computed and stored in ``_cached_d_loss``
        for ``get_auxiliary_loss()`` to return to the training loop,
        which adds it to the main CE loss (not to scores).

        Returns:
            ``(B, C)`` raw logit scores.
        """
        # Get K interest vectors through user_encoder's clean path
        interest_vecs = self.user_encoder.encode_interests(
            hist_tokens, training=training
        )  # (B, K, E)

        # Compute disagreement and cache it (added to loss by training loop)
        if training and self.config.disagreement_beta > 0:
            self._cached_d_loss.value = _disagreement_loss(interest_vecs)

        # Encode candidates through news_encoder
        B, C, T = cand_tokens.shape
        flat_cands = cand_tokens.reshape(B * C, T)
        flat_cand_vecs = self.news_encoder(flat_cands, training=training)
        cand_embeds = flat_cand_vecs.reshape(B, C, -1)  # (B, C, E)

        # Per-interest scores: s_i = e_i^T h^c -> (B, K, C)
        interest_scores = jnp.einsum("bke,bce->bkc", interest_vecs, cand_embeds)

        # Target-aware attention: project INTERESTS (not candidates)
        # Reference: proj = gelu(linear(query=interests))
        # then: weights = softmax(candidates @ proj^T, dim over K)
        projected_interests = jax.nn.gelu(self.W_e(interest_vecs))  # (B, K, E)

        # Each candidate attends over K projected interests: (B, C, K)
        agg_logits = jnp.einsum("bce,bke->bck", cand_embeds, projected_interests)
        agg_weights = jax.nn.softmax(agg_logits, axis=2)  # softmax over K

        # interest_scores transposed to (B, C, K) for element-wise multiply
        scores = jnp.sum(
            agg_weights * interest_scores.transpose(0, 2, 1), axis=2
        )  # (B, C)

        return scores

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        return self.score_training_batch(
            inputs["hist_tokens"], inputs["cand_tokens"], training=training
        )
