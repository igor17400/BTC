"""CAUM candidate-aware interaction model (Flax NNX).

Mirror of the PyTorch :mod:`inter_model`. Three candidate-aware modules
combine clicked-news vectors with a candidate vector to produce a scalar
matching score:

    1. Candi-CNN     — circular-shift window over clicks + candidate -> Dense
    2. Candi-SelfAtt — concat candidate with each click -> Dense -> MHSA
    3. Candi-Att     — DNN scorer over [fused, candidate] -> softmax
                       -> weighted sum (user_vec)
    4. Score         — dot(user_vec, candidate)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import CAUMConfig


class DenseAttentionScorer(nnx.Module):
    """Two-layer MLP scorer for Candi-Att."""

    def __init__(self, hidden_dim: int, mid_dim: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, mid_dim, rngs=rngs)
        self.dense3 = nnx.Linear(mid_dim, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """inputs: ``(*, 2 * hidden_dim)`` → ``(*, 1)``."""
        return self.dense3(jnp.tanh(self.dense2(jnp.tanh(self.dense1(inputs)))))


class InterModel(nnx.Module):
    """Candidate-aware user interest model.

    Takes pre-encoded clicked-news vectors and a single candidate news
    vector, produces a scalar matching score. Operates entirely in the
    ``news_dim`` space so it does not need to know whether GloVe or PLM
    was used upstream.
    """

    def __init__(self, config: CAUMConfig, *, rngs: nnx.Rngs):
        D = config.news_dim
        self.config = config

        self.dropout_cand = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.dropout_clicks = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # Candi-CNN: projects [left, center, right, candidate] -> D
        self.cnn_projection = nnx.Linear(4 * D, D, rngs=rngs)

        # Candi-SelfAtt: projects [candidate, click] -> D, then MHSA
        self.selfatt_input_projection = nnx.Linear(2 * D, D, rngs=rngs)
        self.selfatt_mha = nnx.MultiHeadAttention(
            num_heads=config.candi_selfatt_num_heads,
            in_features=D,
            qkv_features=(
                config.candi_selfatt_num_heads * config.candi_selfatt_head_dim
            ),
            decode=False,
            rngs=rngs,
        )

        # Fusion
        self.fusion_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.fusion_projection = nnx.Linear(2 * D, D, rngs=rngs)

        # Candi-Att (DNN scorer applied per click)
        self.dense_att = DenseAttentionScorer(
            hidden_dim=config.candi_att_hidden_dim,
            mid_dim=config.candi_att_mid_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        cand_vec: jax.Array,
        clicked_vecs: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Score a candidate against clicked news.

        Args:
            cand_vec: ``(B, D)`` single candidate news vector.
            clicked_vecs: ``(B, H, D)`` encoded clicked news vectors.

        Returns:
            ``(B,)`` scalar matching scores.
        """
        D = self.config.news_dim
        B, H = clicked_vecs.shape[:2]

        can_vec_dropped = self.dropout_cand(cand_vec, deterministic=not training)
        user_vecs = self.dropout_clicks(clicked_vecs, deterministic=not training)

        cand_repeated = jnp.repeat(cand_vec[:, None, :], H, axis=1)

        # Candi-CNN — circular-shift window.
        left = jnp.concatenate([user_vecs[:, -1:, :], user_vecs[:, :-1, :]], axis=1)
        right = jnp.concatenate([user_vecs[:, 1:, :], user_vecs[:, :1, :]], axis=1)
        cnn_input = jnp.concatenate(
            [left, user_vecs, right, cand_repeated], axis=-1
        )  # (B, H, 4*D)
        cnn_out = self.cnn_projection(cnn_input)  # (B, H, D)

        # Candi-SelfAtt — concat candidate with each click, project, MHSA.
        selfatt_input = jnp.concatenate(
            [cand_repeated, user_vecs], axis=-1
        )  # (B, H, 2*D)
        selfatt_input = self.selfatt_input_projection(selfatt_input)

        history_mask = jnp.any(jnp.not_equal(clicked_vecs, 0.0), axis=-1)  # (B, H)
        attn_mask = history_mask[:, None, None, :]
        selfatt_out = self.selfatt_mha(
            selfatt_input,
            selfatt_input,
            mask=attn_mask,
            deterministic=not training,
        )

        # Fusion
        fused = jnp.concatenate([cnn_out, selfatt_out], axis=-1)
        fused = self.fusion_dropout(fused, deterministic=not training)
        fused = self.fusion_projection(fused)

        # Candi-Att — score each fused click against the candidate.
        att_input = jnp.concatenate([fused, cand_repeated], axis=-1)
        flat_att = att_input.reshape(B * H, 2 * D)
        flat_scores = self.dense_att(flat_att)
        att_scores = flat_scores.reshape(B, H)

        att_scores = jnp.where(history_mask, att_scores, -1e9)
        att_weights = jax.nn.softmax(att_scores, axis=-1)

        user_vec = jnp.sum(fused * att_weights[:, :, None], axis=1)
        return jnp.sum(user_vec * can_vec_dropped, axis=-1)
