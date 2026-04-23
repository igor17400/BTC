"""JAX/Flax NNX implementation of the framework adapter Protocol.

The shared evaluator at :mod:`src.core.models.evaluation` invokes JAX
encoder modules through this adapter and converts JAX arrays to numpy.
This is the only place in the JAX path that needs to know about the
shared evaluation pipeline; everything downstream is pure numpy.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# DIGAT eval pads each impression's candidate count up to this value so
# the JIT-compiled scoring function has a single stable shape.  7824 val
# + 72903 test MIND-small impressions average ~37 candidates with a long
# tail; 256 covers the maximum seen in the dataset while keeping memory
# reasonable (~3 MB per impression for the user graph).  Padded
# candidates receive zero embeddings and identity (self-loop) adjacency,
# and their scores are discarded after the call.
_DIGAT_MAX_C = 512


@nnx.jit(static_argnums=(8,))
def _score_digat_impression_core(
    graph_encoder: nnx.Module,
    cand_sag_emb: jax.Array,  # (max_C, G_n, D)
    cand_sag_graph: jax.Array,  # (max_C, G_n, G_n)
    cand_sag_mask: jax.Array,  # (max_C, G_n)
    user_hist_emb: jax.Array,  # (H, D)
    u_graph: jax.Array,  # (G_u, G_u)
    u_cat_mask: jax.Array,  # (num_cat,)
    u_cat_indices: jax.Array,  # (H,)
    num_categories: int,  # static
) -> jax.Array:
    """JIT-compiled per-impression scoring with fixed max candidate count.

    Called for every impression in eval.  Stable shapes → one
    compilation, cached and reused across all 7824+ impressions.
    """
    C = cand_sag_emb.shape[0]
    H = user_hist_emb.shape[0]
    D = user_hist_emb.shape[1]
    num_cat = u_cat_mask.shape[0]
    G_u = u_graph.shape[0]

    hist_b = jnp.broadcast_to(user_hist_emb[None, :, :], (C, H, D))
    u_graph_b = jnp.broadcast_to(u_graph[None, :, :], (C, G_u, G_u))
    u_cat_mask_b = jnp.broadcast_to(u_cat_mask[None, :], (C, num_cat))
    u_cat_idx_b = jnp.broadcast_to(u_cat_indices[None, :], (C, H))

    news_ctx, user_ctx = graph_encoder(
        cand_sag_emb,
        cand_sag_graph,
        cand_sag_mask,
        hist_b,
        u_graph_b,
        u_cat_mask_b,
        u_cat_idx_b,
        num_categories,
        training=False,
    )
    return jnp.sum(news_ctx * user_ctx, axis=-1)


class JAXAdapter:
    """Framework adapter for Flax NNX news recommendation models."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a JAX array (or anything array-like) to numpy."""
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run the news encoder on a feature batch and return numpy vectors."""
        if not isinstance(features, jnp.ndarray):
            features = jnp.asarray(features)
        return np.asarray(encoder(features, training=False))

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run the user encoder on a history batch and return numpy vectors.

        For LSTUR-style encoders (``process_user_id=True``) the encoder is
        called as ``encoder(features, user_ids, training=False)``. For all
        other models it's ``encoder(features, training=False)``.
        """
        if not isinstance(features, jnp.ndarray):
            features = jnp.asarray(features)
        if process_user_id:
            if not isinstance(user_ids, jnp.ndarray):
                user_ids = jnp.asarray(user_ids)
            vec = encoder(features, user_ids, training=False)
        else:
            vec = encoder(features, training=False)
        return np.asarray(vec)

    def run_activity_gater(self, gater: Any, user_vecs: Any) -> np.ndarray:
        """Run the ActivityGater on a batch of user vectors.

        Args:
            gater: Framework-native ActivityGater module.
            user_vecs: ``(B, news_dim)`` user vectors.

        Returns:
            ``(B,)`` numpy array of gate values (eta) in [0, 1].
        """
        user_jnp = jnp.asarray(user_vecs)
        return np.asarray(gater(user_jnp))

    def run_popularity_predictor(
        self,
        predictor: Any,
        bias_vecs: Any,
        recency: Any | None,
        ctr: Any | None,
    ) -> np.ndarray:
        """Run the PopularityPredictor with full inputs (bias + recency + CTR).

        Args:
            predictor: Framework-native PopularityPredictor module.
            bias_vecs: ``(C, news_dim)`` bias news vectors.
            recency: Optional ``(C,)`` int recency bucket indices.
            ctr: Optional ``(C,)`` float CTR values.

        Returns:
            ``(C,)`` numpy array of popularity scores.
        """
        bias_jnp = jnp.asarray(bias_vecs)
        recency_jnp = (
            jnp.asarray(recency).astype(jnp.int32) if recency is not None else None
        )
        ctr_jnp = jnp.asarray(ctr).astype(jnp.float32) if ctr is not None else None
        pop_scores = predictor(
            bias_jnp, recency_indices=recency_jnp, ctr_values=ctr_jnp
        )
        return np.asarray(pop_scores)

    # ------------------------------------------------------------------
    # DIGAT-specific methods
    # ------------------------------------------------------------------

    def encode_digat_news(
        self,
        news_encoder: Any,
        tokens: Any,
        mask: Any,
    ) -> np.ndarray:
        """Run the DIGAT MSA news encoder on a raw token batch."""
        tokens_j = jnp.asarray(tokens).astype(jnp.int32)
        mask_j = jnp.asarray(mask).astype(jnp.float32)
        # News encoder expects (B, N, T) — treat each news as a single-item batch.
        emb = news_encoder(tokens_j[:, None, :], mask_j[:, None, :], training=False)
        return np.asarray(jnp.squeeze(emb, axis=1))

    def encode_digat_graph_context(
        self,
        graph_encoder: Any,
        sag_emb: Any,
        sag_mask: Any,
    ) -> np.ndarray:
        """Compute SAG graph context vectors for a batch of news."""
        sag_emb_j = jnp.asarray(sag_emb).astype(jnp.float32)
        sag_mask_j = jnp.asarray(sag_mask).astype(jnp.float32)
        ctx = graph_encoder._news_graph_context(
            sag_emb_j,
            sag_mask_j,
            deterministic=True,
        )
        return np.asarray(ctx)

    def score_digat_impression(
        self,
        graph_encoder: Any,
        cand_sag_emb: Any,
        cand_sag_graph: Any,
        cand_sag_mask: Any,
        user_hist_emb: Any,
        u_graph: Any,
        u_cat_mask: Any,
        u_cat_indices: Any,
        num_categories: int,
    ) -> np.ndarray:
        """Run dual graph interaction and return per-candidate scores.

        Candidates are padded up to ``_DIGAT_MAX_C`` before invoking a
        JIT-compiled core.  Padded slots contribute meaningless scores
        which are discarded via the final slice — they exist only so
        the compiled function can be reused across impressions with
        varying candidate counts.  Without this, JAX's eager-mode per-op
        latency in a Python loop over thousands of impressions makes
        eval two orders of magnitude slower than PyTorch.
        """
        C = int(cand_sag_emb.shape[0])
        if C > _DIGAT_MAX_C:
            raise ValueError(
                f"Impression has {C} candidates, exceeds _DIGAT_MAX_C={_DIGAT_MAX_C}. "
                f"Bump the constant if your dataset has larger impressions."
            )

        G_n = cand_sag_emb.shape[1]
        D = cand_sag_emb.shape[2]
        pad = _DIGAT_MAX_C - C

        if pad > 0:
            # Padded candidates get zero embeddings and identity
            # adjacency (self-loops only) so the graph encoder produces
            # finite values.  Their scores are discarded.
            cand_sag_emb = np.concatenate(
                [cand_sag_emb, np.zeros((pad, G_n, D), dtype=np.float32)],
                axis=0,
            )
            eye = np.eye(G_n, dtype=np.float32)
            cand_sag_graph = np.concatenate(
                [cand_sag_graph, np.broadcast_to(eye[None], (pad, G_n, G_n))],
                axis=0,
            )
            cand_sag_mask = np.concatenate(
                [cand_sag_mask, np.ones((pad, G_n), dtype=np.float32)],
                axis=0,
            )

        cand_emb_j = jnp.asarray(cand_sag_emb, dtype=jnp.float32)
        cand_graph_j = jnp.asarray(cand_sag_graph, dtype=jnp.float32)
        cand_mask_j = jnp.asarray(cand_sag_mask, dtype=jnp.float32)
        hist_j = jnp.asarray(user_hist_emb, dtype=jnp.float32)
        u_graph_j = jnp.asarray(u_graph, dtype=jnp.float32)
        u_cat_mask_j = jnp.asarray(u_cat_mask, dtype=jnp.float32)
        u_cat_idx_j = jnp.asarray(u_cat_indices, dtype=jnp.int32)

        scores = _score_digat_impression_core(
            graph_encoder,
            cand_emb_j,
            cand_graph_j,
            cand_mask_j,
            hist_j,
            u_graph_j,
            u_cat_mask_j,
            u_cat_idx_j,
            num_categories,
        )
        return np.asarray(scores)[:C]
