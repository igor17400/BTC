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
    # MINER-specific methods
    # ------------------------------------------------------------------

    def encode_user_interests(self, user_encoder: Any, features: Any) -> np.ndarray:
        """Run the MINER user encoder to get K interest vectors.

        Args:
            user_encoder: MINER UserEncoder with ``encode_interests`` method.
            features: ``(B, H, T)`` history token ids.

        Returns:
            ``(B, K, E)`` numpy array of interest vectors.
        """
        if not isinstance(features, jnp.ndarray):
            features = jnp.asarray(features)
        return np.asarray(user_encoder.encode_interests(features, training=False))

    # ------------------------------------------------------------------
    # TCCM-specific methods
    # ------------------------------------------------------------------

    def run_tccm_popularity_encoder(
        self,
        encoder: Any,
        bucket_input: Any,
        time_input: Any,
        title_len: int,
    ) -> np.ndarray:
        """Run the TCCM popularity encoder on a candidate batch.

        Args:
            encoder: :class:`TCCMPopularityEncoder` instance.
            bucket_input: ``(C, T+E)`` int — per-token CTR bucket indices.
            time_input: ``(C,)`` int — clamped age in hours since publish.
            title_len: Number of title-token positions ``T``.

        Returns:
            ``(C,)`` numpy array of popularity scores.
        """
        b = jnp.asarray(bucket_input, dtype=jnp.int32)
        t = jnp.asarray(time_input, dtype=jnp.int32)
        scores = encoder(b, t, title_len=title_len, training=False)
        return np.asarray(scores)

    # ------------------------------------------------------------------
    # CAUM-specific methods
    # ------------------------------------------------------------------

    def score_caum_impression(
        self,
        inter_model: Any,
        cand_vectors: Any,
        clicked_vecs: Any,
        _chunk_size: int = 64,
    ) -> np.ndarray:
        """Score all candidates in fixed-size chunks.

        Uses a fixed chunk size so JAX only compiles the inter_model
        once (for the padded chunk shape) instead of recompiling for
        every unique impression size.

        Args:
            inter_model: The candidate-aware interaction model.
            cand_vectors: (C, D) numpy array of candidate news vectors.
            clicked_vecs: (H, D) numpy array of clicked news vectors.
            _chunk_size: Fixed batch size for each forward pass.

        Returns:
            (C,) numpy array of scores, one per candidate.
        """
        C = cand_vectors.shape[0]
        D = cand_vectors.shape[1]
        H = clicked_vecs.shape[0]

        all_scores = []
        for start in range(0, C, _chunk_size):
            end = min(start + _chunk_size, C)
            actual = end - start

            # Pad chunk to fixed size so JAX sees a stable shape
            cand_chunk = np.zeros((_chunk_size, D), dtype=np.float32)
            cand_chunk[:actual] = cand_vectors[start:end]

            clicked_batch = np.broadcast_to(
                clicked_vecs[None], (_chunk_size, H, D)
            ).copy()

            cand_j = jnp.asarray(cand_chunk, dtype=jnp.float32)
            clicked_j = jnp.asarray(clicked_batch, dtype=jnp.float32)

            chunk_scores = np.asarray(inter_model(cand_j, clicked_j, training=False))
            all_scores.append(chunk_scores[:actual])

        return np.concatenate(all_scores, axis=0)

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

    # ------------------------------------------------------------------
    # GLORY-specific methods
    # ------------------------------------------------------------------

    def encode_glory_news(
        self,
        news_encoder: Any,
        news_features: Any,
        batch_size: int,
    ) -> np.ndarray:
        """Run GLORY local news encoder over the full corpus."""
        num_news = news_features.shape[0]
        news_dim = news_encoder.news_dim
        out = np.zeros((num_news, news_dim), dtype=np.float32)
        for start in range(0, num_news, batch_size):
            end = min(start + batch_size, num_news)
            batch = jnp.asarray(news_features[start:end], dtype=jnp.int32)
            emb = news_encoder(
                batch[None, :, :],
                training=False,
            ).squeeze(axis=0)  # (B, D)
            out[start:end] = np.asarray(emb)
        return out

    def encode_glory_global(
        self,
        graph_encoder: Any,
        all_news_emb: Any,
        edge_index: Any,
    ) -> np.ndarray:
        """Run the global GatedGraphConv on the full news graph."""
        x = jnp.asarray(all_news_emb, dtype=jnp.float32)
        ei = jnp.asarray(edge_index, dtype=jnp.int32)
        out = graph_encoder(x, ei)
        return np.asarray(out)

    def encode_glory_entity(
        self,
        entity_embedding: Any,
        entity_encoder: Any,
        entity_ids: Any,
        batch_size: int = 0,
    ) -> np.ndarray:
        """Embed + encode entity IDs via the local entity encoder.

        Args:
            entity_embedding: nnx.Embed module.
            entity_encoder: EntityEncoder module.
            entity_ids: (N, E) int array of entity IDs.
            batch_size: If > 0, process in batches (for pre-computation).

        Returns:
            (N, D) entity representations.
        """
        N = entity_ids.shape[0]
        if batch_size > 0 and batch_size < N:
            chunks = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                ids = jnp.asarray(entity_ids[start:end], dtype=jnp.int32)
                embedded = entity_embedding(ids)
                out = entity_encoder(embedded[None], training=False).squeeze(axis=0)
                chunks.append(np.asarray(out))
            return np.concatenate(chunks, axis=0)
        ids = jnp.asarray(entity_ids, dtype=jnp.int32)
        embedded = entity_embedding(ids)
        out = entity_encoder(embedded[None], training=False).squeeze(axis=0)
        return np.asarray(out)

    def encode_glory_global_entity(
        self,
        entity_embedding: Any,
        global_entity_encoder: Any,
        neighbor_ids: Any,
        entity_mask: Any,
        batch_size: int = 0,
    ) -> np.ndarray:
        """Embed + encode neighbor entity IDs via global entity encoder.

        Args:
            entity_embedding: nnx.Embed module.
            global_entity_encoder: GlobalEntityEncoder module.
            neighbor_ids: (N, E*EN) int array of neighbor entity IDs.
            entity_mask: (N, E*EN) float mask.
            batch_size: If > 0, process in batches (for pre-computation).

        Returns:
            (N, D) entity representations.
        """
        N = neighbor_ids.shape[0]
        if batch_size > 0 and batch_size < N:
            chunks = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                ids = jnp.asarray(neighbor_ids[start:end], dtype=jnp.int32)
                embedded = entity_embedding(ids)
                mask = (
                    jnp.asarray(entity_mask[start:end], dtype=jnp.float32)
                    if entity_mask is not None
                    else None
                )
                out = global_entity_encoder(
                    embedded[None], mask, training=False
                ).squeeze(axis=0)
                chunks.append(np.asarray(out))
            return np.concatenate(chunks, axis=0)
        ids = jnp.asarray(neighbor_ids, dtype=jnp.int32)
        embedded = entity_embedding(ids)
        mask = (
            jnp.asarray(entity_mask, dtype=jnp.float32)
            if entity_mask is not None
            else None
        )
        out = global_entity_encoder(embedded[None], mask, training=False).squeeze(
            axis=0
        )
        return np.asarray(out)

    def score_glory_impression(
        self,
        click_encoder: Any,
        user_encoder: Any,
        candidate_encoder: Any,
        click_predictor: Any,
        clicked_title: Any,
        clicked_graph: Any,
        cand_local: Any,
        clicked_entity_emb: Any = None,
        cand_origin_emb: Any = None,
        cand_neighbor_emb: Any = None,
    ) -> np.ndarray:
        """Fuse + score a single impression's candidates."""
        ct = jnp.asarray(clicked_title, dtype=jnp.float32)[None]  # (1, H, D)
        cg = jnp.asarray(clicked_graph, dtype=jnp.float32)[None]
        cl = jnp.asarray(cand_local, dtype=jnp.float32)[None]  # (1, C, D)

        ce = None
        if clicked_entity_emb is not None:
            ce = jnp.asarray(clicked_entity_emb, dtype=jnp.float32)[None]

        fused = click_encoder(ct, cg, ce)  # (1, H, D)
        user_emb = user_encoder(fused)  # (1, D)

        co = None
        cn = None
        if cand_origin_emb is not None:
            co = jnp.asarray(cand_origin_emb, dtype=jnp.float32)[None]
        if cand_neighbor_emb is not None:
            cn = jnp.asarray(cand_neighbor_emb, dtype=jnp.float32)[None]

        cand_final = candidate_encoder(cl, co, cn)  # (1, C, D)
        # Dot-product scoring
        scores = jnp.sum(cand_final * user_emb[:, None, :], axis=-1)  # (1, C)
        return np.asarray(scores.squeeze(axis=0))
