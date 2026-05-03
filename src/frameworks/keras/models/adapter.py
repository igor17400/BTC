"""Keras implementation of the framework adapter Protocol.

The shared evaluator at :mod:`src.core.models.evaluation` invokes Keras
encoder modules through this adapter and converts Keras tensors to
numpy via the backend-agnostic ``keras.ops`` API. This is the only
place in the Keras path that needs to know about the shared evaluation
pipeline; everything downstream is pure numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from keras import ops


def _no_grad_context():
    """Return a torch.no_grad() context if the backend is torch, else nullcontext."""
    import keras
    if keras.backend.backend() == "torch":
        import torch
        return torch.no_grad()
    from contextlib import nullcontext
    return nullcontext()


class KerasAdapter:
    """Framework adapter for Keras news recommendation models."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert any Keras tensor (or numpy array) to numpy."""
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "numpy") or hasattr(value, "__array__"):
            return ops.convert_to_numpy(value)
        return np.asarray(value)

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run the news encoder on a feature batch and return numpy vectors."""
        return ops.convert_to_numpy(encoder(features, training=False))

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run the user encoder on a history batch and return numpy vectors.

        For LSTUR-style encoders (``process_user_id=True``) the encoder is
        called as ``encoder([features, user_ids], training=False)``. For
        all other models it's ``encoder(features, training=False)``.
        """
        if process_user_id:
            vec = encoder([features, user_ids], training=False)
        else:
            vec = encoder(features, training=False)
        return ops.convert_to_numpy(vec)

    def encode_user_interests(
        self, user_encoder: Any, features: Any
    ) -> np.ndarray:
        """Run the MINER user encoder to get K interest vectors.

        Returns:
            ``(B, K, E)`` numpy array of interest vectors.
        """
        return ops.convert_to_numpy(
            user_encoder.encode_interests(features, training=False)
        )

    def run_activity_gater(self, gater: Any, user_vecs: Any) -> np.ndarray:
        """Run the ActivityGater on a batch of user vectors."""
        return ops.convert_to_numpy(gater(user_vecs, training=False))

    def run_popularity_predictor(
        self,
        predictor: Any,
        bias_vecs: Any,
        recency: Any | None,
        ctr: Any | None,
    ) -> np.ndarray:
        """Run the PopularityPredictor with full inputs."""
        return ops.convert_to_numpy(
            predictor(
                bias_vecs, recency_indices=recency, ctr_values=ctr, training=False
            )
        )

    # ------------------------------------------------------------------
    # TCCM-specific methods
    # ------------------------------------------------------------------

    def run_tccm_popularity_encoder(
        self,
        encoder: Any,
        bucket_input: np.ndarray,
        time_input: np.ndarray,
        title_len: int,
    ) -> np.ndarray:
        """Run the TCCM popularity encoder on a candidate batch.

        Args:
            encoder: The :class:`TCCMPopularityEncoder` instance.
            bucket_input: ``(C, T+E)`` int32 — per-token CTR bucket indices.
            time_input: ``(C,)`` int32 — clamped age in hours since publish.
            title_len: Number of title-token positions ``T``.

        Returns:
            ``(C,)`` numpy array of popularity scores.
        """
        import keras
        b = keras.ops.convert_to_tensor(bucket_input, dtype="int32")
        t = keras.ops.convert_to_tensor(time_input, dtype="int32")
        with _no_grad_context():
            scores = encoder(b, t, title_len=title_len, training=False)
        return ops.convert_to_numpy(scores)

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
        import keras
        tokens_k = keras.ops.convert_to_tensor(tokens, dtype="int32")
        mask_k = keras.ops.convert_to_tensor(mask, dtype="float32")
        with _no_grad_context():
            emb = news_encoder(
                keras.ops.expand_dims(tokens_k, axis=1),
                keras.ops.expand_dims(mask_k, axis=1),
                training=False,
            )
        return ops.convert_to_numpy(keras.ops.squeeze(emb, axis=1))

    def encode_digat_graph_context(
        self,
        graph_encoder: Any,
        sag_emb: Any,
        sag_mask: Any,
    ) -> np.ndarray:
        """Compute SAG graph context vectors for a batch of news."""
        import keras
        sag_emb_k = keras.ops.convert_to_tensor(sag_emb, dtype="float32")
        sag_mask_k = keras.ops.convert_to_tensor(sag_mask, dtype="float32")
        with _no_grad_context():
            ctx = graph_encoder._news_graph_context(sag_emb_k, sag_mask_k)
        return ops.convert_to_numpy(ctx)

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
        _chunk_size: int = 32,
    ) -> np.ndarray:
        """Run dual graph interaction and return per-candidate scores.

        Large impressions (100+ candidates) are chunked to avoid OOM.
        """
        import keras
        C = int(cand_sag_emb.shape[0])
        H = user_hist_emb.shape[0]
        D = user_hist_emb.shape[1]
        num_cat = u_cat_mask.shape[0]
        G_u = u_graph.shape[0]

        all_scores = []
        with _no_grad_context():
            for start in range(0, C, _chunk_size):
                end = min(start + _chunk_size, C)
                chunk_C = end - start

                hist_b = np.broadcast_to(user_hist_emb[None], (chunk_C, H, D))
                u_graph_b = np.broadcast_to(u_graph[None], (chunk_C, G_u, G_u))
                u_cat_mask_b = np.broadcast_to(u_cat_mask[None], (chunk_C, num_cat))
                u_cat_idx_b = np.broadcast_to(u_cat_indices[None], (chunk_C, H))

                news_ctx, user_ctx = graph_encoder(
                    (
                        keras.ops.convert_to_tensor(cand_sag_emb[start:end], dtype="float32"),
                        keras.ops.convert_to_tensor(cand_sag_graph[start:end], dtype="float32"),
                        keras.ops.convert_to_tensor(cand_sag_mask[start:end], dtype="float32"),
                        keras.ops.convert_to_tensor(hist_b, dtype="float32"),
                        keras.ops.convert_to_tensor(u_graph_b, dtype="float32"),
                        keras.ops.convert_to_tensor(u_cat_mask_b, dtype="float32"),
                        keras.ops.convert_to_tensor(u_cat_idx_b, dtype="int32"),
                    ),
                    training=False,
                )
                chunk_scores = ops.convert_to_numpy(
                    keras.ops.sum(news_ctx * user_ctx, axis=-1)
                )
                all_scores.append(chunk_scores)

        return np.concatenate(all_scores, axis=0)

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
        import keras
        num_news = news_features.shape[0]
        news_dim = news_encoder.news_dim
        out = np.zeros((num_news, news_dim), dtype=np.float32)
        with _no_grad_context():
            for start in range(0, num_news, batch_size):
                end = min(start + batch_size, num_news)
                batch = keras.ops.convert_to_tensor(
                    news_features[start:end], dtype="int32",
                )
                emb = news_encoder(
                    keras.ops.expand_dims(batch, axis=0), training=False,
                )
                out[start:end] = ops.convert_to_numpy(
                    keras.ops.squeeze(emb, axis=0),
                )
        return out

    def encode_glory_global(
        self,
        graph_encoder: Any,
        all_news_emb: Any,
        edge_index: Any,
    ) -> np.ndarray:
        """Run the global GatedGraphConv on the full news graph."""
        import keras
        x = keras.ops.convert_to_tensor(all_news_emb, dtype="float32")
        ei = keras.ops.convert_to_tensor(edge_index, dtype="int32")
        with _no_grad_context():
            out = graph_encoder(x, ei)
        return ops.convert_to_numpy(out)

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
        """Fuse + score a single impression's candidates.

        The entity-emb kwargs are accepted for API parity with the JAX
        adapter but currently ignored — the Keras GLORY encoders don't
        yet implement the entity branch. Eval falls back to title+graph
        fusion.
        """
        import keras
        ct = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(clicked_title, dtype="float32"), axis=0,
        )
        cg = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(clicked_graph, dtype="float32"), axis=0,
        )
        cl = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(cand_local, dtype="float32"), axis=0,
        )
        with _no_grad_context():
            fused = click_encoder(ct, cg)
            user_emb = user_encoder(fused)
            cand_final = candidate_encoder(cl)
            scores = keras.ops.sum(
                cand_final * keras.ops.expand_dims(user_emb, axis=1), axis=-1,
            )
        return ops.convert_to_numpy(keras.ops.squeeze(scores, axis=0))
