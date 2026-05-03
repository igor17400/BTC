"""PyTorch implementation of the framework adapter Protocol.

The shared evaluator at :mod:`src.core.models.evaluation` invokes PyTorch
encoder modules through this adapter and converts PyTorch tensors to
numpy. This is the only place in the PyTorch path that needs to know
about the shared evaluation pipeline; everything downstream is pure
numpy.

Device placement, ``model.eval()``, and ``torch.no_grad()`` are all
handled inside the adapter so the shared evaluator stays
framework-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class PyTorchAdapter:
    """Framework adapter for PyTorch news recommendation models."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert any PyTorch tensor (or numpy array) to numpy."""
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run the news encoder on a feature batch and return numpy vectors.

        Handles device placement, eval-mode, and no-grad internally.
        """
        encoder.eval()
        device = next(encoder.parameters()).device
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features)
        features = features.to(device)
        with torch.no_grad():
            vec = encoder(features, training=False)
        return vec.detach().cpu().numpy()

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run the user encoder on a history batch and return numpy vectors.

        For LSTUR-style encoders (``process_user_id=True``) the encoder is
        called as ``encoder([features, user_ids], training=False)``. For all
        other models it's ``encoder(features, training=False)``.
        """
        encoder.eval()
        device = next(encoder.parameters()).device
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features)
        features = features.to(device)
        with torch.no_grad():
            if process_user_id:
                if isinstance(user_ids, np.ndarray):
                    user_ids = torch.as_tensor(user_ids)
                user_ids = user_ids.to(device)
                vec = encoder([features, user_ids], training=False)
            else:
                vec = encoder(features, training=False)
        return vec.detach().cpu().numpy()

    def encode_user_interests(
        self, user_encoder: Any, features: Any
    ) -> np.ndarray:
        """Run the MINER user encoder to get K interest vectors.

        Returns:
            ``(B, K, E)`` numpy array of interest vectors.
        """
        user_encoder.eval()
        device = next(user_encoder.parameters()).device
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features)
        features = features.to(device)
        with torch.no_grad():
            vecs = user_encoder.encode_interests(features, training=False)
        return vecs.detach().cpu().numpy()

    def run_activity_gater(self, gater: Any, user_vecs: Any) -> np.ndarray:
        """Run the ActivityGater on a batch of user vectors."""
        gater.eval()
        device = next(gater.parameters()).device
        user_t = torch.as_tensor(user_vecs, dtype=torch.float32).to(device)
        with torch.no_grad():
            return gater(user_t).detach().cpu().numpy()

    def run_popularity_predictor(
        self,
        predictor: Any,
        bias_vecs: Any,
        recency: Any | None,
        ctr: Any | None,
    ) -> np.ndarray:
        """Run the PopularityPredictor with full inputs."""
        predictor.eval()
        device = next(predictor.parameters()).device
        bias_t = torch.as_tensor(bias_vecs, dtype=torch.float32).to(device)
        recency_t = (
            torch.as_tensor(recency).long().to(device) if recency is not None else None
        )
        ctr_t = (
            torch.as_tensor(ctr, dtype=torch.float32).to(device)
            if ctr is not None
            else None
        )
        with torch.no_grad():
            return (
                predictor(bias_t, recency_indices=recency_t, ctr_values=ctr_t)
                .detach()
                .cpu()
                .numpy()
            )

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
        encoder.eval()
        device = next(encoder.parameters()).device
        b = torch.as_tensor(bucket_input, dtype=torch.long).to(device)
        t = torch.as_tensor(time_input, dtype=torch.long).to(device)
        with torch.no_grad():
            scores = encoder(b, t, title_len=title_len)
        return scores.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # DIGAT-specific methods
    # ------------------------------------------------------------------

    def encode_digat_news(
        self,
        news_encoder: Any,
        tokens: Any,
        mask: Any,
    ) -> Any:
        """Run the DIGAT MSA news encoder on a raw token batch."""
        news_encoder.eval()
        device = next(news_encoder.parameters()).device
        tokens_t = torch.as_tensor(tokens, dtype=torch.long).to(device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32).to(device)
        with torch.no_grad():
            emb = news_encoder(tokens_t.unsqueeze(1), mask_t.unsqueeze(1))
        return emb.squeeze(1).detach().cpu().numpy()

    def encode_digat_graph_context(
        self,
        graph_encoder: Any,
        sag_emb: Any,
        sag_mask: Any,
    ) -> Any:
        """Compute SAG graph context vectors for a batch of news."""
        graph_encoder.eval()
        device = next(graph_encoder.parameters()).device
        sag_emb_t = torch.as_tensor(sag_emb, dtype=torch.float32).to(device)
        sag_mask_t = torch.as_tensor(sag_mask, dtype=torch.float32).to(device)
        with torch.no_grad():
            ctx = graph_encoder._news_graph_context(sag_emb_t, sag_mask_t)
        return ctx.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # GLORY-specific methods
    # ------------------------------------------------------------------

    def encode_glory_news(
        self,
        news_encoder: Any,
        news_features: Any,
        batch_size: int,
    ) -> Any:
        """Run GLORY local news encoder over the full corpus."""
        news_encoder.eval()
        device = next(news_encoder.parameters()).device
        num_news = news_features.shape[0]
        out = np.zeros((num_news, news_encoder.news_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, num_news, batch_size):
                end = min(start + batch_size, num_news)
                batch = torch.as_tensor(
                    news_features[start:end], dtype=torch.long,
                ).to(device).unsqueeze(0)             # (1, B, feat)
                emb = news_encoder(batch).squeeze(0)   # (B, D)
                out[start:end] = emb.detach().cpu().numpy()
        return out

    def encode_glory_global(
        self,
        graph_encoder: Any,
        all_news_emb: Any,
        edge_index: Any,
    ) -> Any:
        """Run the global GatedGraphConv on the full news graph."""
        graph_encoder.eval()
        device = next(graph_encoder.parameters()).device
        x = torch.as_tensor(all_news_emb, dtype=torch.float32).to(device)
        ei = torch.as_tensor(edge_index, dtype=torch.long).to(device)
        with torch.no_grad():
            out = graph_encoder(x, ei)
        return out.detach().cpu().numpy()

    def encode_glory_entity(
        self,
        entity_embedding: Any,
        entity_encoder: Any,
        entity_ids: Any,
        batch_size: int = 0,
    ) -> np.ndarray:
        """Embed + encode entity IDs via the local entity encoder.

        Args:
            entity_embedding: ``nn.Embedding`` module.
            entity_encoder: :class:`EntityEncoder` module.
            entity_ids: ``(N, E)`` int array of entity IDs.
            batch_size: If > 0, process in chunks (for pre-computation).

        Returns:
            ``(N, D)`` entity representations.
        """
        entity_embedding.eval()
        entity_encoder.eval()
        device = next(entity_encoder.parameters()).device
        N = entity_ids.shape[0]

        def _encode(ids_np: np.ndarray) -> np.ndarray:
            ids = torch.as_tensor(ids_np, dtype=torch.long).to(device)
            with torch.no_grad():
                embedded = entity_embedding(ids)        # (n, E, dim)
                # add a leading batch axis (treated as B=1, N=n)
                out = entity_encoder(embedded.unsqueeze(0)).squeeze(0)  # (n, D)
            return out.cpu().numpy()

        if batch_size > 0 and N > batch_size:
            chunks: list[np.ndarray] = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                chunks.append(_encode(np.asarray(entity_ids[start:end])))
            return np.concatenate(chunks, axis=0)
        return _encode(np.asarray(entity_ids))

    def encode_glory_global_entity(
        self,
        entity_embedding: Any,
        global_entity_encoder: Any,
        neighbor_ids: Any,
        entity_mask: Any,
        batch_size: int = 0,
    ) -> np.ndarray:
        """Embed + encode neighbor entity IDs via the global entity encoder.

        Args:
            entity_embedding: ``nn.Embedding`` module.
            global_entity_encoder: :class:`GlobalEntityEncoder` module.
            neighbor_ids: ``(N, E*EN)`` int array of neighbor entity IDs.
            entity_mask: ``(N, E*EN)`` float mask (or ``None``).
            batch_size: If > 0, process in chunks (for pre-computation).

        Returns:
            ``(N, D)`` entity representations.
        """
        entity_embedding.eval()
        global_entity_encoder.eval()
        device = next(global_entity_encoder.parameters()).device
        N = neighbor_ids.shape[0]

        def _encode(ids_np: np.ndarray, mask_np: np.ndarray | None) -> np.ndarray:
            ids = torch.as_tensor(ids_np, dtype=torch.long).to(device)
            mask = (
                torch.as_tensor(mask_np, dtype=torch.float32).to(device)
                if mask_np is not None
                else None
            )
            with torch.no_grad():
                embedded = entity_embedding(ids)  # (n, E*EN, dim)
                out = global_entity_encoder(embedded.unsqueeze(0), mask).squeeze(0)
            return out.cpu().numpy()

        if batch_size > 0 and N > batch_size:
            chunks: list[np.ndarray] = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                m = (
                    np.asarray(entity_mask[start:end])
                    if entity_mask is not None
                    else None
                )
                chunks.append(_encode(np.asarray(neighbor_ids[start:end]), m))
            return np.concatenate(chunks, axis=0)
        m = np.asarray(entity_mask) if entity_mask is not None else None
        return _encode(np.asarray(neighbor_ids), m)

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
    ) -> Any:
        """Fuse + score a single impression's candidates."""
        click_encoder.eval()
        user_encoder.eval()
        candidate_encoder.eval()
        device = next(click_encoder.parameters()).device

        # Add batch dim for the encoders.
        ct = torch.as_tensor(clicked_title, dtype=torch.float32).to(device).unsqueeze(0)   # (1, H, D)
        cg = torch.as_tensor(clicked_graph, dtype=torch.float32).to(device).unsqueeze(0)
        cl = torch.as_tensor(cand_local, dtype=torch.float32).to(device).unsqueeze(0)      # (1, C, D)

        ce = None
        if clicked_entity_emb is not None:
            ce = torch.as_tensor(
                clicked_entity_emb, dtype=torch.float32,
            ).to(device).unsqueeze(0)
        co = None
        cn = None
        if cand_origin_emb is not None:
            co = torch.as_tensor(
                cand_origin_emb, dtype=torch.float32,
            ).to(device).unsqueeze(0)
        if cand_neighbor_emb is not None:
            cn = torch.as_tensor(
                cand_neighbor_emb, dtype=torch.float32,
            ).to(device).unsqueeze(0)

        with torch.no_grad():
            fused = click_encoder(ct, cg, ce)               # (1, H, D)
            user_emb = user_encoder(fused)                  # (1, D)
            cand_final = candidate_encoder(cl, co, cn)      # (1, C, D)
            scores = click_predictor(cand_final, user_emb)  # (1, C)
        return scores.squeeze(0).detach().cpu().numpy()

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
    ) -> Any:
        """Run dual graph interaction and return per-candidate scores."""
        graph_encoder.eval()
        device = next(graph_encoder.parameters()).device
        C = cand_sag_emb.shape[0]

        cand_emb_t = torch.as_tensor(cand_sag_emb, dtype=torch.float32).to(device)
        cand_graph_t = torch.as_tensor(cand_sag_graph, dtype=torch.float32).to(device)
        cand_mask_t = torch.as_tensor(cand_sag_mask, dtype=torch.float32).to(device)

        # Broadcast per-impression user data across the C candidates
        hist_t = (
            torch.as_tensor(user_hist_emb, dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
            .expand(C, -1, -1)
        )
        u_graph_t = (
            torch.as_tensor(u_graph, dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
            .expand(C, -1, -1)
        )
        u_cat_mask_t = (
            torch.as_tensor(u_cat_mask, dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
            .expand(C, -1)
        )
        u_cat_idx_t = (
            torch.as_tensor(u_cat_indices, dtype=torch.long)
            .to(device)
            .unsqueeze(0)
            .expand(C, -1)
        )

        with torch.no_grad():
            news_ctx, user_ctx = graph_encoder(
                cand_emb_t, cand_graph_t, cand_mask_t,
                hist_t, u_graph_t, u_cat_mask_t, u_cat_idx_t,
                num_categories,
            )
        return (news_ctx * user_ctx).sum(dim=-1).detach().cpu().numpy()
