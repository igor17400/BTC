"""Framework adapter Protocol for the shared evaluation pipeline.

The shared :mod:`src.core.models.evaluation` module needs to invoke
framework-specific encoder modules and convert framework-native tensors
to numpy. To stay framework-agnostic it never imports torch /
jax directly. Instead, each framework provides a small adapter that
implements the :class:`FrameworkAdapter` protocol below, and the shared
evaluator only depends on this protocol.

This is the **only** seam between the shared evaluation algorithm and
the per-framework implementations. Adding a new framework means writing
a new ~30-line adapter and nothing more.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class FrameworkAdapter(Protocol):
    """Glue between the shared evaluator and a specific ML framework.

    Implementations live in ``src.frameworks.{framework}.models.adapter``.
    The adapter only handles operations that fundamentally require
    framework knowledge: invoking framework-native encoder modules and
    converting framework-native tensors to numpy. Everything else
    (impression iteration, dot-product scoring, NaN handling, loss,
    metric aggregation) is pure numpy in the shared evaluator.
    """

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert any framework-native tensor (or numpy array) to numpy.

        Must be a no-op when ``value`` is already a numpy array.
        """
        ...

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run a news encoder on a feature batch and return numpy vectors.

        Args:
            encoder: Framework-native module that maps news features to
                vectors. The shared evaluator never inspects this object;
                it just passes it through.
            features: Batched news features as produced by the
                framework's ``NewsBatchDataloader``.

        Returns:
            ``(batch_size, news_dim)`` numpy array of news vectors.
        """
        ...

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run a user encoder on a history batch and return numpy vectors.

        Args:
            encoder: Framework-native user encoder module.
            features: Batched history features.
            user_ids: Optional user-id tensor (only used by LSTUR).
            process_user_id: ``True`` if the encoder requires explicit
                user IDs alongside history (LSTUR pattern).

        Returns:
            ``(batch_size, user_dim)`` numpy array of user vectors.
        """
        ...

    # ------------------------------------------------------------------
    # DIGAT-specific methods (optional — only needed by DIGAT adapters)
    # ------------------------------------------------------------------

    def encode_digat_news(
        self,
        news_encoder: Any,
        tokens: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Run the DIGAT news (MSA) encoder on a raw token batch.

        Unlike :meth:`encode_news`, this accepts raw token arrays rather
        than dataloader feature dicts, because DIGAT pre-computes
        embeddings for the entire corpus by iterating index ranges.

        Args:
            news_encoder: Framework-native DIGAT news encoder module.
            tokens: ``(B, T)`` int array of title token IDs.
            mask: ``(B, T)`` float array of token masks (1 = valid).

        Returns:
            ``(B, D)`` numpy array of news embeddings.
        """
        ...

    def encode_digat_graph_context(
        self,
        graph_encoder: Any,
        sag_emb: np.ndarray,
        sag_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute SAG graph context vectors for a batch of news.

        Wraps the graph encoder's internal ``_news_graph_context`` method
        which runs gated attention over a news article's SAG subgraph
        nodes to produce a single context vector per news.

        Args:
            graph_encoder: Framework-native DIGAT graph encoder module.
            sag_emb: ``(B, G_n, D)`` float — SAG node embeddings for
                each news in the batch.
            sag_mask: ``(B, G_n)`` float — valid-node mask per SAG graph.

        Returns:
            ``(B, D)`` numpy array of graph context vectors.
        """
        ...

    # ------------------------------------------------------------------
    # GLORY-specific methods (optional — only needed by GLORY adapters)
    # ------------------------------------------------------------------

    def encode_glory_news(
        self,
        news_encoder: Any,
        news_features: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Run GLORY's local news encoder on the full corpus.

        Args:
            news_encoder: Framework-native ``GLORYNewsEncoder``.
            news_features: ``(num_news, feat_dim)`` int packed features.
            batch_size: Mini-batch size for the loop.

        Returns:
            ``(num_news, D)`` float embeddings.
        """
        ...

    def encode_glory_global(
        self,
        graph_encoder: Any,
        all_news_emb: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Run GLORY's global GatedGraphConv on the full news graph.

        Args:
            graph_encoder: Framework-native ``GatedGraphConv`` module.
            all_news_emb: ``(num_news, D)`` node features.
            edge_index: ``(2, E)`` edges.

        Returns:
            ``(num_news, D)`` graph-enhanced embeddings.
        """
        ...

    def score_glory_impression(
        self,
        click_encoder: Any,
        user_encoder: Any,
        candidate_encoder: Any,
        click_predictor: Any,
        clicked_title: np.ndarray,
        clicked_graph: np.ndarray,
        cand_local: np.ndarray,
    ) -> np.ndarray:
        """Fuse + score a single impression's candidates.

        Args:
            clicked_title: ``(H, D)`` locally-encoded clicked news.
            clicked_graph: ``(H, D)`` globally-enhanced clicked news.
            cand_local: ``(C, D)`` locally-encoded candidates.

        Returns:
            ``(C,)`` relevance scores.
        """
        ...

    def score_digat_impression(
        self,
        graph_encoder: Any,
        cand_sag_emb: np.ndarray,
        cand_sag_graph: np.ndarray,
        cand_sag_mask: np.ndarray,
        user_hist_emb: np.ndarray,
        u_graph: np.ndarray,
        u_cat_mask: np.ndarray,
        u_cat_indices: np.ndarray,
        num_categories: int,
    ) -> np.ndarray:
        """Run dual graph interaction and return per-candidate scores.

        Feeds the candidate SAG subgraphs and the user history graph
        through the DIGAT graph encoder simultaneously.  The two
        co-dependent output context vectors are combined via element-wise
        dot product to produce one score per candidate.

        The adapter is responsible for broadcasting per-impression user
        data to match the candidate batch dimension C before invoking
        the framework-native graph encoder.

        Args:
            graph_encoder: Framework-native DIGAT graph encoder module.
            cand_sag_emb: ``(C, G_n, D)`` — candidate SAG node embeddings.
            cand_sag_graph: ``(C, G_n, G_n)`` — candidate SAG adjacency.
            cand_sag_mask: ``(C, G_n)`` — candidate SAG valid-node mask.
            user_hist_emb: ``(H, D)`` — history news embeddings for this
                impression (un-broadcast; adapter expands to C).
            u_graph: ``(G_u, G_u)`` — user-topic graph adjacency.
            u_cat_mask: ``(num_categories,)`` — active category mask.
            u_cat_indices: ``(H,)`` — category index per history slot.
            num_categories: Total number of topic categories.

        Returns:
            ``(C,)`` numpy array of relevance scores.
        """
        ...
