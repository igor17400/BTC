"""DIGAT evaluation pipeline — framework-agnostic.

DIGAT's dual graph interaction makes news and user representations
co-dependent: they cannot be pre-computed independently like in
NRMS/NAML/CROWN.  This module pre-computes what CAN be done once
(news MSA embeddings, SAG graph contexts) and delegates the per-impression
dual graph pass to the framework adapter.

This file is the canonical specification of DIGAT evaluation.  Every
framework implementation (PyTorch, JAX, Keras) must produce equivalent
results by implementing the three adapter methods it calls:
:meth:`~src.core.models.adapter.FrameworkAdapter.encode_digat_news`,
:meth:`~src.core.models.adapter.FrameworkAdapter.encode_digat_graph_context`,
and
:meth:`~src.core.models.adapter.FrameworkAdapter.score_digat_impression`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.core.data.processing.models.digat import build_user_graphs
from src.core.io.progress import create_progress
from src.core.io.saving import save_predictions_to_file_fn
from src.core.models.evaluations.utils import compute_metrics

logger = logging.getLogger(__name__)


def digat_evaluate(
    *,
    news_encoder: Any,
    graph_encoder: Any,
    dataset_provider: Any,
    processed_news: dict,
    sag_data: dict,
    adapter: Any,
    metrics_calculator: Any,
    D: int,
    num_categories: int,
    max_history: int,
    mode: str = "val",
    batch_size: int = 64,
    id_remap: np.ndarray | None = None,
    encoder_type: str = "glove",
    save_predictions_path: str | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    """Evaluate DIGAT on the dev or test set.

    Algorithm:

    1. Pre-compute news MSA embeddings for the entire corpus.
    2. Pre-compute SAG graph context vectors per news article.
    3. For each impression: assemble the user-topic graph, run the dual
       graph interaction over all candidates, compute dot-product scores.
    4. Aggregate metrics.

    Args:
        news_encoder: Framework-native DIGAT news (MSA) encoder.
        graph_encoder: Framework-native DIGAT graph encoder.
        dataset_provider: Dataset instance with val/test behaviors.
        processed_news: Dict with ``"tokens"`` (num_news, T) and
            ``"category_indices"`` (num_news,).
        sag_data: SAG cache with ``"news_node_ID"`` (num_news, G_n),
            ``"news_graph"`` (num_news, G_n, G_n), and
            ``"news_graph_mask"`` (num_news, G_n).
        adapter: Framework adapter implementing the three DIGAT methods.
        metrics_calculator: Metrics engine with ``compute_metrics``.
        D: News embedding dimension.
        num_categories: Total number of topic categories (including pad).
        max_history: Maximum user history length H.
        mode: ``"val"`` or ``"test"``.
        batch_size: Batch size for corpus-level pre-computation.
        id_remap: Optional ``remap[behaviors_id] → sag_id`` array.

    Returns:
        ``{metric_name: value}`` dictionary.
    """
    news_tokens = processed_news["tokens"]  # (num_news, T)
    news_node_ID = sag_data["news_node_ID"]  # (num_news, G_n)
    news_graph_adj = sag_data["news_graph"]  # (num_news, G_n, G_n)
    news_graph_mask_arr = sag_data["news_graph_mask"]  # (num_news, G_n)

    num_news = news_tokens.shape[0]

    # Under PLM the encoder looks up token features by parsed news_idx,
    # so build a row_id -> parsed_id table once.
    parsed_news_ids: np.ndarray | None = None
    if encoder_type != "glove":
        from src.core.data.encoders.plm import _parse_news_id

        news_id_strings = list(processed_news["news_ids_original_strings"])
        parsed_news_ids = np.asarray(
            [_parse_news_id(nid, "N") for nid in news_id_strings], dtype=np.int64
        )

    # ------------------------------------------------------------------
    # 1. Pre-compute news MSA embeddings for ALL corpus news
    # ------------------------------------------------------------------
    logger.info("Pre-computing news embeddings...")
    all_news_emb = np.zeros((num_news, D), dtype=np.float32)
    for start in range(0, num_news, batch_size):
        end = min(start + batch_size, num_news)
        if encoder_type == "glove":
            tokens_batch = news_tokens[start:end]  # (B, T)
            mask_batch = (tokens_batch != 0).astype(np.float32)
        else:
            tokens_batch = parsed_news_ids[start:end]  # (B,) parsed ids
            mask_batch = np.zeros_like(tokens_batch, dtype=np.float32)  # ignored
        all_news_emb[start:end] = adapter.encode_digat_news(
            news_encoder, tokens_batch, mask_batch
        )

    # ------------------------------------------------------------------
    # 2. Pre-compute SAG graph context vectors per news
    # ------------------------------------------------------------------
    logger.info("Pre-computing SAG graph contexts...")
    all_news_ctx = np.zeros((num_news, D), dtype=np.float32)
    for start in range(0, num_news, batch_size):
        end = min(start + batch_size, num_news)
        node_ids_batch = news_node_ID[start:end]  # (B, G_n)
        sag_emb = all_news_emb[node_ids_batch]  # (B, G_n, D)
        sag_mask = news_graph_mask_arr[start:end].astype(np.float32)
        all_news_ctx[start:end] = adapter.encode_digat_graph_context(
            graph_encoder, sag_emb, sag_mask
        )

    # ------------------------------------------------------------------
    # 3. Score each impression
    # ------------------------------------------------------------------
    behaviors = (
        dataset_provider.val_behaviors_data
        if mode == "val"
        else dataset_provider.test_behaviors_data
    )

    if "histories_news_ids" not in behaviors:
        logger.warning(
            "DIGAT eval: histories_news_ids not available, returning dummy metrics"
        )
        return {
            "auc": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "_num_impressions": 0,
        }

    hist_ids_raw = np.asarray(behaviors["histories_news_ids"]).astype(np.int64)
    if id_remap is not None:
        hist_ids = np.where(
            hist_ids_raw < len(id_remap), id_remap[hist_ids_raw], 0
        ).astype(np.int64)
    else:
        hist_ids = np.clip(hist_ids_raw, 0, num_news - 1)

    hist_cats = np.asarray(behaviors["history_news_categories"]).astype(np.int64)
    impression_ids = np.asarray(behaviors["impression_ids"])
    labels_all = behaviors["labels"]

    user_graphs = build_user_graphs(hist_cats, max_history, num_categories)

    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []

    num_impressions = len(impression_ids)
    logger.info(f"Scoring {num_impressions} impressions...")

    scoring_progress = create_progress(transient=True)
    scoring_progress.start()
    scoring_task = scoring_progress.add_task(
        "Scoring DIGAT impressions...", total=num_impressions
    )

    for idx in range(num_impressions):
        imp_labels = np.asarray(labels_all[idx])

        if "candidate_news_ids" not in behaviors:
            continue
        cand_ids_raw = np.asarray(behaviors["candidate_news_ids"][idx])
        if not hasattr(cand_ids_raw, "__len__"):
            cand_ids_raw = np.array([int(cand_ids_raw)])
        cand_ids = cand_ids_raw.astype(np.int64)

        if id_remap is not None:
            cand_ids = np.where(cand_ids < len(id_remap), id_remap[cand_ids], 0).astype(
                np.int64
            )
        else:
            cand_ids = np.clip(cand_ids, 0, num_news - 1)

        valid_mask = cand_ids > 0
        if not valid_mask.any():
            continue
        cand_ids = cand_ids[valid_mask]
        if len(imp_labels) == len(valid_mask):
            imp_labels = imp_labels[valid_mask]

        # Per-impression inputs (un-broadcast — adapter handles expansion)
        h_ids = hist_ids[idx]  # (H,)
        user_hist_emb = all_news_emb[h_ids]  # (H, D)
        u_graph = user_graphs["user_graph"][idx].astype(np.float32)  # (G_u, G_u)
        u_cat_mask = user_graphs["user_category_mask"][idx].astype(np.float32)  # (C,)
        u_cat_indices = user_graphs["user_category_indices"][idx]  # (H,)

        # Candidate SAG inputs
        cand_node_ids = news_node_ID[cand_ids]  # (C, G_n)
        cand_sag_emb = all_news_emb[cand_node_ids]  # (C, G_n, D)
        cand_sag_graph = news_graph_adj[cand_ids].astype(np.float32)  # (C, G_n, G_n)
        cand_sag_mask = news_graph_mask_arr[cand_ids].astype(np.float32)  # (C, G_n)

        scores = adapter.score_digat_impression(
            graph_encoder,
            cand_sag_emb,
            cand_sag_graph,
            cand_sag_mask,
            user_hist_emb,
            u_graph,
            u_cat_mask,
            u_cat_indices,
            num_categories,
        )  # (C,)

        group_labels.append(imp_labels)
        group_preds.append(scores)
        scoring_progress.advance(scoring_task)

    scoring_progress.remove_task(scoring_task)
    scoring_progress.stop()

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    with create_progress(transient=True) as progress:
        metrics = compute_metrics(
            group_labels, group_preds, metrics_calculator, progress
        )
    metrics["_num_impressions"] = len(group_labels)

    if save_predictions_path:
        predictions = {}
        for i, (labels, preds) in enumerate(zip(group_labels, group_preds)):
            predictions[str(i)] = (labels.tolist(), preds.tolist())
        save_predictions_to_file_fn(
            predictions, save_predictions_path, epoch, mode=mode
        )

    return metrics
