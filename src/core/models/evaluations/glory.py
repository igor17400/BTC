"""GLORY evaluation pipeline — framework-agnostic.

GLORY's training uses per-sample k-hop subgraphs, but at eval time the
whole corpus has been clicked at least once by training users so the
global news graph already captures the connectivity we need.  The
natural eval flow is:

1. Pre-encode every news article once via the local news encoder.
2. Run the global GNN over the full news graph once, giving every
   article a graph-aware embedding.
3. For each impression, gather the user's clicked + candidate
   embeddings from those two caches, fuse them, score by dot product.

This is strictly simpler and faster than the reference
``ValidGraphDataset`` path (per-impression subgraph + per-impression
GNN) and produces equivalent-or-better ranking scores because messages
are not truncated to the user's local neighborhood.  If we need the
exact reference behavior later, we can swap step 2 for per-impression
subgraph sampling without changing the rest of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.core.io.progress import create_progress
from src.core.models.evaluations.utils import compute_metrics

logger = logging.getLogger(__name__)


def glory_evaluate(
    *,
    news_encoder: Any,
    graph_encoder: Any,
    click_encoder: Any,
    user_encoder: Any,
    candidate_encoder: Any,
    click_predictor: Any,
    dataset_provider: Any,
    processed_news: dict,
    news_features: np.ndarray,  # (num_news, feat_dim) packed features
    graph: dict,  # {edge_index, edge_attr, num_nodes}
    neighbor_dict: dict[int, list[int]],
    adapter: Any,
    metrics_calculator: Any,
    his_size: int,
    mode: str = "val",
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate GLORY on the dev or test set.

    Returns:
        ``{metric_name: value}`` dictionary.
    """
    # Step 1: pre-encode every news article.
    logger.info("Pre-encoding all news...")
    num_news = news_features.shape[0]
    all_news_emb = adapter.encode_glory_news(
        news_encoder,
        news_features,
        batch_size=batch_size,
    )  # (num_news, D)

    # Step 2: run global GNN once on the full news graph.
    logger.info("Running global GNN on full news graph...")
    all_news_graph_emb = adapter.encode_glory_global(
        graph_encoder,
        all_news_emb,
        graph["edge_index"],
    )  # (num_news, D)

    # Step 3: score each impression.
    behaviors = (
        dataset_provider.val_behaviors_data
        if mode == "val"
        else dataset_provider.test_behaviors_data
    )
    if "histories_news_ids" not in behaviors or "candidate_news_ids" not in behaviors:
        logger.warning(
            "GLORY eval: required behavior fields missing; returning dummy metrics"
        )
        return {
            "auc": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "num_impressions": 0,
        }

    hist_ids_all = np.asarray(behaviors["histories_news_ids"]).astype(np.int64)
    impression_ids = np.asarray(behaviors["impression_ids"])
    labels_all = behaviors["labels"]

    num_impressions = len(impression_ids)
    logger.info(f"Scoring {num_impressions} impressions...")

    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []

    scoring_progress = create_progress(transient=True)
    scoring_progress.start()
    scoring_task = scoring_progress.add_task(
        "Scoring GLORY impressions...", total=num_impressions
    )

    for idx in range(num_impressions):
        hist = hist_ids_all[idx]
        hist_valid_mask = hist > 0
        n_valid = int(hist_valid_mask.sum())
        if n_valid == 0:
            continue
        hist_valid = hist[hist_valid_mask][-his_size:]

        cand_ids_raw = np.asarray(behaviors["candidate_news_ids"][idx]).astype(np.int64)
        if cand_ids_raw.ndim == 0:
            cand_ids_raw = cand_ids_raw.reshape(1)
        cand_ids = np.clip(cand_ids_raw, 0, num_news - 1)

        labels = np.asarray(labels_all[idx])
        if labels.size == 0:
            continue

        # Gather embeddings.
        clicked_title = all_news_emb[hist_valid]  # (H_valid, D)
        clicked_graph = all_news_graph_emb[hist_valid]  # (H_valid, D)
        cand_local = all_news_emb[cand_ids]  # (C, D)

        scores = adapter.score_glory_impression(
            click_encoder,
            user_encoder,
            candidate_encoder,
            click_predictor,
            clicked_title,
            clicked_graph,
            cand_local,
        )  # (C,)
        group_labels.append(labels)
        group_preds.append(scores)
        scoring_progress.advance(scoring_task)

    scoring_progress.remove_task(scoring_task)
    scoring_progress.stop()

    with create_progress(transient=True) as progress:
        metrics = compute_metrics(
            group_labels,
            group_preds,
            metrics_calculator,
            progress,
        )
    metrics["num_impressions"] = len(group_labels)
    return metrics
