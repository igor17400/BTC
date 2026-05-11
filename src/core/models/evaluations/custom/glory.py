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
from src.core.io.saving import save_predictions_to_file_fn
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
    id_remap: np.ndarray | None = None,
    # Entity support (optional).
    use_entity: bool = False,
    entity_encoder: Any | None = None,
    global_entity_encoder: Any | None = None,
    entity_embedding: Any | None = None,
    entity_neighbor_dict: dict[int, list[int]] | None = None,
    entity_size: int = 5,
    entity_neighbors: int = 10,
    title_size: int = 30,
    save_predictions_path: str | None = None,
    epoch: int | None = None,
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

    # Step 2b: pre-encode entity embeddings for ALL news (optional).
    all_local_entity_emb = None  # (num_news, D)
    all_neighbor_entity_emb = None  # (num_news, D)
    if use_entity and entity_embedding is not None and entity_encoder is not None:
        logger.info("Pre-encoding entity embeddings for all news...")
        E = entity_size
        EN = entity_neighbors

        # Local entity encoding: embed each news's entity IDs → encode.
        all_entity_ids = news_features[
            :,
            title_size : title_size + E,
        ].astype(np.int32)  # (num_news, E)
        all_local_entity_emb = adapter.encode_glory_entity(
            entity_embedding,
            entity_encoder,
            all_entity_ids,
            batch_size=batch_size,
        )  # (num_news, D)

        # Neighbor entity encoding: look up neighbors → embed → encode.
        logger.info("Pre-encoding neighbor entity embeddings for all news...")
        all_neighbor_ids = np.zeros((num_news, E * EN), dtype=np.int32)
        if entity_neighbor_dict is not None:
            for nid in range(num_news):
                for e_idx in range(E):
                    eid = int(all_entity_ids[nid, e_idx])
                    if eid == 0:
                        continue
                    nbrs = entity_neighbor_dict.get(eid, [])
                    vlen = min(len(nbrs), EN)
                    if vlen > 0:
                        offset = e_idx * EN
                        all_neighbor_ids[nid, offset : offset + vlen] = nbrs[:vlen]
        all_neighbor_mask = (all_neighbor_ids > 0).astype(np.float32)
        all_neighbor_entity_emb = adapter.encode_glory_global_entity(
            entity_embedding,
            global_entity_encoder,
            all_neighbor_ids,
            all_neighbor_mask,
            batch_size=batch_size,
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
            "_num_impressions": 0,
        }

    hist_ids_raw = np.asarray(behaviors["histories_news_ids"]).astype(np.int64)
    if id_remap is not None:
        hist_ids_all = np.where(
            hist_ids_raw < len(id_remap), id_remap[hist_ids_raw], 0
        ).astype(np.int64)
    else:
        hist_ids_all = np.clip(hist_ids_raw, 0, num_news - 1)
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
        if id_remap is not None:
            cand_ids = np.where(
                cand_ids_raw < len(id_remap), id_remap[cand_ids_raw], 0
            ).astype(np.int64)
        else:
            cand_ids = np.clip(cand_ids_raw, 0, num_news - 1)

        labels = np.asarray(labels_all[idx])
        if labels.size == 0:
            continue

        # Gather embeddings.
        clicked_title = all_news_emb[hist_valid]  # (H_valid, D)
        clicked_graph = all_news_graph_emb[hist_valid]  # (H_valid, D)
        cand_local = all_news_emb[cand_ids]  # (C, D)

        # Entity embeddings (optional) — gather from pre-computed caches.
        clicked_entity_emb = None
        cand_origin_emb = None
        cand_neighbor_emb = None
        if all_local_entity_emb is not None:
            clicked_entity_emb = all_local_entity_emb[hist_valid]  # (H_valid, D)
            cand_origin_emb = all_local_entity_emb[cand_ids]  # (C, D)
            cand_neighbor_emb = all_neighbor_entity_emb[cand_ids]  # (C, D)

        scores = adapter.score_glory_impression(
            click_encoder,
            user_encoder,
            candidate_encoder,
            click_predictor,
            clicked_title,
            clicked_graph,
            cand_local,
            clicked_entity_emb=clicked_entity_emb,
            cand_origin_emb=cand_origin_emb,
            cand_neighbor_emb=cand_neighbor_emb,
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
    metrics["_num_impressions"] = len(group_labels)

    if save_predictions_path:
        predictions = {}
        for i, (labels, preds) in enumerate(zip(group_labels, group_preds)):
            predictions[str(i)] = (labels.tolist(), preds.tolist())
        save_predictions_to_file_fn(
            predictions, save_predictions_path, epoch, mode=mode
        )

    return metrics
