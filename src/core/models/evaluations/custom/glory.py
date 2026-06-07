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
    # Per-impression eval (matches reference ValidGraphDataset). When
    # "per_impression", the global GNN runs on a k-hop subgraph for each
    # impression instead of once on the full ~65k-node graph — matches
    # the training-time GNN graph size and avoids over-smoothing.
    eval_mode: str = "precomputed",
    k_hops: int = 2,
    num_neighbors: int = 8,
    csr_in_adjacency: tuple | None = None,
    # Padded-subgraph mode (for JAX). When both are set, every
    # per-impression subgraph is padded/truncated to a fixed
    # ``(max_subgraph_nodes, max_subgraph_edges)`` shape so XLA compiles
    # the GNN forward once and reuses it across all impressions. The
    # last subgraph position is reserved as a padding sentinel; padded
    # edges are self-loops there, leaving real-node embeddings intact.
    max_subgraph_nodes: int | None = None,
    max_subgraph_edges: int | None = None,
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

    # Step 2: run global GNN. In "precomputed" mode we do one pass over
    # the full graph and look up history embeddings in the loop. In
    # "per_impression" mode we defer this to a tiny per-impression GNN
    # call on the user's k-hop subgraph (matches reference + training).
    all_news_graph_emb = None
    if eval_mode == "precomputed":
        logger.info("Running global GNN on full news graph...")
        all_news_graph_emb = adapter.encode_glory_global(
            graph_encoder,
            all_news_emb,
            graph["edge_index"],
        )  # (num_news, D)
    elif eval_mode == "per_impression":
        logger.info(
            f"eval_mode=per_impression: GNN will run on k-hop subgraphs "
            f"(k_hops={k_hops}, num_neighbors={num_neighbors})"
        )
    else:
        raise ValueError(
            f"Unknown eval_mode={eval_mode!r}; "
            "expected 'precomputed' or 'per_impression'."
        )

    # Step 2b: pre-encode entity embeddings for ALL news (optional).
    # In "per_impression" mode we skip this — entity encoders run on the
    # impression's own clicked + candidate entity IDs at scoring time
    # (matches reference GLORY's ``validation_process`` which calls
    # ``self.local_entity_encoder(clicked_entity.unsqueeze(0), None)`` and
    # the candidate origin/neighbor encoders per impression).
    all_local_entity_emb = None  # (num_news, D)
    all_neighbor_entity_emb = None  # (num_news, D)
    all_entity_ids: np.ndarray | None = None
    E = entity_size
    EN = entity_neighbors
    if use_entity and entity_embedding is not None and entity_encoder is not None:
        # Always grab the (num_news, E) entity_indices table — both eval
        # modes need it (precomputed batches it; per_impression indexes
        # into it per impression).
        all_entity_ids = news_features[
            :,
            title_size : title_size + E,
        ].astype(np.int32)  # (num_news, E)

        if eval_mode == "precomputed":
            logger.info("Pre-encoding entity embeddings for all news...")
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
        else:
            logger.info(
                "eval_mode=per_impression: entity encoders will run on each "
                "impression's clicked + candidate entity IDs."
            )

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
            # Empty history — score with a single padding sentinel so the
            # impression isn't dropped from the eval set. Matches the
            # reference GLORY ValidGraphDataset behavior. The score will
            # be near-noise (~0.5 AUC contribution) but the eval-set size
            # stays directly comparable to the reference's 73154 dev
            # impressions. See [[project_glory_training_pipeline_gaps]].
            hist_valid = np.array([0], dtype=np.int64)
        else:
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
        if eval_mode == "precomputed":
            clicked_graph = all_news_graph_emb[hist_valid]  # (H_valid, D)
        else:
            # Per-impression: build the user's k-hop subgraph, run the
            # GNN on just those nodes, gather output at the history
            # positions. Mirrors reference ValidGraphDataset.line_mapper.
            from src.core.data.processing.models.glory import (
                extract_edges_for_subgraph,
                sample_subgraph,
            )

            sub_node_ids, sub_hist_mapping = sample_subgraph(
                hist_valid,
                neighbor_dict,
                k_hops,
                num_neighbors,
                max_nodes=max_subgraph_nodes,
            )
            # When padding, the first ``n_real`` positions are real
            # nodes; the rest are sentinels that must be excluded from
            # edge extraction. Padded edges target the last position.
            if max_subgraph_nodes is not None:
                n_real = int((sub_node_ids != 0).sum()) if len(sub_node_ids) > 0 else 0
                # ``hist_valid`` slots are guaranteed real (we always
                # place them first); cap n_real at where the first 0
                # sentinel appears after the history block.
                # The simplest tight bound is: positions occupied before
                # padding (= len(sub_node_ids) - num_trailing_zeros).
                trailing_zeros = 0
                for v in reversed(sub_node_ids):
                    if int(v) == 0:
                        trailing_zeros += 1
                    else:
                        break
                n_real = max_subgraph_nodes - trailing_zeros
                pad_node_idx = max_subgraph_nodes - 1
            else:
                n_real = None
                pad_node_idx = None
            sub_edge_index, _ = extract_edges_for_subgraph(
                sub_node_ids,
                np.asarray(graph["edge_index"]),
                np.asarray(graph["edge_attr"]),
                csr=csr_in_adjacency,
                num_nodes=num_news,
                max_edges=max_subgraph_edges,
                n_real_nodes=n_real,
                pad_node_idx=pad_node_idx,
            )
            sub_x = all_news_emb[sub_node_ids]  # (N_sub, D)
            sub_graph_emb = adapter.encode_glory_global(
                graph_encoder,
                sub_x,
                sub_edge_index,
            )  # (N_sub, D)
            clicked_graph = sub_graph_emb[sub_hist_mapping]  # (H_valid, D)
        cand_local = all_news_emb[cand_ids]  # (C, D)

        # Entity embeddings (optional).
        clicked_entity_emb = None
        cand_origin_emb = None
        cand_neighbor_emb = None
        if use_entity and entity_embedding is not None and all_entity_ids is not None:
            if eval_mode == "precomputed":
                # Gather from pre-computed caches.
                clicked_entity_emb = all_local_entity_emb[hist_valid]  # (H_valid, D)
                cand_origin_emb = all_local_entity_emb[cand_ids]  # (C, D)
                cand_neighbor_emb = all_neighbor_entity_emb[cand_ids]  # (C, D)
            else:
                # Per-impression entity encoding — matches reference
                # validation_process: encoders run on this impression's
                # own clicked entity IDs and candidate entity IDs.
                clicked_entity_ids = all_entity_ids[hist_valid]  # (H_valid, E)
                clicked_entity_emb = adapter.encode_glory_entity(
                    entity_embedding,
                    entity_encoder,
                    clicked_entity_ids,
                    batch_size=0,
                )  # (H_valid, D)

                cand_entity_ids = all_entity_ids[cand_ids]  # (C, E)
                cand_origin_emb = adapter.encode_glory_entity(
                    entity_embedding,
                    entity_encoder,
                    cand_entity_ids,
                    batch_size=0,
                )  # (C, D)

                n_cands = len(cand_ids)
                cand_neighbor_ids = np.zeros((n_cands, E * EN), dtype=np.int32)
                if entity_neighbor_dict is not None:
                    for c_idx in range(n_cands):
                        for e_idx in range(E):
                            eid = int(cand_entity_ids[c_idx, e_idx])
                            if eid == 0:
                                continue
                            nbrs = entity_neighbor_dict.get(eid, [])
                            vlen = min(len(nbrs), EN)
                            if vlen > 0:
                                offset = e_idx * EN
                                cand_neighbor_ids[c_idx, offset : offset + vlen] = nbrs[
                                    :vlen
                                ]
                cand_neighbor_mask = (cand_neighbor_ids > 0).astype(np.float32)
                cand_neighbor_emb = adapter.encode_glory_global_entity(
                    entity_embedding,
                    global_entity_encoder,
                    cand_neighbor_ids,
                    cand_neighbor_mask,
                    batch_size=0,
                )  # (C, D)

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
