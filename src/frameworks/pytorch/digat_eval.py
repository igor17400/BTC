"""DIGAT-specific evaluation pipeline.

DIGAT's dual graph interaction makes news and user representations
co-dependent — they can't be pre-computed independently like in
NRMS/NAML/CROWN.  This module pre-computes what CAN be independent
(news embeddings, SAG graph contexts) and runs the full graph encoder
per impression.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from rich.progress import Progress

logger = logging.getLogger(__name__)


@torch.no_grad()
def digat_evaluate(
    model: Any,
    dataset_provider: Any,
    processed_news: dict,
    sag_data: dict,
    metrics_calculator: Any,
    mode: str = "val",
    batch_size: int = 64,
    id_remap: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate DIGAT on the dev or test set.

    Steps:
        1. Pre-compute news embeddings for all corpus news (MSA encoder).
        2. Pre-compute SAG graph contexts (gated attention) per news.
        3. For each impression: build user graph, run dual graph interaction
           per candidate, compute dot-product score.
        4. Aggregate metrics.

    Args:
        model: DIGAT PyTorch model (in eval mode).
        dataset_provider: Dataset instance with val/test behaviors.
        processed_news: Dict with ``tokens``, ``category_indices``, etc.
        sag_data: SAG cache (``news_node_ID``, ``news_graph``, ``news_graph_mask``).
        metrics_calculator: Metrics engine.
        mode: ``"val"`` or ``"test"``.
        batch_size: Batch size for news pre-computation.

    Returns:
        Dict of metric values.
    """
    model.eval()
    device = next(model.parameters()).device

    news_tokens = processed_news["tokens"]  # (num_news, T)
    news_categories = processed_news["category_indices"]  # (num_news,)
    news_node_ID = sag_data["news_node_ID"]  # (num_news, G_n)
    news_graph_adj = sag_data["news_graph"]  # (num_news, G_n, G_n)
    news_graph_mask_arr = sag_data["news_graph_mask"]  # (num_news, G_n)

    num_news = news_tokens.shape[0]
    G_n = model.news_graph_size
    T = news_tokens.shape[1]
    D = model.D
    num_categories = model.num_categories
    max_history = model.max_history

    # ------------------------------------------------------------------
    # 1. Pre-compute news embeddings for ALL corpus news
    # ------------------------------------------------------------------
    logger.info("Pre-computing news embeddings...")
    all_news_emb = np.zeros((num_news, D), dtype=np.float32)
    for start in range(0, num_news, batch_size):
        end = min(start + batch_size, num_news)
        tokens_batch = torch.tensor(news_tokens[start:end], dtype=torch.long, device=device)
        mask_batch = (tokens_batch != 0).float()
        # News encoder expects (B, N, T) — treat each news as a single-item batch
        emb = model.news_encoder(tokens_batch.unsqueeze(1), mask_batch.unsqueeze(1))
        all_news_emb[start:end] = emb.squeeze(1).cpu().numpy()

    # ------------------------------------------------------------------
    # 2. Pre-compute SAG graph contexts per news
    # ------------------------------------------------------------------
    logger.info("Pre-computing SAG graph contexts...")
    all_news_ctx = np.zeros((num_news, D), dtype=np.float32)
    for start in range(0, num_news, batch_size):
        end = min(start + batch_size, num_news)
        # Get SAG node embeddings: for each news, its SAG node IDs → embeddings
        node_ids_batch = news_node_ID[start:end]  # (bs, G_n)
        sag_emb = torch.tensor(
            all_news_emb[node_ids_batch], dtype=torch.float32, device=device
        )  # (bs, G_n, D)
        sag_mask = torch.tensor(
            news_graph_mask_arr[start:end], dtype=torch.float32, device=device
        )
        ctx = model.graph_encoder._news_graph_context(sag_emb, sag_mask)
        all_news_ctx[start:end] = ctx.cpu().numpy()

    # ------------------------------------------------------------------
    # 3. Score each impression
    # ------------------------------------------------------------------
    behaviors = (
        dataset_provider.val_behaviors_data
        if mode == "val"
        else dataset_provider.test_behaviors_data
    )

    if "histories_news_ids" not in behaviors:
        logger.warning("DIGAT eval: histories_news_ids not available, returning dummy metrics")
        return {"auc": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0, "num_impressions": 0}

    hist_ids_raw = np.asarray(behaviors["histories_news_ids"]).astype(np.int64)
    if id_remap is not None:
        hist_ids = np.where(hist_ids_raw < len(id_remap), id_remap[hist_ids_raw], 0).astype(np.int64)
    else:
        hist_ids = np.clip(hist_ids_raw, 0, num_news - 1)
    hist_cats = np.asarray(behaviors["history_news_categories"]).astype(np.int64)
    impression_ids = np.asarray(behaviors["impression_ids"])
    labels_all = behaviors["labels"]

    # Parse candidate IDs per impression (variable length)
    from src.frameworks.pytorch.digat_features import build_user_graphs

    user_graphs = build_user_graphs(hist_cats, max_history, num_categories)

    group_labels = []
    group_preds = []

    num_impressions = len(impression_ids)
    logger.info(f"Scoring {num_impressions} impressions...")

    for idx in range(num_impressions):
        imp_labels = np.asarray(labels_all[idx])
        # Get candidate news IDs for this impression
        if "candidate_news_ids" in behaviors:
            cand_ids_raw = np.asarray(behaviors["candidate_news_ids"][idx])
            # For variable-length impressions at test time
            if hasattr(cand_ids_raw, '__len__'):
                cand_ids = cand_ids_raw.astype(np.int64)
            else:
                cand_ids = np.array([int(cand_ids_raw)])
        else:
            continue

        # Remap to SAG index space
        if id_remap is not None:
            cand_ids = np.where(cand_ids < len(id_remap), id_remap[cand_ids], 0).astype(np.int64)
        else:
            cand_ids = np.clip(cand_ids, 0, num_news - 1)
        valid_mask = cand_ids > 0
        if not valid_mask.any():
            continue
        cand_ids = cand_ids[valid_mask]
        imp_labels = imp_labels[valid_mask] if len(imp_labels) == len(valid_mask) else imp_labels

        C = len(cand_ids)

        # User history embeddings
        h_ids = hist_ids[idx]  # (H,)
        user_hist_emb = torch.tensor(
            all_news_emb[h_ids], dtype=torch.float32, device=device
        ).unsqueeze(0).expand(C, -1, -1)  # (C, H, D)

        # User graph
        u_graph = torch.tensor(
            user_graphs["user_graph"][idx], dtype=torch.float32, device=device
        ).unsqueeze(0).expand(C, -1, -1)  # (C, G_u, G_u)
        u_cat_mask = torch.tensor(
            user_graphs["user_category_mask"][idx], dtype=torch.float32, device=device
        ).unsqueeze(0).expand(C, -1)  # (C, num_cat)
        u_cat_indices = torch.tensor(
            user_graphs["user_category_indices"][idx], dtype=torch.long, device=device
        ).unsqueeze(0).expand(C, -1)  # (C, H)

        # Candidate SAG embeddings
        cand_node_ids = news_node_ID[cand_ids]  # (C, G_n)
        cand_sag_emb = torch.tensor(
            all_news_emb[cand_node_ids], dtype=torch.float32, device=device
        )  # (C, G_n, D)
        cand_graph = torch.tensor(
            news_graph_adj[cand_ids].astype(np.float32), device=device
        )  # (C, G_n, G_n)
        cand_mask = torch.tensor(
            news_graph_mask_arr[cand_ids].astype(np.float32), device=device
        )  # (C, G_n)

        # Run dual graph interaction
        news_ctx, user_ctx = model.graph_encoder(
            cand_sag_emb, cand_graph, cand_mask,
            user_hist_emb, u_graph, u_cat_mask, u_cat_indices,
            num_categories,
        )  # each (C, D)

        scores = (news_ctx * user_ctx).sum(dim=-1).cpu().numpy()  # (C,)
        group_labels.append(imp_labels)
        group_preds.append(scores)

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    from src.core.models.evaluations.utils import compute_metrics

    with Progress(transient=True) as progress:
        metrics = compute_metrics(group_labels, group_preds, metrics_calculator, progress)
    metrics["num_impressions"] = len(group_labels)
    return metrics
