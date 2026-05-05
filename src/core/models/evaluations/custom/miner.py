"""MINER evaluation with multi-interest scoring.

Unlike the default evaluator (single user vector → dot product), MINER
computes K interest vectors per user and aggregates per-interest scores
via MINER-weighted (target-aware attention) or MINER-mean.

The evaluator precomputes:
1. News vectors (same as default).
2. K interest vectors per user (via ``encode_interests``).

Then for each impression it computes per-interest dot products and
aggregates them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.io.progress import ProgressManager
from src.core.io.saving import save_predictions_to_file_fn

from ..utils import compute_metrics, precompute_news_vectors


def miner_fast_evaluate(
    *,
    model: Any,
    news_dataloader: Any,
    user_hist_dataloader: Any,
    impression_iterator: Any,
    metrics_calculator: Any,
    progress: ProgressManager,
    adapter: Any,
    behaviors_data: dict | None = None,
    int_to_news_id_map: dict[int, str] | None = None,
    save_predictions_path: str | None = None,
    epoch: int | None = None,
    mode: str = "val",
) -> dict[str, float]:
    """Multi-interest evaluation for MINER.

    Algorithm:
    1. Precompute news vectors for every candidate news article.
    2. Precompute K interest vectors for every user impression.
    3. For each impression: compute per-interest scores with each
       candidate, aggregate via mean, record labels + scores.
    4. Aggregate per-impression ranking metrics.
    """
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # 1. Precompute news vectors (reuse standard helper)
    news_vecs = precompute_news_vectors(
        news_encoder, news_dataloader, adapter, progress
    )

    # 2. Precompute K interest vectors per user
    user_interest_vecs = _precompute_user_interests(
        user_encoder, user_hist_dataloader, adapter, progress
    )

    # 3. Score every impression with multi-interest matching
    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []
    predictions_to_save: dict = {}

    imp_task = progress.add_task(
        "Processing impressions...", total=len(impression_iterator), visible=True
    )

    for impression in impression_iterator:
        _, labels, impression_id, cand_ids = impression

        interest_vectors = user_interest_vecs.get(int(impression_id))
        if interest_vectors is None:
            progress.update(imp_task, advance=1)
            continue

        cand_ids_np = adapter.to_numpy(cand_ids)

        # Gather candidate news vectors
        news_vectors = []
        for nid in cand_ids_np:
            if isinstance(nid, (str, np.str_)):
                news_key = str(nid)
            elif int_to_news_id_map and nid in int_to_news_id_map:
                news_key = int_to_news_id_map[nid]
            else:
                news_key = f"N{nid}"
            vec = news_vecs.get(news_key)
            if vec is not None:
                news_vectors.append(vec)

        if not news_vectors:
            scores = np.array([])
        else:
            news_mat = np.stack(news_vectors, axis=0)  # (C, E)

            # Per-interest scores: (K, C)
            # interest_vectors: (K, E), news_mat: (C, E)
            per_interest_scores = interest_vectors @ news_mat.T  # (K, C)

            # MINER-mean aggregation: average across K interests
            scores = np.mean(per_interest_scores, axis=0)  # (C,)

        labels_np = adapter.to_numpy(labels)
        group_labels.append(labels_np)
        group_preds.append(scores)

        if save_predictions_path:
            predictions_to_save[str(cand_ids_np)] = (
                labels_np.tolist(),
                scores.tolist(),
            )

        progress.update(imp_task, advance=1)

    progress.remove_task(imp_task)

    # 4. Compute metrics
    final_metrics = compute_metrics(
        group_labels, group_preds, metrics_calculator, progress
    )
    final_metrics["num_impressions"] = len(group_labels)

    if save_predictions_path:
        save_predictions_to_file_fn(
            predictions_to_save, save_predictions_path, epoch, mode
        )

    return final_metrics


def _precompute_user_interests(
    user_encoder: Any,
    user_dataloader: Any,
    adapter: Any,
    progress: ProgressManager,
) -> dict[int, np.ndarray]:
    """Precompute K interest vectors for every user impression.

    Returns:
        ``{impression_id: (K, E) numpy array}`` dictionary.
    """
    user_vecs: dict[int, np.ndarray] = {}
    task = progress.add_task(
        "Computing user interest vectors...",
        total=len(user_dataloader),
        visible=True,
    )

    for impression_ids, user_ids, features in user_dataloader:
        # encode_interests returns (B, K, E)
        vecs_np = adapter.encode_user_interests(user_encoder, features)

        for i, imp_id in enumerate(impression_ids):
            user_vecs[int(imp_id)] = vecs_np[i]  # (K, E)

        progress.update(task, advance=len(impression_ids))

    progress.remove_task(task)
    return user_vecs
