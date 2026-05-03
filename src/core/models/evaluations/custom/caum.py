"""CAUM evaluation with candidate-aware scoring.

Unlike the default evaluator (single user vector -> dot product), CAUM
produces a different user representation for each candidate via its
candidate-aware inter_model.  The evaluator:

1. Precomputes news vectors for all articles (same as default).
2. For each impression, looks up clicked news vectors from the precomputed
   pool and candidate news vectors, then runs ``model.inter_model`` to
   score each candidate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.io.progress import ProgressManager
from src.core.io.saving import save_predictions_to_file_fn

from ..utils import compute_metrics, precompute_news_vectors


def caum_fast_evaluate(
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
    """Candidate-aware evaluation for CAUM.

    Algorithm:
    1. Precompute news vectors for every article.
    2. Precompute user history vectors (clicked news) per impression by
       looking up precomputed news vectors.
    3. For each impression: gather candidate news vectors, score each
       candidate via the inter_model, record labels + scores.
    4. Aggregate per-impression ranking metrics.
    """
    news_encoder = model.news_encoder

    # 1. Precompute news vectors
    news_vecs = precompute_news_vectors(
        news_encoder, news_dataloader, adapter, progress
    )

    # 2. Precompute clicked-news matrices per impression
    user_clicked_vecs = _precompute_user_clicked_news(
        user_hist_dataloader, news_vecs, adapter, progress,
        news_encoder=news_encoder,
    )

    # 3. Score every impression via inter_model
    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []
    predictions_to_save: dict = {}

    imp_task = progress.add_task(
        "Processing impressions...", total=len(impression_iterator), visible=True
    )

    for impression in impression_iterator:
        _, labels, impression_id, cand_ids = impression

        clicked_matrix = user_clicked_vecs.get(int(impression_id))
        if clicked_matrix is None:
            progress.update(imp_task, advance=1)
            continue

        cand_ids_np = adapter.to_numpy(cand_ids)

        # Gather candidate news vectors
        cand_vectors = []
        for nid in cand_ids_np:
            if isinstance(nid, (str, np.str_)):
                news_key = str(nid)
            elif int_to_news_id_map and nid in int_to_news_id_map:
                news_key = int_to_news_id_map[nid]
            else:
                news_key = f"N{nid}"
            vec = news_vecs.get(news_key)
            if vec is not None:
                cand_vectors.append(vec)

        if not cand_vectors:
            scores = np.array([])
        else:
            # Score each candidate via inter_model
            scores = adapter.score_caum_impression(
                model.inter_model,
                cand_vectors=np.stack(cand_vectors, axis=0),
                clicked_vecs=clicked_matrix,
            )

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


def _precompute_user_clicked_news(
    user_dataloader: Any,
    news_vecs: dict,
    adapter: Any,
    progress: ProgressManager,
    news_encoder: Any,
) -> dict[int, np.ndarray]:
    """Build per-impression clicked-news matrices.

    For each impression, encodes the user's click history through the
    news encoder to produce an ``(H, D)`` matrix of clicked news vectors.

    Returns:
        ``{impression_id: (H, D) numpy array}`` dictionary.
    """
    user_clicked: dict[int, np.ndarray] = {}
    task = progress.add_task(
        "Computing user clicked-news vectors...",
        total=len(user_dataloader),
        visible=True,
    )

    for impression_ids, _user_ids, features in user_dataloader:
        # features: (B, H, T) — history token sequences
        # Encode each click through news_encoder
        features_np = adapter.to_numpy(features)
        B, H, T = features_np.shape

        flat = features_np.reshape(B * H, T)
        flat_vecs = adapter.encode_news(news_encoder, flat)  # (B*H, D)
        hist_vecs = flat_vecs.reshape(B, H, -1)  # (B, H, D)

        for i, imp_id in enumerate(impression_ids):
            user_clicked[int(imp_id)] = hist_vecs[i]  # (H, D)

        progress.update(task, advance=len(impression_ids))

    progress.remove_task(task)
    return user_clicked
