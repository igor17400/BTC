"""Default evaluation pipeline (dot-product scoring).

Used by NRMS, NAML, LSTUR, and any model that scores candidates via a
simple dot product between user and news representations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.progress import Progress

from src.core.io.saving import save_predictions_to_file_fn
from src.core.models.adapter import FrameworkAdapter

from .utils import compute_metrics, precompute_news_vectors, precompute_user_vectors


def fast_evaluate(
    *,
    model: Any,
    news_dataloader: Any,
    user_hist_dataloader: Any,
    impression_iterator: Any,
    metrics_calculator: Any,
    progress: Progress,
    adapter: FrameworkAdapter,
    behaviors_data: dict | None = None,
    int_to_news_id_map: dict[int, str] | None = None,
    save_predictions_path: str | None = None,
    epoch: int | None = None,
    mode: str = "validate",
) -> dict[str, float]:
    """Framework-agnostic fast evaluation pipeline.

    Algorithm:

    1. Precompute news vectors for every candidate news article.
    2. Precompute user vectors for every impression.
    3. For each impression: gather candidate vectors, score by dot product
       with the user vector, record labels + scores.
    4. Aggregate per-impression ranking metrics and the canonical numpy
       softmax-CE eval loss.
    """
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder
    process_user_id = getattr(model, "process_user_id", False)

    # 1. Precompute news vectors
    news_vecs = precompute_news_vectors(
        news_encoder, news_dataloader, adapter, progress
    )

    # 2. Precompute user vectors
    user_vecs = precompute_user_vectors(
        user_encoder, user_hist_dataloader, adapter, progress, process_user_id
    )

    # 3. Score every impression
    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []
    predictions_to_save: dict = {}

    imp_task = progress.add_task(
        "Processing impressions...", total=len(impression_iterator), visible=True
    )

    for impression in impression_iterator:
        _, labels, impression_id, cand_ids = impression

        user_vector = user_vecs.get(int(impression_id))
        if user_vector is None:
            progress.update(imp_task, advance=1)
            continue

        cand_ids_np = adapter.to_numpy(cand_ids)

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
            news_mat = np.stack(news_vectors, axis=0)
            scores = news_mat @ user_vector

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
