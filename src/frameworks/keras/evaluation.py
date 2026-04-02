"""Fast evaluation with precomputed vectors for Keras models.

Provides a standalone ``fast_evaluate`` function isomorphic with the
PyTorch and JAX evaluation modules.
"""

from typing import Any

import numpy as np
from keras import ops
from rich.progress import Progress


def precompute_news_vectors(
    news_encoder,
    news_dataloader,
    progress: Progress,
) -> dict[str, np.ndarray]:
    """Precompute news vectors using a Keras news encoder.

    Args:
        news_encoder: Callable ``(features, training=False) -> (B, D)``.
        news_dataloader: Iterable yielding
            ``{"news_id": ..., "news_features": ...}``.
        progress: Rich progress bar.

    Returns:
        ``{news_id_str: numpy_vector}`` dictionary.
    """
    news_vecs: dict[str, np.ndarray] = {}
    task = progress.add_task(
        "Computing news vectors...", total=len(news_dataloader), visible=True
    )

    for batch in news_dataloader:
        news_ids = batch["news_id"]
        news_features = batch["news_features"]

        batch_vecs = ops.convert_to_numpy(
            news_encoder(news_features, training=False)
        )

        for i, nid in enumerate(news_ids):
            key = ops.convert_to_numpy(nid).item() if hasattr(nid, "item") else nid
            news_vecs[key] = batch_vecs[i]

        progress.update(task, advance=len(news_ids))

    progress.remove_task(task)
    return news_vecs


def precompute_user_vectors(
    user_encoder,
    user_dataloader,
    progress: Progress,
    process_user_id: bool = False,
) -> dict[int, np.ndarray]:
    """Precompute user vectors using a Keras user encoder.

    Args:
        user_encoder: Callable. Signature depends on *process_user_id*.
        user_dataloader: Iterable yielding
            ``(impression_ids, user_ids_or_None, features)``.
        progress: Rich progress bar.
        process_user_id: If ``True`` the encoder is called as
            ``user_encoder([features, user_ids], training=False)``.

    Returns:
        ``{impression_id: numpy_vector}`` dictionary.
    """
    user_vecs: dict[int, np.ndarray] = {}
    task = progress.add_task(
        "Computing user vectors...", total=len(user_dataloader), visible=True
    )

    for impression_ids, user_ids, features in user_dataloader:
        if process_user_id:
            vec = user_encoder([features, user_ids], training=False)
        else:
            vec = user_encoder(features, training=False)

        vec_np = ops.convert_to_numpy(vec)

        for i, imp_id in enumerate(impression_ids):
            user_vecs[int(imp_id)] = vec_np[i]

        progress.update(task, advance=len(impression_ids))

    progress.remove_task(task)
    return user_vecs


def fast_evaluate(
    news_encoder,
    user_encoder,
    user_hist_dataloader,
    news_dataloader,
    impression_iterator,
    metrics_calculator,
    progress: Progress,
    process_user_id: bool = False,
    int_to_news_id_map: dict[int, str] | None = None,
) -> dict[str, float]:
    """Full fast-evaluation pipeline.

    Isomorphic with ``fast_evaluate`` in PyTorch and JAX evaluation modules.

    1. Precompute news vectors.
    2. Precompute user vectors.
    3. Score impressions via dot product.
    4. Aggregate metrics using *metrics_calculator*.

    Returns:
        ``{metric_name: value}`` dictionary.
    """
    news_vecs = precompute_news_vectors(news_encoder, news_dataloader, progress)
    user_vecs = precompute_user_vectors(
        user_encoder, user_hist_dataloader, progress, process_user_id
    )

    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []

    imp_task = progress.add_task(
        "Processing impressions...", total=len(impression_iterator), visible=True
    )

    for _, labels, impression_id, cand_ids in impression_iterator:
        user_vector = user_vecs.get(int(impression_id))
        if user_vector is None:
            continue

        cand_ids_np = ops.convert_to_numpy(cand_ids) if hasattr(cand_ids, "numpy") else np.asarray(cand_ids)

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
            scores = np.dot(news_mat, user_vector)

        group_labels.append(ops.convert_to_numpy(labels) if hasattr(labels, "numpy") else np.asarray(labels))
        group_preds.append(scores)
        progress.update(imp_task, advance=1)

    progress.remove_task(imp_task)

    # Aggregate metrics
    val_loss_total = 0.0
    num_valid = 0
    metric_agg: dict[str, list] = {k: [] for k in metrics_calculator.METRIC_NAMES}

    for labels_np, scores_np in zip(group_labels, group_preds):
        if labels_np.size == 0 or scores_np.size == 0:
            continue
        if np.isnan(scores_np).any() or np.isinf(scores_np).any():
            continue

        # Softmax-based loss
        scores_tensor = ops.convert_to_tensor([scores_np], dtype="float32")
        labels_tensor = ops.convert_to_tensor([labels_np], dtype="float32")
        probs = ops.softmax(scores_tensor, axis=-1)
        loss = -ops.sum(labels_tensor * ops.log(probs + 1e-7), axis=-1)
        val_loss_total += float(ops.convert_to_numpy(ops.mean(loss)))
        num_valid += 1

        imp_metrics = metrics_calculator.compute_metrics(
            y_true=labels_np, y_pred_logits=scores_np
        )
        for name, value in imp_metrics.items():
            if name in metric_agg:
                metric_agg[name].append(float(value))

    final: dict[str, float] = {
        "loss": val_loss_total / num_valid if num_valid > 0 else 0.0,
    }
    for name, vals in metric_agg.items():
        final[name] = float(np.mean(vals)) if vals else 0.0
    final["num_impressions"] = len(group_labels)

    return final
