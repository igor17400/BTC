"""Framework-agnostic fast evaluation for news recommendation models.

Replaces the previously-duplicated ``fast_evaluate`` implementations in
``src/frameworks/{keras,pytorch,jax}/evaluation.py``. The shared
algorithm here calls into a small :class:`FrameworkAdapter` for the
operations that fundamentally require framework knowledge — invoking
encoder modules and converting tensors to numpy — and does everything
else (impression iteration, dot-product scoring, NaN handling, loss,
metric aggregation) in pure numpy.

**Loss formula:** evaluation loss is computed as numpy softmax + cross
entropy. This is intentional and framework-independent. The training
loss path is unchanged and still uses each framework's native loss
function — only the *evaluation* loss reported alongside AUC/MRR/nDCG
goes through this module. Numerical results across frameworks are now
bit-identical for AUC/MRR/nDCG and within ~1e-6 for loss.

**PP-Rec note:** PP-Rec is currently evaluated by the same dot-product
scoring path as NRMS/NAML/LSTUR (see the ``score_candidates`` methods
on each framework's PP-Rec class). The popularity predictor and
activity gate are trained but not used at inference time. If that
changes in the future, add a ``score_impressions`` extension hook here
and let PP-Rec override it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.progress import Progress

from src.core.io.saving import save_predictions_to_file_fn
from src.core.models.adapter import FrameworkAdapter

# ---------------------------------------------------------------------------
# Pre-computation helpers (numpy after the adapter call)
# ---------------------------------------------------------------------------


def precompute_news_vectors(
    news_encoder: Any,
    news_dataloader: Any,
    adapter: FrameworkAdapter,
    progress: Progress,
) -> dict:
    """Precompute news vectors for every article in the dataloader.

    Args:
        news_encoder: Framework-native news encoder module.
        news_dataloader: Iterable yielding ``{"news_id": ..., "news_features": ...}``.
        adapter: Framework adapter providing :meth:`encode_news` and :meth:`to_numpy`.
        progress: Rich progress bar manager.

    Returns:
        ``{news_id: numpy_vector}`` dictionary. Keys are whatever the
        dataloader emits as ``news_id`` (typically a string like ``"N42"``).
    """
    news_vecs: dict = {}
    task = progress.add_task(
        "Computing news vectors...", total=len(news_dataloader), visible=True
    )

    for batch in news_dataloader:
        news_ids = batch["news_id"]
        features = batch["news_features"]

        vecs_np = adapter.encode_news(news_encoder, features)

        if np.isnan(vecs_np).any() or np.isinf(vecs_np).any():
            progress.console.print(
                f"[WARNING fast_evaluate] news vectors have NaN/Inf — "
                f"min={float(np.min(vecs_np)):.6f} max={float(np.max(vecs_np)):.6f}"
            )

        for i, nid in enumerate(news_ids):
            key = nid.item() if hasattr(nid, "item") else nid
            news_vecs[key] = vecs_np[i]

        progress.update(task, advance=len(news_ids))

    progress.remove_task(task)
    return news_vecs


def precompute_user_vectors(
    user_encoder: Any,
    user_dataloader: Any,
    adapter: FrameworkAdapter,
    progress: Progress,
    process_user_id: bool = False,
) -> dict[int, np.ndarray]:
    """Precompute user vectors for every impression in the dataloader.

    Args:
        user_encoder: Framework-native user encoder module.
        user_dataloader: Iterable yielding
            ``(impression_ids, user_ids_or_None, features)``.
        adapter: Framework adapter providing :meth:`encode_user`.
        progress: Rich progress bar manager.
        process_user_id: ``True`` for LSTUR-style encoders that need an
            explicit user-id tensor.

    Returns:
        ``{impression_id: numpy_vector}`` dictionary.
    """
    user_vecs: dict[int, np.ndarray] = {}
    task = progress.add_task(
        "Computing user vectors...", total=len(user_dataloader), visible=True
    )

    for impression_ids, user_ids, features in user_dataloader:
        vecs_np = adapter.encode_user(user_encoder, features, user_ids, process_user_id)

        if np.isnan(vecs_np).any() or np.isinf(vecs_np).any():
            progress.console.print(
                f"[WARNING fast_evaluate] user vectors have NaN/Inf — "
                f"min={float(np.min(vecs_np)):.6f} max={float(np.max(vecs_np)):.6f}"
            )

        for i, imp_id in enumerate(impression_ids):
            user_vecs[int(imp_id)] = vecs_np[i]

        progress.update(task, advance=len(impression_ids))

    progress.remove_task(task)
    return user_vecs


# ---------------------------------------------------------------------------
# Numerically-stable numpy primitives (the canonical eval-loss formula)
# ---------------------------------------------------------------------------


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _compute_metrics(
    group_labels: list[np.ndarray],
    group_preds: list[np.ndarray],
    metrics_calculator: Any,
    progress: Progress,
) -> dict[str, float]:
    """Compute eval loss + ranking metrics from per-impression labels and scores.

    Eval loss is numpy softmax + cross-entropy (canonical, framework-
    independent). Ranking metrics are delegated to ``metrics_calculator``,
    which is itself a numpy implementation in :mod:`src.core.metrics`.
    """
    val_loss_total = 0.0
    num_valid = 0
    metric_agg: dict[str, list] = {k: [] for k in metrics_calculator.METRIC_NAMES}

    for labels_np, scores_np in zip(group_labels, group_preds):
        if labels_np.size == 0 or scores_np.size == 0:
            continue
        if np.isnan(scores_np).any() or np.isinf(scores_np).any():
            progress.console.print(
                "[WARNING fast_evaluate] NaN/Inf detected in scores. Skipping impression."
            )
            continue

        # Pure numpy softmax + cross-entropy. This is the canonical eval
        # loss formula across all frameworks.
        probs = _stable_softmax(scores_np[None, :])
        loss = -np.sum(labels_np[None, :] * np.log(probs + 1e-7), axis=-1)
        val_loss_total += float(loss.mean())
        num_valid += 1

        impression_metrics = metrics_calculator.compute_metrics(
            y_true=labels_np, y_pred_logits=scores_np
        )
        for name, value in impression_metrics.items():
            if name in metric_agg:
                metric_agg[name].append(float(value))

    final: dict[str, float] = {
        "loss": (val_loss_total / num_valid) if num_valid > 0 else 0.0,
    }
    for name, vals in metric_agg.items():
        final[name] = float(np.mean(vals)) if vals else 0.0
    return final


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


def fast_evaluate(
    *,
    news_encoder: Any,
    user_encoder: Any,
    user_hist_dataloader: Any,
    news_dataloader: Any,
    impression_iterator: Any,
    metrics_calculator: Any,
    progress: Progress,
    adapter: FrameworkAdapter,
    process_user_id: bool = False,
    int_to_news_id_map: dict[int, str] | None = None,
    save_predictions_path: str | None = None,
    epoch: int | None = None,
    mode: str = "validate",
) -> dict[str, float]:
    """Framework-agnostic fast evaluation pipeline.

    Algorithm (identical across all frameworks):

    1. Precompute news vectors for every candidate news article.
    2. Precompute user vectors for every impression in the user-history loader.
    3. For each impression: gather candidate vectors, score by dot product
       with the user vector, record labels + scores.
    4. Aggregate per-impression ranking metrics and the canonical numpy
       softmax-CE eval loss.

    Args:
        news_encoder: Framework-native news encoder module.
        user_encoder: Framework-native user encoder module.
        user_hist_dataloader: Iterable yielding ``(imp_ids, user_ids, features)``.
        news_dataloader: Iterable yielding ``{"news_id", "news_features"}``.
        impression_iterator: Iterable yielding ``(_, labels, imp_id, cand_ids)``.
        metrics_calculator: Object with ``METRIC_NAMES`` attribute and a
            ``compute_metrics(y_true, y_pred_logits)`` method.
        progress: Rich progress bar manager.
        adapter: Framework adapter implementing :class:`FrameworkAdapter`.
        process_user_id: ``True`` for LSTUR-style encoders.
        int_to_news_id_map: Optional ``{int: news_id_str}`` lookup used when
            the impression iterator emits integer candidate ids.
        save_predictions_path: Optional path to dump predictions for analysis.
        epoch: Current epoch number (only used when saving predictions).
        mode: ``"validate"`` or ``"test"``.

    Returns:
        ``{metric_name: value}`` dictionary including ``loss``,
        ``num_impressions``, and every metric in ``metrics_calculator.METRIC_NAMES``.
    """
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
    final_metrics = _compute_metrics(
        group_labels, group_preds, metrics_calculator, progress
    )
    final_metrics["num_impressions"] = len(group_labels)

    if save_predictions_path:
        save_predictions_to_file_fn(
            predictions_to_save, save_predictions_path, epoch, mode
        )

    return final_metrics
