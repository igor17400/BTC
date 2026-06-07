"""Shared utilities for evaluation pipelines.

Helpers used by both :mod:`.default` and :mod:`.pp_rec` (and any future
model-specific evaluator).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.io.progress import ProgressManager
from src.core.models.adapter import FrameworkAdapter

# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------


def precompute_news_vectors(
    news_encoder: Any,
    news_dataloader: Any,
    adapter: FrameworkAdapter,
    progress: ProgressManager,
) -> dict:
    """Precompute news vectors for every article in the dataloader.

    Args:
        news_encoder: Framework-native news encoder module.
        news_dataloader: Iterable yielding ``{"news_id": ..., "news_features": ...}``.
        adapter: Framework adapter providing :meth:`encode_news` and :meth:`to_numpy`.
        progress: Progress bar manager.

    Returns:
        ``{news_id: numpy_vector}`` dictionary.
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
            progress.print(
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
    progress: ProgressManager,
    process_user_id: bool = False,
) -> dict[int, np.ndarray]:
    """Precompute user vectors for every impression in the dataloader.

    Args:
        user_encoder: Framework-native user encoder module.
        user_dataloader: Iterable yielding
            ``(impression_ids, user_ids_or_None, features)``.
        adapter: Framework adapter providing :meth:`encode_user`.
        progress: Progress bar manager.
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
            progress.print(
                f"[WARNING fast_evaluate] user vectors have NaN/Inf — "
                f"min={float(np.min(vecs_np)):.6f} max={float(np.max(vecs_np)):.6f}"
            )

        for i, imp_id in enumerate(impression_ids):
            user_vecs[int(imp_id)] = vecs_np[i]

        progress.update(task, advance=len(impression_ids))

    progress.remove_task(task)
    return user_vecs


# ---------------------------------------------------------------------------
# Numerically-stable numpy primitives
# ---------------------------------------------------------------------------


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def compute_metrics(
    group_labels: list[np.ndarray],
    group_preds: list[np.ndarray],
    metrics_calculator: Any,
    progress: ProgressManager,
) -> dict[str, float]:
    """Compute eval loss + ranking metrics from per-impression labels and scores.

    Eval loss is numpy softmax + cross-entropy (canonical, framework-
    independent). Ranking metrics are delegated to ``metrics_calculator``.
    """
    val_loss_total = 0.0
    num_valid = 0
    metric_agg: dict[str, list] = {k: [] for k in metrics_calculator.METRIC_NAMES}

    for labels_np, scores_np in zip(group_labels, group_preds):
        if labels_np.size == 0 or scores_np.size == 0:
            continue
        if np.isnan(scores_np).any() or np.isinf(scores_np).any():
            progress.print(
                "[WARNING fast_evaluate] NaN/Inf detected in scores. Skipping impression."
            )
            continue

        probs = stable_softmax(scores_np[None, :])
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
        # nanmean: skip degenerate impressions (AUC=NaN when one class
        # only), matching reference GLORY's np.nanmean over per-impression
        # sklearn AUC.
        final[name] = float(np.nanmean(vals)) if vals else 0.0
    return final
