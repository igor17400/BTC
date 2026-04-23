"""PP-Rec evaluation with full popularity-aware scoring.

Unlike the default :func:`fast_evaluate` (dot-product only), this uses
the full PP-Rec formula:

    score = eta * relevance_score + (1 - eta) * popularity_score

where ``eta`` is the per-user activity gate and ``popularity_score``
comes from the bias news encoder + PopularityPredictor.

This requires access to the full model (not just encoders) and the
raw behaviors data (for per-candidate CTR and recency).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.io.progress import ProgressManager

from .utils import compute_metrics, precompute_news_vectors, precompute_user_vectors

# ---------------------------------------------------------------------------
# PP-Rec evaluation
# ---------------------------------------------------------------------------


def pprec_fast_evaluate(
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
    mode: str = "validate",
) -> dict[str, float]:
    """PP-Rec evaluation with full popularity-aware scoring.

    Args:
        model: Full PP-Rec model (needs ``news_encoder``, ``bias_news_encoder``,
            ``user_encoder``, ``activity_gater``, ``popularity_predictor``).
        news_dataloader: Iterable yielding ``{"news_id", "news_features"}``.
        user_hist_dataloader: Iterable yielding ``(imp_ids, user_ids, features)``.
        impression_iterator: Iterable yielding ``(_, labels, imp_id, cand_ids)``.
        metrics_calculator: Object with ``METRIC_NAMES`` and ``compute_metrics``.
        progress: Progress bar manager.
        adapter: Framework adapter implementing :class:`FrameworkAdapter`.
        behaviors_data: Dict with optional ``candidate_news_ctr``,
            ``candidate_news_recency``, ``candidate_news_ids``.
        int_to_news_id_map: Optional ``{int: news_id_str}`` lookup.
        save_predictions_path: Optional path to dump predictions for analysis.
        epoch: Current epoch number (only used when saving predictions).
        mode: ``"validate"`` or ``"test"``.

    Returns:
        ``{metric_name: value}`` dictionary.
    """
    # 1. Precompute relevance news vectors
    rel_news_vecs = precompute_news_vectors(
        model.news_encoder, news_dataloader, adapter, progress
    )

    # 2. Precompute bias news vectors (same dataloader, different encoder)
    bias_news_vecs = precompute_news_vectors(
        model.bias_news_encoder, news_dataloader, adapter, progress
    )

    # 3. Precompute user vectors
    user_vecs = precompute_user_vectors(
        model.user_encoder,
        user_hist_dataloader,
        adapter,
        progress,
        process_user_id=False,
    )

    # 4. Precompute activity gate eta per user (numpy, from user_vecs)
    #    We run the gate on all user vecs at once for efficiency.
    user_etas: dict[int, float] = {}
    if model.activity_gater is not None:
        all_imp_ids = list(user_vecs.keys())
        all_user_vecs_np = np.stack([user_vecs[k] for k in all_imp_ids], axis=0)
        eta_np = adapter.run_activity_gater(model.activity_gater, all_user_vecs_np)
        for i, imp_id in enumerate(all_imp_ids):
            user_etas[imp_id] = float(eta_np[i])

    # 5. Build per-news popularity scores.
    #    The PopularityPredictor needs (bias_vec, recency, ctr) per news.
    #    We precompute pop_score per news using dataset-level CTR/recency.
    behaviors_data = behaviors_data or {}
    cand_ctr_data = behaviors_data.get("candidate_news_ctr")
    cand_recency_data = behaviors_data.get("candidate_news_recency")

    # 6. Score every impression with the full formula
    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []

    imp_task = progress.add_task(
        "Scoring impressions (PP-Rec)...",
        total=len(impression_iterator),
        visible=True,
    )

    for idx, impression in enumerate(impression_iterator):
        _, labels, impression_id, cand_ids = impression

        user_vector = user_vecs.get(int(impression_id))
        if user_vector is None:
            progress.update(imp_task, advance=1)
            continue

        cand_ids_np = adapter.to_numpy(cand_ids)
        eta = user_etas.get(int(impression_id), 0.5)

        rel_vecs = []
        bias_vecs = []
        for nid in cand_ids_np:
            if isinstance(nid, (str, np.str_)):
                news_key = str(nid)
            elif int_to_news_id_map and nid in int_to_news_id_map:
                news_key = int_to_news_id_map[nid]
            else:
                news_key = f"N{nid}"
            rv = rel_news_vecs.get(news_key)
            bv = bias_news_vecs.get(news_key)
            if rv is not None and bv is not None:
                rel_vecs.append(rv)
                bias_vecs.append(bv)

        if not rel_vecs:
            scores = np.array([])
        else:
            rel_mat = np.stack(rel_vecs, axis=0)
            bias_mat = np.stack(bias_vecs, axis=0)

            # Relevance scores (dot product)
            rel_scores = rel_mat @ user_vector

            # Popularity scores via the predictor
            # Get per-candidate recency/CTR for this impression
            recency_arr = None
            ctr_arr = None
            if cand_recency_data is not None:
                recency_arr = adapter.to_numpy(cand_recency_data[idx])
            if cand_ctr_data is not None:
                ctr_arr = adapter.to_numpy(cand_ctr_data[idx]).astype(np.float32)

            # Run popularity predictor through adapter (framework-native)
            pop_input = {
                "bias_vecs": bias_mat,
                "recency": recency_arr,
                "ctr": ctr_arr,
            }
            pop_scores = _compute_pprec_pop_scores(
                model.popularity_predictor, pop_input, adapter
            )

            # Full PP-Rec formula
            scores = eta * rel_scores + (1.0 - eta) * pop_scores

        labels_np = adapter.to_numpy(labels)
        group_labels.append(labels_np)
        group_preds.append(scores)
        progress.update(imp_task, advance=1)

    progress.remove_task(imp_task)

    final_metrics = compute_metrics(
        group_labels, group_preds, metrics_calculator, progress
    )
    final_metrics["num_impressions"] = len(group_labels)
    return final_metrics


def _compute_pprec_pop_scores(
    popularity_predictor: Any,
    pop_input: dict,
    adapter: Any,
) -> np.ndarray:
    """Run the PopularityPredictor on a batch of candidates.

    Args:
        popularity_predictor: Framework-native PopularityPredictor module.
        pop_input: Dict with ``bias_vecs`` (C, news_dim), optional
            ``recency`` (C,) int, optional ``ctr`` (C,) float.
        adapter: Framework adapter.

    Returns:
        (C,) numpy array of popularity scores.
    """
    bias_vecs = pop_input["bias_vecs"]  # (C, news_dim) numpy
    recency = pop_input.get("recency")
    ctr = pop_input.get("ctr")

    try:
        pop_scores = adapter.run_popularity_predictor(
            popularity_predictor, bias_vecs, recency, ctr
        )
    except AttributeError:
        # Adapter doesn't support run_popularity_predictor yet —
        # fall back to content-only scoring (no recency/CTR).
        pop_scores = adapter.encode_news(popularity_predictor, bias_vecs)
    return pop_scores
