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
from src.core.io.saving import save_predictions_to_file_fn

from ..utils import compute_metrics, precompute_news_vectors, precompute_user_vectors

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
    mode: str = "val",
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

    # 6. Score every impression with the full formula.
    #
    # JAX-friendly three-pass design (mirrors TCCM's eval):
    #   Pass 1 — Python prep: collect per-impression bias_vecs / recency /
    #            ctr into FLAT arrays + remember (start, stop) offsets.
    #            No JAX calls.
    #   Pass 2 — Batched popularity predictor: one padded chunked JAX
    #            call over all flat inputs. With a fixed chunk size the
    #            traced shape is stable -> XLA compiles ONCE and reuses
    #            the kernel for every chunk. Avoids the per-impression
    #            JIT-recompile pathology that made the original
    #            inner-loop variant unusable on JAX (variable C_i ->
    #            fresh compile per unique candidate count).
    #   Pass 3 — Python scatter: per impression, slice all_pop_scores
    #            and combine with rel_scores via eta * rel + (1-eta) * pop.

    impression_records: list[dict] = []
    flat_bias: list[np.ndarray] = []
    flat_recency: list[np.ndarray] = []
    flat_ctr: list[np.ndarray] = []

    prep_task = progress.add_task(
        "Preparing PP-Rec eval inputs...",
        total=len(impression_iterator),
        visible=True,
    )

    have_recency = cand_recency_data is not None
    have_ctr = cand_ctr_data is not None

    for idx, impression in enumerate(impression_iterator):
        _, labels, impression_id, cand_ids = impression
        labels_np = adapter.to_numpy(labels)

        rec: dict = {
            "labels": labels_np,
            "user_vector": user_vecs.get(int(impression_id)),
            "eta": user_etas.get(int(impression_id), 0.5),
            "rel_mat": None,
            "start": None,
            "stop": None,
        }
        if rec["user_vector"] is None:
            impression_records.append(rec)
            progress.update(prep_task, advance=1)
            continue

        cand_ids_np = adapter.to_numpy(cand_ids)
        rel_vecs: list[np.ndarray] = []
        bias_vecs: list[np.ndarray] = []
        kept_indices: list[int] = []
        for c_pos, nid in enumerate(cand_ids_np):
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
                kept_indices.append(c_pos)

        if not rel_vecs:
            impression_records.append(rec)
            progress.update(prep_task, advance=1)
            continue

        rec["rel_mat"] = np.stack(rel_vecs, axis=0)
        bias_mat = np.stack(bias_vecs, axis=0)
        idx_arr = np.asarray(kept_indices, dtype=np.int64)

        # Subset recency/ctr to the kept candidates so all flat arrays
        # have the same length as bias_mat (= len(rel_vecs)).
        if have_recency:
            full_recency = adapter.to_numpy(cand_recency_data[idx])
            flat_recency.append(full_recency[idx_arr])
        if have_ctr:
            full_ctr = adapter.to_numpy(cand_ctr_data[idx]).astype(np.float32)
            flat_ctr.append(full_ctr[idx_arr])

        start = sum(arr.shape[0] for arr in flat_bias)
        flat_bias.append(bias_mat)
        rec["start"] = start
        rec["stop"] = start + bias_mat.shape[0]
        impression_records.append(rec)
        progress.update(prep_task, advance=1)

    progress.remove_task(prep_task)

    # Pass 2 — Batched popularity predictor (padded chunks).
    if flat_bias:
        all_bias = np.concatenate(flat_bias, axis=0).astype(np.float32)
        all_recency = (
            np.concatenate(flat_recency, axis=0).astype(np.int32)
            if have_recency
            else None
        )
        all_ctr = (
            np.concatenate(flat_ctr, axis=0).astype(np.float32) if have_ctr else None
        )
        N, news_dim = all_bias.shape
        chunk_size = 4096
        all_pop_scores = np.empty(N, dtype=np.float32)
        pop_task = progress.add_task(
            "Computing popularity scores (PP-Rec)...",
            total=int(np.ceil(N / chunk_size)),
            visible=True,
        )
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk_bias = all_bias[s:e]
            chunk_recency = all_recency[s:e] if all_recency is not None else None
            chunk_ctr = all_ctr[s:e] if all_ctr is not None else None
            # Pad the last chunk so XLA traces a single stable shape.
            if e - s < chunk_size:
                pad = chunk_size - (e - s)
                chunk_bias = np.pad(chunk_bias, ((0, pad), (0, 0)), mode="constant")
                if chunk_recency is not None:
                    chunk_recency = np.pad(chunk_recency, (0, pad), mode="constant")
                if chunk_ctr is not None:
                    chunk_ctr = np.pad(chunk_ctr, (0, pad), mode="constant")
            scores = adapter.run_popularity_predictor(
                model.popularity_predictor,
                chunk_bias,
                chunk_recency,
                chunk_ctr,
            )
            all_pop_scores[s:e] = np.asarray(scores)[: e - s]
            progress.update(pop_task, advance=1)
        progress.remove_task(pop_task)
    else:
        all_pop_scores = np.array([], dtype=np.float32)

    # Pass 3 — Python scatter: fuse rel + pop per impression.
    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []

    fuse_task = progress.add_task(
        "Scoring impressions (PP-Rec)...",
        total=len(impression_records),
        visible=True,
    )
    for rec in impression_records:
        labels_np = rec["labels"]
        if rec["user_vector"] is None or rec["rel_mat"] is None:
            scores = np.array([])
        else:
            rel_scores = rec["rel_mat"] @ rec["user_vector"]
            pop_scores = all_pop_scores[rec["start"] : rec["stop"]]
            scores = rec["eta"] * rel_scores + (1.0 - rec["eta"]) * pop_scores
        group_labels.append(labels_np)
        group_preds.append(scores)
        progress.update(fuse_task, advance=1)

    progress.remove_task(fuse_task)

    final_metrics = compute_metrics(
        group_labels, group_preds, metrics_calculator, progress
    )
    final_metrics["_num_impressions"] = len(group_labels)

    if save_predictions_path:
        predictions = {}
        for i, (labels, preds) in enumerate(zip(group_labels, group_preds)):
            predictions[str(i)] = (labels.tolist(), preds.tolist())
        save_predictions_to_file_fn(
            predictions, save_predictions_path, epoch, mode=mode
        )

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
