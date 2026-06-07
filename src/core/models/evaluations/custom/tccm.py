"""TCCM evaluation pipeline.

Mirrors the PP-Rec evaluator but routes the popularity score through
the TCCM popularity encoder (per-token CTR + reciprocal-power
timeliness) instead of PP-Rec's content-based predictor.

Eval-time fusion (matches the official TCCM ``news_ranking``)::

    score = eta * relevance + (1 - eta) * popularity

If ``config.use_causal_intervention`` is set, the popularity score is
replaced with the configured constant so the recommendation no longer
depends on data-driven popularity (paper §3.4 ``do(P)`` intervention).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.io.progress import ProgressManager

from ..utils import compute_metrics, precompute_news_vectors, precompute_user_vectors


def tccm_fast_evaluate(
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
    tccm_cache: dict | None = None,
    processed_news: dict | None = None,
    save_predictions_path: str | None = None,
    epoch: int | None = None,
    mode: str = "validate",
) -> dict[str, float]:
    """Full TCCM evaluation with per-token popularity + timeliness fusion.

    Args:
        model: Full TCCM model (needs ``news_encoder``, ``user_encoder``,
            ``popularity_encoder``, ``activity_gater``, ``config``).
        news_dataloader: Yields ``{"news_id": ..., "news_features": ...}``.
        user_hist_dataloader: Yields ``(imp_ids, user_ids, features)``.
        impression_iterator: Yields ``(_, labels, imp_id, cand_ids)``.
        metrics_calculator: Object with ``METRIC_NAMES`` /
            ``compute_metrics``.
        progress: Progress bar manager.
        adapter: Framework adapter (must implement
            :meth:`run_tccm_popularity_encoder`).
        behaviors_data: Validation/test behaviour data dict — needed for
            per-impression candidate IDs and impression timestamps.
        int_to_news_id_map: ``int -> "Nxxx"`` lookup for the relevance
            news-vector cache.
        tccm_cache: ``_build_tccm_cache`` output: token-CTR tables +
            wall-clock anchor + bucket size.
        processed_news: Processed news dict (titles, entities, publish
            times, ID mapping).
        mode: ``"validate"`` or ``"test"``.
    """
    if tccm_cache is None or processed_news is None or behaviors_data is None:
        raise ValueError(
            "tccm_fast_evaluate needs tccm_cache, processed_news, and behaviors_data"
        )

    rel_news_vecs = precompute_news_vectors(
        model.news_encoder, news_dataloader, adapter, progress
    )
    user_vecs = precompute_user_vectors(
        model.user_encoder,
        user_hist_dataloader,
        adapter,
        progress,
        process_user_id=False,
    )

    # Activity gate per user (vectorised).
    user_etas: dict[int, float] = {}
    if model.activity_gater is not None:
        all_imp_ids = list(user_vecs.keys())
        if all_imp_ids:
            stacked = np.stack([user_vecs[k] for k in all_imp_ids], axis=0)
            eta_np = adapter.run_activity_gater(model.activity_gater, stacked)
            for i, imp_id in enumerate(all_imp_ids):
                user_etas[imp_id] = float(eta_np[i])

    # Lookup tables for the popularity branch.
    news_str_to_idx: dict[str, int] = {
        nid: i for i, nid in enumerate(processed_news["news_ids_original_strings"])
    }
    word_pop = tccm_cache["word_pop"]
    entity_pop = tccm_cache["entity_pop"]
    dataset_start = tccm_cache["dataset_start"]
    bucket_hours = tccm_cache["bucket_hours"]
    timeliness_bins = tccm_cache["timeliness_bins"]
    num_buckets = word_pop.shape[0]
    title_tokens = processed_news["tokens"]
    entity_indices = processed_news["entity_indices"]
    publish_time = processed_news["news_publish_time"]
    title_len = title_tokens.shape[1]

    cfg = getattr(model, "config", None)
    use_intervention = bool(getattr(cfg, "use_causal_intervention", False))
    intervention_value = float(getattr(cfg, "intervention_value", 0.5))

    impression_times = behaviors_data.get("impression_times")

    # Pre-compute wall-clock buckets for every eval impression once.
    import pandas as pd

    if impression_times is not None and len(impression_times) > 0:
        impr_ts = pd.to_datetime(np.asarray(impression_times))
        delta_h = (
            impr_ts - pd.Timestamp(dataset_start)
        ).total_seconds().to_numpy() / 3600.0
        impr_buckets = np.clip(
            np.floor(delta_h / bucket_hours).astype(np.int64) - 1,
            0,
            num_buckets - 1,
        )
        impr_hours = delta_h.astype(np.float64)
    else:
        impr_buckets = None
        impr_hours = None

    # Map impression_id -> row index in behaviors_data so we can pull the
    # right candidate IDs and timestamp without scanning every row.
    imp_id_to_row: dict[int, int] = {}
    if "impression_ids" in behaviors_data:
        for i, iid in enumerate(behaviors_data["impression_ids"]):
            imp_id_to_row[int(iid)] = i

    publish_h_lookup = (
        pd.to_datetime(np.asarray(publish_time)) - pd.Timestamp(dataset_start)
    ).total_seconds().to_numpy() / 3600.0

    # ------------------------------------------------------------------
    # Pass 1 — resolve per-impression candidate row indices and assemble
    # ALL popularity inputs into one flat numpy batch. This avoids paying
    # the JAX dispatch / JIT-cache miss cost per impression (which would
    # be ~6 s/imp × 7.8K imps ≈ 14 h on JAX backend).
    # ------------------------------------------------------------------
    impression_records: list[dict] = []
    flat_bucket_inputs: list[np.ndarray] = []
    flat_time_inputs: list[np.ndarray] = []
    impr_list = list(impression_iterator)

    prep_task = progress.add_task(
        "Preparing TCCM eval inputs...", total=len(impr_list), visible=True
    )

    for impression in impr_list:
        _, labels, impression_id, cand_ids = impression
        user_vector = user_vecs.get(int(impression_id))
        labels_np = adapter.to_numpy(labels)
        rec = {
            "labels": labels_np,
            "user_vector": user_vector,
            "rel_vecs": None,
            "start": None,
            "stop": None,
            "eta": user_etas.get(int(impression_id), 0.5),
        }
        if user_vector is None:
            impression_records.append(rec)
            progress.update(prep_task, advance=1)
            continue

        cand_ids_np = adapter.to_numpy(cand_ids)
        rel_vecs: list[np.ndarray] = []
        cand_idx_int: list[int] = []
        for nid in cand_ids_np:
            if isinstance(nid, (str, np.str_)):
                news_key = str(nid)
            elif int_to_news_id_map and int(nid) in int_to_news_id_map:
                news_key = int_to_news_id_map[int(nid)]
            else:
                news_key = f"N{int(nid)}"
            rv = rel_news_vecs.get(news_key)
            idx = news_str_to_idx.get(news_key)
            if rv is None or idx is None:
                continue
            rel_vecs.append(rv)
            cand_idx_int.append(int(idx))

        if not rel_vecs:
            impression_records.append(rec)
            progress.update(prep_task, advance=1)
            continue

        rec["rel_vecs"] = np.stack(rel_vecs, axis=0)

        row = imp_id_to_row.get(int(impression_id))
        if row is None or impr_buckets is None or impr_hours is None:
            # No timestamp -> we'll fall back to intervention_value below.
            rec["start"] = -1
            rec["stop"] = -1
            impression_records.append(rec)
            progress.update(prep_task, advance=1)
            continue

        b = int(impr_buckets[row])
        cand_idx_arr = np.asarray(cand_idx_int, dtype=np.int64)
        bucket_input = np.concatenate(
            [
                word_pop[b][title_tokens[cand_idx_arr]],
                entity_pop[b][entity_indices[cand_idx_arr]],
            ],
            axis=-1,
        ).astype(np.int32)
        age = impr_hours[row] - publish_h_lookup[cand_idx_arr]
        age = np.where(np.isnan(age), 0.0, age)
        time_input = np.clip(np.round(age), 0, timeliness_bins - 1).astype(np.int32)

        start = sum(arr.shape[0] for arr in flat_bucket_inputs)
        flat_bucket_inputs.append(bucket_input)
        flat_time_inputs.append(time_input)
        rec["start"] = start
        rec["stop"] = start + bucket_input.shape[0]
        impression_records.append(rec)
        progress.update(prep_task, advance=1)

    progress.remove_task(prep_task)

    # ------------------------------------------------------------------
    # Pass 2 — batched popularity-encoder forward over the flat inputs.
    # ------------------------------------------------------------------
    if flat_bucket_inputs and not use_intervention:
        all_buckets = np.concatenate(flat_bucket_inputs, axis=0)
        all_times = np.concatenate(flat_time_inputs, axis=0)
        N = all_buckets.shape[0]
        # Pad to a fixed chunk size so the JAX trace is stable across calls.
        chunk_size = 4096
        all_pop_scores = np.empty(N, dtype=np.float32)
        pop_task = progress.add_task(
            "Computing popularity scores (TCCM)...",
            total=int(np.ceil(N / chunk_size)),
            visible=True,
        )
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk_buckets = all_buckets[s:e]
            chunk_times = all_times[s:e]
            # Pad the last chunk so the traced shape stays the same.
            if e - s < chunk_size:
                pad = chunk_size - (e - s)
                chunk_buckets = np.pad(
                    chunk_buckets, ((0, pad), (0, 0)), mode="constant"
                )
                chunk_times = np.pad(chunk_times, (0, pad), mode="constant")
            scores = adapter.run_tccm_popularity_encoder(
                model.popularity_encoder,
                chunk_buckets,
                chunk_times,
                title_len=title_len,
            )
            all_pop_scores[s:e] = scores[: e - s]
            progress.update(pop_task, advance=1)
        progress.remove_task(pop_task)
    else:
        all_pop_scores = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Pass 3 — final η · rel + (1-η) · pop fusion per impression.
    # ------------------------------------------------------------------
    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []

    imp_task = progress.add_task(
        "Scoring impressions (TCCM)...", total=len(impression_records), visible=True
    )

    for rec in impression_records:
        labels_np = rec["labels"]
        if rec["user_vector"] is None or rec["rel_vecs"] is None:
            scores = np.array([])
        else:
            rel_scores = rec["rel_vecs"] @ rec["user_vector"]
            if use_intervention or rec["start"] is None or rec["start"] < 0:
                pop_scores = np.full(
                    rel_scores.shape, intervention_value, dtype=np.float32
                )
            else:
                pop_scores = all_pop_scores[rec["start"] : rec["stop"]]
            scores = rec["eta"] * rel_scores + (1.0 - rec["eta"]) * pop_scores
        group_labels.append(labels_np)
        group_preds.append(scores)
        progress.update(imp_task, advance=1)

    progress.remove_task(imp_task)

    final = compute_metrics(group_labels, group_preds, metrics_calculator, progress)
    final["_num_impressions"] = len(group_labels)
    return final
