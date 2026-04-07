"""Popularity feature computation: aggregate CTR, publish times, and time-bucketed CTR.

Standalone pure functions used by datasets that need popularity-aware features
(e.g. PP-Rec). No dependencies on dataset class hierarchies — operates on a
behaviors DataFrame and a news ID lookup.

CTR computation methods
-----------------------

Three CTR computation methods are supported, each producing a 2D table
``news_ctr_bucketed[news_idx, bucket_idx]``:

1. **``aggregate``** — single lifetime CTR per news, broadcast across all
   buckets. No temporal information; baseline.

2. **``age_bucketed``** — bucket index counts the **age** of an article in
   ``bucket_hours`` units relative to its own publish time. Each article has
   a private timeline that starts at bucket 0 when the article is first seen.

   Lookup at prediction time t::

       age = t - publish_time[news]
       bucket = age // bucket_hours        # e.g. 20h / 2h = bucket 10
       ctr   = news_ctr_bucketed[news, bucket]

   Answers: *"How popular was this article at this stage of its life?"*

3. **``wall_clock``** — bucket index counts **absolute wall-clock time** in
   ``bucket_hours`` units relative to a single shared ``dataset_start``. ALL
   articles share the same global timeline. The same wall_bucket index means
   the same wall-clock window for every article.

   Lookup at prediction time t (causal — strictly past)::

       wall = t - dataset_start
       wall_bucket = wall // bucket_hours
       ctr_bucket  = wall_bucket - 1       # use the *previous* bucket
       ctr         = news_ctr_bucketed[news, ctr_bucket]

   Answers: *"How popular was this article during the wall-clock window
   immediately before the prediction?"* — i.e. PP-Rec's "near real-time CTR".

Mathematical relationship
-------------------------

For a single article A at prediction time t::

    age_bucket(A, t)  = wall_bucket(A, t) - publish_offset[A]

where ``publish_offset[A] = (publish_time[A] - dataset_start) / bucket_hours``.
The two methods coincide only for articles published exactly at
``dataset_start``; otherwise they differ by a per-article offset.

Example with ``bucket_hours=2``, dataset starts at day 1, 0:00:

- Article A published at day 1, 8:00 (offset 4)
- Article B published at day 3, 14:00 (offset 31)
- Prediction at day 4, 10:00 (wall_bucket 41)

  ====================== =========== ===========
  Article                age_bucket  wall_bucket
  ====================== =========== ===========
  A (4 buckets old)      37          41
  B (31 buckets old)     10          41
  ====================== =========== ===========

In ``age_bucketed`` mode, A and B are looked up at completely different
indices even though they're queried at the same wall-clock moment. In
``wall_clock`` mode, both are looked up at the same global index 41 (or
40 with the causal -1 shift), so the model receives a *coherent global
"what's hot now" signal* across all articles.

The ``wall_clock`` method matches PP-Rec's approach: it gives a denser
temporal signal because each wall bucket aggregates events from all
articles active during that window, and it's safe to compute over
train+val behaviors because the causal -1 shift ensures the lookup
never includes events at or after the prediction time.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_news_ctr_and_publish_times(
    behaviors_df: pd.DataFrame,
    news_str_to_idx: dict[str, int],
    num_news: int,
    bucket_hours: int = 2,
    max_buckets: int = 1500,
    method: str = "age_bucketed",
    dataset_start: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    """Compute aggregate CTR, publish times, and time-bucketed CTR.

    Reads ``behaviors_df`` (columns: ``time``, ``impressions``, and
    optionally ``history``) and derives three quantities per news article:

    1. **Aggregate CTR** ``[num_news]`` — lifetime click rate.
    2. **Publish time** ``[num_news]`` — earliest timestamp across both
       ``impressions`` AND ``history`` columns. If news N appears in user
       U's history at impression time t, we know N was published at or
       before t, so scanning history tightens the publish-time estimate.
    3. **Bucketed CTR** ``[num_news, max_buckets]`` — CTR per
       (news, time bucket). Bucket semantics depend on ``method`` (see
       module docstring for the full conceptual explanation of methods).
       Click/impression counts always come from the ``impressions`` column;
       ``history`` has no click labels.

    Args:
        behaviors_df: Raw behaviors DataFrame. Required columns: ``time``,
            ``impressions``. Optional: ``history`` (used for publish time).
        news_str_to_idx: Mapping ``"N12345" -> int row index``.
        num_news: Total number of news articles.
        bucket_hours: Width of each time bucket in hours (default 2).
        max_buckets: Number of time buckets (default 1500, matches
            ``recency_embedding_bins`` in PPRecConfig).
        method: ``"aggregate"``, ``"age_bucketed"``, or ``"wall_clock"``.
            See module docstring for the differences.
        dataset_start: Wall-clock reference for bucket 0 (``wall_clock``
            method only). Defaults to ``min(timestamps)`` if ``None``.

    Returns:
        ``(news_ctr, news_publish_time, news_ctr_bucketed, dataset_start)``.
        ``dataset_start`` is ``pd.NaT`` for non-wall-clock methods.
    """
    if method not in ("aggregate", "age_bucketed", "wall_clock"):
        raise ValueError(
            f"Unknown popularity_ctr_method: {method!r}. "
            f"Expected one of: aggregate, age_bucketed, wall_clock"
        )
    # Ensure timestamps are parsed (no-op if already datetime)
    if not pd.api.types.is_datetime64_any_dtype(behaviors_df["time"]):
        behaviors_df = behaviors_df.copy()
        behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])

    timestamps = behaviors_df["time"].to_numpy()
    impressions = behaviors_df["impressions"].to_numpy()
    has_history = "history" in behaviors_df.columns
    histories = behaviors_df["history"].to_numpy() if has_history else None

    # --- First pass: earliest timestamp per news (publish proxy) ---
    # Scan BOTH the impressions column and the history column. If news N
    # appears in user U's history at impression time t, N was published
    # at or before t — so this tightens the publish time estimate.
    publish_times: dict[int, pd.Timestamp] = {}
    for i, (ts, impressions_str) in enumerate(zip(timestamps, impressions)):
        # Impressions column: "N123-1 N456-0 ..."
        if not pd.isna(impressions_str):
            for item in str(impressions_str).split():
                parts = item.split("-")
                if len(parts) < 2:
                    continue
                idx = news_str_to_idx.get(parts[0])
                if idx is None:
                    continue
                prev = publish_times.get(idx)
                if prev is None or ts < prev:
                    publish_times[idx] = ts

        # History column: "N123 N456 N789 ..." (no labels)
        if has_history and histories is not None:
            history_str = histories[i]
            if pd.isna(history_str):
                continue
            for nid_str in str(history_str).split():
                idx = news_str_to_idx.get(nid_str)
                if idx is None:
                    continue
                prev = publish_times.get(idx)
                if prev is None or ts < prev:
                    publish_times[idx] = ts

    publish_time_arr = np.array([publish_times.get(i, pd.NaT) for i in range(num_news)])

    # Resolve dataset_start for wall_clock method.
    if method == "wall_clock":
        if dataset_start is None:
            dataset_start = pd.Timestamp(timestamps.min())
        else:
            dataset_start = pd.Timestamp(dataset_start)
    else:
        dataset_start = pd.NaT

    # --- Second pass: aggregate counts + bucketed counts ---
    click_counts = np.zeros(num_news, dtype=np.float32)
    impression_counts = np.zeros(num_news, dtype=np.float32)
    bucket_clicks = np.zeros((num_news, max_buckets), dtype=np.float32)
    bucket_imps = np.zeros((num_news, max_buckets), dtype=np.float32)

    for ts, impressions_str in zip(timestamps, impressions):
        if pd.isna(impressions_str):
            continue
        for item in str(impressions_str).split():
            parts = item.split("-")
            if len(parts) < 2:
                continue
            nid_str, label = parts[0], parts[1]
            idx = news_str_to_idx.get(nid_str)
            if idx is None:
                continue
            impression_counts[idx] += 1.0
            clicked = label == "1"
            if clicked:
                click_counts[idx] += 1.0

            # Bucket index depends on the method.
            if method == "age_bucketed":
                pub = publish_times.get(idx)
                if pub is None:
                    continue
                delta = pd.Timestamp(ts) - pd.Timestamp(pub)
            elif method == "wall_clock":
                delta = pd.Timestamp(ts) - dataset_start
            else:  # aggregate — skip bucketed accounting
                continue

            delta_hours = delta.total_seconds() / 3600.0
            bucket = int(delta_hours / bucket_hours)
            if 0 <= bucket < max_buckets:
                bucket_imps[idx, bucket] += 1.0
                if clicked:
                    bucket_clicks[idx, bucket] += 1.0

    news_ctr = click_counts / (impression_counts + 0.01)

    if method == "aggregate":
        # Broadcast aggregate CTR to all buckets so the model interface is uniform.
        news_ctr_bucketed = np.tile(news_ctr[:, None], (1, max_buckets))
    else:
        news_ctr_bucketed = bucket_clicks / (bucket_imps + 0.01)

    active_news = int((impression_counts > 0).sum())
    max_observed_bucket = (
        int((bucket_imps.sum(axis=0) > 0).nonzero()[0].max() + 1)
        if bucket_imps.sum() > 0
        else 0
    )
    logger.info(
        f"Computed CTR ({method}) for {active_news} news articles "
        f"(mean CTR={news_ctr[impression_counts > 0].mean():.4f}). "
        f"Bucket dim uses {max_observed_bucket} active "
        f"{bucket_hours}-hour buckets (of {max_buckets} max)."
    )
    if method == "wall_clock":
        logger.info(f"Wall-clock dataset_start = {dataset_start}")

    return news_ctr, publish_time_arr, news_ctr_bucketed, dataset_start


def save_popularity_cache(
    cache_dir: Path,
    news_ctr: np.ndarray,
    publish_time_arr: np.ndarray,
    news_ctr_bucketed: np.ndarray,
    news_str_to_idx: dict[str, int],
    method: str = "age_bucketed",
    dataset_start: pd.Timestamp | None = None,
) -> None:
    """Persist popularity arrays + a human-readable publish-time JSON dict.

    Cache filenames are suffixed with ``method`` so different CTR computation
    methods can coexist on disk for A/B comparison.

    Args:
        cache_dir: Directory to write cache files.
        news_ctr: 1D array of aggregate CTR.
        publish_time_arr: 1D array of publish timestamps (np.datetime64).
        news_ctr_bucketed: 2D array of time-bucketed CTR.
        news_str_to_idx: Mapping used to write the JSON dict
            ``{news_id_str: ISO timestamp}``.
        method: CTR computation method (used in filenames).
        dataset_start: Reference time for ``wall_clock`` method.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = method
    np.save(cache_dir / f"news_ctr_{suffix}.npy", news_ctr)
    np.save(cache_dir / f"news_ctr_bucketed_{suffix}.npy", news_ctr_bucketed)
    with open(cache_dir / f"news_publish_time_{suffix}.pkl", "wb") as f:
        pickle.dump(publish_time_arr, f)
    if dataset_start is not None and not pd.isna(dataset_start):
        with open(cache_dir / f"news_dataset_start_{suffix}.pkl", "wb") as f:
            pickle.dump(pd.Timestamp(dataset_start), f)

    # Human-readable publish times for inspection / debugging
    publish_dict: dict[str, str] = {}
    for nid_str, i in news_str_to_idx.items():
        ts = publish_time_arr[i]
        if not pd.isna(ts):
            publish_dict[nid_str] = pd.Timestamp(ts).isoformat()
    with open(cache_dir / f"news_publish_time_{suffix}.json", "w") as f:
        json.dump(publish_dict, f, indent=2)
    logger.info(
        f"Saved popularity cache ({method}): {len(publish_dict)} publish times "
        f"in news_publish_time_{suffix}.json"
    )


def load_popularity_cache(
    cache_dir: Path,
    method: str = "age_bucketed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, "pd.Timestamp | None"] | None:
    """Load cached popularity arrays if present, else return ``None``.

    Returns ``(news_ctr, publish_time_arr, news_ctr_bucketed, dataset_start)``.
    ``dataset_start`` is ``None`` for non-wall-clock methods.
    """
    suffix = method
    ctr_path = cache_dir / f"news_ctr_{suffix}.npy"
    bucketed_path = cache_dir / f"news_ctr_bucketed_{suffix}.npy"
    publish_path = cache_dir / f"news_publish_time_{suffix}.pkl"
    if not (ctr_path.exists() and bucketed_path.exists() and publish_path.exists()):
        return None
    news_ctr = np.load(ctr_path)
    news_ctr_bucketed = np.load(bucketed_path)
    with open(publish_path, "rb") as f:
        publish_time_arr = pickle.load(f)
    dataset_start_path = cache_dir / f"news_dataset_start_{suffix}.pkl"
    dataset_start: "pd.Timestamp | None" = None
    if dataset_start_path.exists():
        with open(dataset_start_path, "rb") as f:
            dataset_start = pickle.load(f)
    return news_ctr, publish_time_arr, news_ctr_bucketed, dataset_start
