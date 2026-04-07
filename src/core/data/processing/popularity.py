"""Popularity feature computation: aggregate CTR, publish times, and time-bucketed CTR.

Standalone pure functions used by datasets that need popularity-aware features
(e.g. PP-Rec). No dependencies on dataset class hierarchies — operates on a
behaviors DataFrame and a news ID lookup.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute aggregate CTR, publish times, and time-bucketed CTR.

    Reads ``behaviors_df`` (with columns ``time`` and ``impressions``) and
    derives three quantities for every news article:

    1. **Aggregate CTR** ``[num_news]`` — lifetime click rate.
    2. **Publish time** ``[num_news]`` — earliest impression timestamp where
       the article appears (proxy for true publish time).
    3. **Bucketed CTR** ``[num_news, max_buckets]`` — CTR at each 2-hour
       window since publish time.

    Args:
        behaviors_df: Raw behaviors DataFrame with ``time`` and ``impressions``
            columns. ``time`` will be parsed if not already a datetime.
        news_str_to_idx: Mapping ``"N12345" -> int row index``.
        num_news: Total number of news articles (size of the output arrays).
        bucket_hours: Width of each recency bucket in hours (default 2).
        max_buckets: Number of recency buckets (default 1500, matches
            ``recency_embedding_bins`` in PPRecConfig).

    Returns:
        Tuple of ``(news_ctr, news_publish_time, news_ctr_bucketed)``.
        ``news_ctr`` and ``news_publish_time`` are 1D arrays of length
        ``num_news``. ``news_ctr_bucketed`` is shape ``(num_news, max_buckets)``.
    """
    # Ensure timestamps are parsed (no-op if already datetime)
    if not pd.api.types.is_datetime64_any_dtype(behaviors_df["time"]):
        behaviors_df = behaviors_df.copy()
        behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])

    timestamps = behaviors_df["time"].to_numpy()
    impressions = behaviors_df["impressions"].to_numpy()

    # --- First pass: earliest impression timestamp per news (publish proxy) ---
    publish_times: dict[int, pd.Timestamp] = {}
    for ts, impressions_str in zip(timestamps, impressions):
        if pd.isna(impressions_str):
            continue
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

    publish_time_arr = np.array(
        [publish_times.get(i, pd.NaT) for i in range(num_news)]
    )

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
            pub = publish_times.get(idx)
            if pub is not None:
                delta = pd.Timestamp(ts) - pd.Timestamp(pub)
                delta_hours = delta.total_seconds() / 3600.0
                bucket = int(delta_hours / bucket_hours)
                if 0 <= bucket < max_buckets:
                    bucket_imps[idx, bucket] += 1.0
                    if clicked:
                        bucket_clicks[idx, bucket] += 1.0

    news_ctr = click_counts / (impression_counts + 0.01)
    news_ctr_bucketed = bucket_clicks / (bucket_imps + 0.01)

    active_news = int((impression_counts > 0).sum())
    max_observed_bucket = (
        int((bucket_imps.sum(axis=0) > 0).nonzero()[0].max() + 1)
        if bucket_imps.sum() > 0 else 0
    )
    logger.info(
        f"Computed CTR for {active_news} news articles "
        f"(mean CTR={news_ctr[impression_counts > 0].mean():.4f}). "
        f"Time-bucketed CTR uses {max_observed_bucket} active "
        f"{bucket_hours}-hour buckets (of {max_buckets} max)."
    )

    return news_ctr, publish_time_arr, news_ctr_bucketed


def save_popularity_cache(
    cache_dir: Path,
    news_ctr: np.ndarray,
    publish_time_arr: np.ndarray,
    news_ctr_bucketed: np.ndarray,
    news_str_to_idx: dict[str, int],
) -> None:
    """Persist popularity arrays + a human-readable publish-time JSON dict.

    Args:
        cache_dir: Directory to write cache files.
        news_ctr: 1D array of aggregate CTR.
        publish_time_arr: 1D array of publish timestamps (np.datetime64).
        news_ctr_bucketed: 2D array of time-bucketed CTR.
        news_str_to_idx: Mapping used to write the JSON dict
            ``{news_id_str: ISO timestamp}``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "news_ctr.npy", news_ctr)
    np.save(cache_dir / "news_ctr_bucketed.npy", news_ctr_bucketed)
    with open(cache_dir / "news_publish_time.pkl", "wb") as f:
        pickle.dump(publish_time_arr, f)

    # Human-readable publish times for inspection / debugging
    publish_dict: dict[str, str] = {}
    for nid_str, i in news_str_to_idx.items():
        ts = publish_time_arr[i]
        if not pd.isna(ts):
            publish_dict[nid_str] = pd.Timestamp(ts).isoformat()
    with open(cache_dir / "news_publish_time.json", "w") as f:
        json.dump(publish_dict, f, indent=2)
    logger.info(
        f"Saved popularity cache: {len(publish_dict)} publish times "
        f"in news_publish_time.json"
    )


def load_popularity_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached popularity arrays if present, else return ``None``."""
    ctr_path = cache_dir / "news_ctr.npy"
    bucketed_path = cache_dir / "news_ctr_bucketed.npy"
    publish_path = cache_dir / "news_publish_time.pkl"
    if not (ctr_path.exists() and bucketed_path.exists() and publish_path.exists()):
        return None
    news_ctr = np.load(ctr_path)
    news_ctr_bucketed = np.load(bucketed_path)
    with open(publish_path, "rb") as f:
        publish_time_arr = pickle.load(f)
    return news_ctr, publish_time_arr, news_ctr_bucketed
