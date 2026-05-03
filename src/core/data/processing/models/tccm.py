"""TCCM-specific preprocessing.

TCCM (Time and Content-Aware Causal Model, CIKM 2023) replaces PP-Rec's
content-based bias news encoder with a popularity encoder driven by
**per-token bucketed CTR** (one CTR value per word and per entity in the
candidate news), plus a reciprocal-power timeliness module.

This module computes the two extra inputs the TCCM popularity encoder
needs at training/inference time:

* ``cand_pop_buckets`` — per-candidate concatenation of [word-CTR-buckets,
  entity-CTR-buckets] with shape ``(B, C, max_title_length + max_entities)``,
  int32 in ``[0, pop_token_embedding_bins - 1]``.
* ``cand_news_exist_time`` — per-candidate age-since-publish in hours,
  shape ``(B, C)``, int32 in ``[0, timeliness_embedding_bins - 1]``.

The CTR is computed with the same wall-clock bucketing as PP-Rec
(``bucket_hours`` units since the dataset start), so events from the
*previous* bucket — which strictly precede the current impression — are
used. This matches the paper's "near real-time CTR" definition while
keeping the lookup causal.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hour_bucket(ts: pd.Timestamp, dataset_start: pd.Timestamp, bucket_hours: int) -> int:
    delta_h = (pd.Timestamp(ts) - pd.Timestamp(dataset_start)).total_seconds() / 3600.0
    return int(delta_h // bucket_hours)


def compute_token_ctr_tables(
    behaviors_df: pd.DataFrame,
    *,
    news_str_to_idx: dict[str, int],
    news_title_tokens: np.ndarray,
    news_entity_indices: np.ndarray,
    vocab_size: int,
    entity_vocab_size: int,
    bucket_hours: int = 1,
    num_buckets: int | None = None,
    dataset_start: pd.Timestamp | None = None,
    bins: int = 200,
    smoothing: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp, int]:
    """Build time-bucketed CTR tables per word and per entity.

    For each impression at wall-clock bucket ``b``:
      * every token (word index in ``news_title_tokens[news_idx]``) and
        every entity (entry of ``news_entity_indices[news_idx]``) of every
        clicked news has its click count incremented at ``ctr_table[b, t]``;
      * every token / entity of every displayed news has its impression
        count incremented.

    CTR floats are then discretized to int via
    ``ceil((clicks / (impressions + smoothing)) * bins)`` clipped to
    ``[0, bins - 1]`` so they can index a single shared ``Embedding(bins, dim)``.

    Args:
        behaviors_df: Raw behaviours DataFrame with at least ``time`` and
            ``impressions`` columns. ``impressions`` is the standard
            ``"NID-1 NID-0 ..."`` MIND format.
        news_str_to_idx: ``"N12345" -> int row index in news arrays``.
        news_title_tokens: ``(num_news, max_title_length)`` int32 array.
        news_entity_indices: ``(num_news, max_entities)`` int32 array.
        vocab_size: Word-vocabulary size (inclusive of padding 0).
        entity_vocab_size: Entity-vocabulary size (inclusive of padding 0).
        bucket_hours: Width of one wall-clock bucket in hours.
        num_buckets: Number of buckets to allocate. If ``None``, sized to
            cover the dataset span exactly.
        dataset_start: Wall-clock reference for bucket 0. Defaults to
            ``min(time)``.
        bins: Bucket count for CTR discretisation (paper uses 200).
        smoothing: Laplace smoothing constant for the CTR ratio.

    Returns:
        ``(word_pop, entity_pop, dataset_start, num_buckets)`` where the
        two arrays are ``(num_buckets, vocab_size)`` and
        ``(num_buckets, entity_vocab_size)`` respectively, dtype ``int32``.
    """
    df = behaviors_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    if dataset_start is None:
        dataset_start = pd.Timestamp(df["time"].min())
    else:
        dataset_start = pd.Timestamp(dataset_start)

    if num_buckets is None:
        max_delta = (df["time"].max() - dataset_start).total_seconds() / 3600.0
        num_buckets = int(np.ceil(max_delta / bucket_hours)) + 1

    word_clicks = np.zeros((num_buckets, vocab_size), dtype=np.float32)
    word_imps = np.zeros((num_buckets, vocab_size), dtype=np.float32)
    ent_clicks = np.zeros((num_buckets, entity_vocab_size), dtype=np.float32)
    ent_imps = np.zeros((num_buckets, entity_vocab_size), dtype=np.float32)

    timestamps = df["time"].to_numpy()
    impressions = df["impressions"].to_numpy()

    for ts, impr_str in zip(timestamps, impressions):
        if pd.isna(impr_str):
            continue
        b = _hour_bucket(ts, dataset_start, bucket_hours)
        if b < 0 or b >= num_buckets:
            continue
        for item in str(impr_str).split():
            parts = item.split("-")
            if len(parts) < 2:
                continue
            nid_str, label = parts[0], parts[1]
            n_idx = news_str_to_idx.get(nid_str)
            if n_idx is None:
                continue
            clicked = label == "1"
            tokens = news_title_tokens[n_idx]
            ents = news_entity_indices[n_idx]
            # Use np.add.at to count each unique token/entity once per news.
            np.add.at(word_imps[b], tokens, 1.0)
            np.add.at(ent_imps[b], ents, 1.0)
            if clicked:
                np.add.at(word_clicks[b], tokens, 1.0)
                np.add.at(ent_clicks[b], ents, 1.0)

    word_ctr = word_clicks / (word_imps + smoothing)
    ent_ctr = ent_clicks / (ent_imps + smoothing)
    word_ctr[:, 0] = 0.0
    ent_ctr[:, 0] = 0.0

    word_pop = np.minimum(np.ceil(word_ctr * bins), bins - 1).astype(np.int32)
    entity_pop = np.minimum(np.ceil(ent_ctr * bins), bins - 1).astype(np.int32)

    logger.info(
        "TCCM token-CTR tables: %d buckets x (%d words / %d entities), "
        "mean nonzero word_ctr=%.4f, entity_ctr=%.4f",
        num_buckets,
        vocab_size,
        entity_vocab_size,
        float(word_ctr[word_imps > 0].mean()) if (word_imps > 0).any() else 0.0,
        float(ent_ctr[ent_imps > 0].mean()) if (ent_imps > 0).any() else 0.0,
    )
    return word_pop, entity_pop, dataset_start, num_buckets


def cache_token_ctr_tables(
    cache_dir: Path,
    *,
    word_pop: np.ndarray,
    entity_pop: np.ndarray,
    dataset_start: pd.Timestamp,
    bucket_hours: int,
    num_buckets: int,
    bins: int,
) -> None:
    """Persist the TCCM token-CTR tables under ``cache_dir/tccm/``."""
    cache_dir = cache_dir / "tccm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "word_pop.npy", word_pop)
    np.save(cache_dir / "entity_pop.npy", entity_pop)
    with open(cache_dir / "meta.pkl", "wb") as f:
        pickle.dump(
            {
                "dataset_start": pd.Timestamp(dataset_start),
                "bucket_hours": int(bucket_hours),
                "num_buckets": int(num_buckets),
                "bins": int(bins),
            },
            f,
        )


def load_token_ctr_tables(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp, int, int, int] | None:
    """Load cached TCCM token-CTR tables if present, else ``None``.

    Returns ``(word_pop, entity_pop, dataset_start, bucket_hours,
    num_buckets, bins)`` or ``None`` if the cache is missing.
    """
    cache_dir = cache_dir / "tccm"
    word_path = cache_dir / "word_pop.npy"
    ent_path = cache_dir / "entity_pop.npy"
    meta_path = cache_dir / "meta.pkl"
    if not (word_path.exists() and ent_path.exists() and meta_path.exists()):
        return None
    word_pop = np.load(word_path)
    entity_pop = np.load(ent_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return (
        word_pop,
        entity_pop,
        pd.Timestamp(meta["dataset_start"]),
        int(meta["bucket_hours"]),
        int(meta["num_buckets"]),
        int(meta["bins"]),
    )


def build_per_token_buckets(
    *,
    candidate_news_ids: np.ndarray,
    impression_times: np.ndarray,
    news_title_tokens: np.ndarray,
    news_entity_indices: np.ndarray,
    word_pop: np.ndarray,
    entity_pop: np.ndarray,
    news_publish_time: np.ndarray,
    dataset_start: pd.Timestamp,
    bucket_hours: int,
    timeliness_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up per-token CTR buckets and per-candidate exist-time per impression.

    Args:
        candidate_news_ids: ``(B, C)`` int array of candidate news indices
            (row indices into the ``news_*`` arrays, *not* the original
            ``"N12345"`` strings).
        impression_times: ``(B,)`` ``datetime64[ns]`` array of impression
            timestamps.
        news_title_tokens: ``(num_news, T)`` int32 word indices.
        news_entity_indices: ``(num_news, E)`` int32 entity indices.
        word_pop / entity_pop: Bucketed-CTR tables produced by
            :func:`compute_token_ctr_tables`.
        news_publish_time: ``(num_news,)`` ``datetime64[ns]`` per-news
            publish times (typically from PP-Rec's popularity cache).
        dataset_start: Wall-clock reference for bucket 0.
        bucket_hours: Width of one CTR bucket.
        timeliness_bins: Upper clamp for the (impression - publish) hour
            count; values >= bins are clipped to ``bins - 1``.

    Returns:
        ``(cand_pop_buckets, cand_news_exist_time)`` of shapes
        ``(B, C, T+E)`` int32 and ``(B, C)`` int32.
    """
    B, C = candidate_news_ids.shape
    T = news_title_tokens.shape[1]
    E = news_entity_indices.shape[1]
    num_buckets = word_pop.shape[0]

    cand_pop_buckets = np.zeros((B, C, T + E), dtype=np.int32)
    cand_news_exist_time = np.zeros((B, C), dtype=np.int32)

    # Convert impression_times to wall-clock buckets in one pass. We
    # apply a **causal -1 shift** so a lookup at impression time t uses
    # the *previous* bucket (events strictly before t). The paper
    # reference omits this shift, but doing so leaks the current
    # impression's events into its own popularity score, which inflates
    # train metrics and hurts test generalization on MIND-small.
    impr_ts = pd.to_datetime(impression_times)
    delta_h = (impr_ts - pd.Timestamp(dataset_start)).total_seconds().to_numpy() / 3600.0
    impr_buckets = np.clip(
        np.floor(delta_h / bucket_hours).astype(np.int64) - 1,
        0,
        num_buckets - 1,
    )
    impr_hours = delta_h.astype(np.float64)

    publish_h = np.where(
        pd.isna(news_publish_time),
        np.nan,
        (
            (
                pd.to_datetime(np.asarray(news_publish_time))
                - pd.Timestamp(dataset_start)
            ).total_seconds()
            / 3600.0
        ),
    )

    for i in range(B):
        b = int(impr_buckets[i])
        cand_ids = candidate_news_ids[i]
        # (C, T) word buckets, (C, E) entity buckets
        title_tok = news_title_tokens[cand_ids]
        ent_idx = news_entity_indices[cand_ids]
        cand_pop_buckets[i, :, :T] = word_pop[b][title_tok]
        cand_pop_buckets[i, :, T:] = entity_pop[b][ent_idx]

        pub_h = publish_h[cand_ids]
        age = impr_hours[i] - pub_h
        # NaN publish time => unknown; default to bin 0 (treated as "fresh").
        age = np.where(np.isnan(age), 0.0, age)
        cand_news_exist_time[i] = np.clip(np.round(age), 0, timeliness_bins - 1).astype(
            np.int32
        )

    return cand_pop_buckets, cand_news_exist_time
