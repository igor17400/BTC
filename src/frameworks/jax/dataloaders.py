"""Simple NumPy-to-JAX data loading utilities.

These dataloaders wrap NumPy arrays and yield JAX arrays in batches,
suitable for use with the Flax NNX training loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Generic batch helper
# ---------------------------------------------------------------------------


def create_batches(
    data: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> Iterator[dict[str, jnp.ndarray]]:
    """Yield dictionaries of JAX arrays from NumPy input arrays.

    All arrays in *data* must share the same first-axis length.

    Args:
        data: Mapping of feature name to NumPy array.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle sample indices before iterating.
        rng: NumPy random generator for shuffling (uses default if ``None``).

    Yields:
        Dictionary with the same keys, but values are ``jnp.ndarray``
        slices of size ``batch_size`` (the last batch may be smaller).
    """
    if not data:
        return

    first_key = next(iter(data))
    num_samples = len(data[first_key])
    indices = np.arange(num_samples)

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield {k: jnp.asarray(v[batch_idx]) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Training data iterator
# ---------------------------------------------------------------------------


class TrainingBatchIterator:
    """Iterate over training data in shuffled batches.

    This is the JAX-side equivalent of ``TrainingSequence`` in the Keras
    dataloader. Each iteration yields ``(features_dict, labels_array)``
    where both are JAX arrays.

    Args:
        features: Dictionary of NumPy feature arrays.
        labels: NumPy label array.
        batch_size: Samples per batch.
        shuffle: Shuffle at the start of each pass.
        seed: Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        features: dict[str, np.ndarray],
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.num_samples = len(labels)

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self) -> Iterator[tuple[dict[str, jnp.ndarray], jnp.ndarray]]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_features = {
                k: jnp.asarray(v[batch_idx]) for k, v in self.features.items()
            }
            batch_labels = jnp.asarray(self.labels[batch_idx])
            yield batch_features, batch_labels


def create_train_dataloader(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> TrainingBatchIterator:
    """Create a training dataloader from numpy feature arrays.

    Isomorphic with ``create_train_dataloader`` in Keras and PyTorch.

    Args:
        features: Dict of feature name -> numpy array.
        labels: Numpy label array.
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        seed: Random seed for reproducible shuffling.

    Returns:
        TrainingBatchIterator instance.
    """
    return TrainingBatchIterator(
        features=features,
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# News batch dataloader (for precomputing news vectors)
# ---------------------------------------------------------------------------


class NewsBatchDataloader:
    """Batch-iterate over news articles for precomputing embeddings.

    Yields dictionaries with ``"news_id"`` and ``"news_features"``
    (JAX array).
    """

    def __init__(
        self,
        news_ids: np.ndarray,
        news_tokens: np.ndarray,
        news_abstract_tokens: np.ndarray | None = None,
        news_category_indices: np.ndarray | None = None,
        news_subcategory_indices: np.ndarray | None = None,
        batch_size: int = 1024,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        news_entity_indices: np.ndarray | None = None,
    ):
        self.news_ids = news_ids
        self.batch_size = batch_size
        self.num_news = len(news_ids)

        # Build the list of arrays to concatenate along the last axis
        parts: list[np.ndarray] = []
        if process_title:
            parts.append(np.asarray(news_tokens))
        if process_abstract and news_abstract_tokens is not None:
            parts.append(np.asarray(news_abstract_tokens))
        if news_entity_indices is not None:
            parts.append(np.asarray(news_entity_indices))
        if process_category and news_category_indices is not None:
            cat = np.asarray(news_category_indices)
            if cat.ndim == 1:
                cat = cat[:, None]
            parts.append(cat)
        if process_subcategory and news_subcategory_indices is not None:
            subcat = np.asarray(news_subcategory_indices)
            if subcat.ndim == 1:
                subcat = subcat[:, None]
            parts.append(subcat)

        self.features = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(0, self.num_news, self.batch_size):
            end = min(i + self.batch_size, self.num_news)
            yield {
                "news_id": self.news_ids[i:end],
                "news_features": jnp.asarray(self.features[i:end]),
            }

    def __len__(self) -> int:
        return self.num_news


# ---------------------------------------------------------------------------
# User history batch dataloader (for precomputing user vectors)
# ---------------------------------------------------------------------------


class UserHistoryBatchDataloader:
    """Batch-iterate over user histories for precomputing user vectors.

    Yields ``(impression_ids, user_ids_or_None, features_jax_array)``.
    """

    def __init__(
        self,
        history_tokens: np.ndarray,
        impression_ids: np.ndarray,
        history_abstract_tokens: np.ndarray | None = None,
        history_category: np.ndarray | None = None,
        history_subcategory: np.ndarray | None = None,
        user_ids: np.ndarray | None = None,
        batch_size: int = 32,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        history_entity_indices: np.ndarray | None = None,
    ):
        self.impression_ids = np.asarray(impression_ids)
        self.user_ids = np.asarray(user_ids) if user_ids is not None else None
        self.batch_size = batch_size
        self.num_users = len(impression_ids)

        parts: list[np.ndarray] = []
        if process_title:
            parts.append(np.asarray(history_tokens))
        if process_abstract and history_abstract_tokens is not None:
            cat_arr = np.asarray(history_abstract_tokens)
            parts.append(cat_arr)
        if history_entity_indices is not None:
            parts.append(np.asarray(history_entity_indices))
        if process_category and history_category is not None:
            cat_arr = np.asarray(history_category)
            if cat_arr.ndim == 2:
                cat_arr = cat_arr[:, :, None]
            parts.append(cat_arr)
        if process_subcategory and history_subcategory is not None:
            subcat_arr = np.asarray(history_subcategory)
            if subcat_arr.ndim == 2:
                subcat_arr = subcat_arr[:, :, None]
            parts.append(subcat_arr)

        self.features = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

    def __iter__(
        self,
    ) -> Iterator[tuple[np.ndarray, jnp.ndarray | None, jnp.ndarray]]:
        for i in range(0, self.num_users, self.batch_size):
            end = min(i + self.batch_size, self.num_users)
            batch_ids = self.impression_ids[i:end]
            batch_user_ids = (
                jnp.asarray(self.user_ids[i:end]) if self.user_ids is not None else None
            )
            batch_features = jnp.asarray(self.features[i:end])
            yield batch_ids, batch_user_ids, batch_features

    def __len__(self) -> int:
        return self.num_users


# ---------------------------------------------------------------------------
# Impression iterator (for evaluation scoring)
# ---------------------------------------------------------------------------


class ImpressionIterator:
    """Iterate over impressions one at a time for evaluation.

    Each impression may have a variable number of candidates, so they
    cannot be easily batched.

    Yields: ``(features, labels, impression_id, candidate_ids)``
    where features and labels are JAX arrays.
    """

    def __init__(
        self,
        impression_tokens: Any,
        labels: Any,
        impression_ids: Any,
        candidate_ids: Any,
        impression_abstract_tokens: Any = None,
        impression_category: Any = None,
        impression_subcategory: Any = None,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ):
        self.impression_tokens = impression_tokens
        self.impression_abstract_tokens = impression_abstract_tokens
        self.impression_category = impression_category
        self.impression_subcategory = impression_subcategory
        self.labels = labels
        self.impression_ids = impression_ids
        self.candidate_ids = candidate_ids
        self.num_impressions = len(labels)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self):
        for idx in range(self.num_impressions):
            parts = []

            if self.process_title:
                parts.append(jnp.asarray(self.impression_tokens[idx], dtype=jnp.int32))
            if self.process_abstract and self.impression_abstract_tokens is not None:
                parts.append(
                    jnp.asarray(self.impression_abstract_tokens[idx], dtype=jnp.int32)
                )
            if self.process_category and self.impression_category is not None:
                cat = jnp.asarray(self.impression_category[idx], dtype=jnp.int32)
                if cat.ndim == 1:
                    cat = jnp.expand_dims(cat, axis=1)
                parts.append(cat)
            if self.process_subcategory and self.impression_subcategory is not None:
                subcat = jnp.asarray(self.impression_subcategory[idx], dtype=jnp.int32)
                if subcat.ndim == 1:
                    subcat = jnp.expand_dims(subcat, axis=1)
                parts.append(subcat)

            if len(parts) > 1:
                features = jnp.concatenate(parts, axis=1)
            else:
                features = parts[0]

            labels_arr = jnp.asarray(self.labels[idx], dtype=jnp.float32)
            imp_id = self.impression_ids[idx]
            cand_ids = self.candidate_ids[idx] if idx < len(self.candidate_ids) else []

            yield features, labels_arr, imp_id, cand_ids

    def __len__(self) -> int:
        return self.num_impressions
