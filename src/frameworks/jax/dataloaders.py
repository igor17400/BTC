"""Simple NumPy-to-JAX data loading utilities.

These dataloaders wrap NumPy arrays and yield JAX arrays in batches,
suitable for use with the Flax NNX training loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

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
    """Batch news IDs for the news-vector precompute stage of eval (JAX).

    Single contract across encoders: yields parsed-int news ids; the
    model's :class:`TextEncoder` owns the per-news text lookup.

    Default (single-view, e.g. NRMS):
        ``news_features`` of shape ``(B,)`` int32.

    Multi-view (NAML, PP-Rec, CAUM, TCCM): pass any of
    ``category_indices``, ``subcategory_indices``, ``entity_indices``
    and ``news_features`` becomes ``(B, k)`` int32 with column order
    ``[news_idx | entities | category | subcategory]``.
    """

    def __init__(
        self,
        news_ids_str: np.ndarray,
        parsed_news_ids: np.ndarray,
        batch_size: int = 1024,
        category_indices: np.ndarray | None = None,
        subcategory_indices: np.ndarray | None = None,
        entity_indices: np.ndarray | None = None,
    ):
        self.news_ids_str = np.asarray(news_ids_str)
        self.parsed_news_ids = np.asarray(parsed_news_ids, dtype=np.int32)
        self.batch_size = batch_size
        self.num_news = len(news_ids_str)

        # Column order: [news_idx | entities | category | subcategory].
        packed = [self.parsed_news_ids[:, None]]
        if entity_indices is not None:
            packed.append(np.asarray(entity_indices, dtype=np.int32))
        if category_indices is not None:
            packed.append(np.asarray(category_indices, dtype=np.int32).reshape(-1, 1))
        if subcategory_indices is not None:
            packed.append(
                np.asarray(subcategory_indices, dtype=np.int32).reshape(-1, 1)
            )
        self._packed = (
            np.concatenate(packed, axis=1) if len(packed) > 1 else self.parsed_news_ids
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(0, self.num_news, self.batch_size):
            end = min(i + self.batch_size, self.num_news)
            yield {
                "news_id": self.news_ids_str[i:end],
                "news_features": jnp.asarray(self._packed[i:end]),
            }

    def __len__(self) -> int:
        return self.num_news


class UserHistoryBatchDataloader:
    """Batch user histories for the user-vector precompute stage (JAX).

    Single contract across encoders: yields parsed-int news ids per
    history slot; the model's TextEncoder owns the per-news text lookup.

    Default (single-view): history shape ``(B, H)`` int32.

    Multi-view: pass any of ``history_category``, ``history_subcategory``,
    ``history_entity`` and history becomes ``(B, H, k)`` with column
    order ``[news_idx | entities | category | subcategory]``.
    """

    def __init__(
        self,
        history_news_ids: np.ndarray,
        impression_ids: np.ndarray,
        user_ids: np.ndarray | None = None,
        batch_size: int = 32,
        history_category: np.ndarray | None = None,
        history_subcategory: np.ndarray | None = None,
        history_entity: np.ndarray | None = None,
    ):
        self.history_news_ids = np.asarray(history_news_ids, dtype=np.int32)
        self.impression_ids = np.asarray(impression_ids)
        self.user_ids = np.asarray(user_ids) if user_ids is not None else None
        self.batch_size = batch_size
        self.num_users = len(impression_ids)

        # Column order: [news_idx | entities | category | subcategory].
        packed = [self.history_news_ids[..., None]]
        if history_entity is not None:
            packed.append(np.asarray(history_entity, dtype=np.int32))
        if history_category is not None:
            packed.append(np.asarray(history_category, dtype=np.int32)[..., None])
        if history_subcategory is not None:
            packed.append(np.asarray(history_subcategory, dtype=np.int32)[..., None])
        self._packed = (
            np.concatenate(packed, axis=-1)
            if len(packed) > 1
            else self.history_news_ids
        )

    def __iter__(
        self,
    ) -> Iterator[tuple[np.ndarray, jnp.ndarray | None, jnp.ndarray]]:
        for i in range(0, self.num_users, self.batch_size):
            end = min(i + self.batch_size, self.num_users)
            yield (
                self.impression_ids[i:end],
                jnp.asarray(self.user_ids[i:end])
                if self.user_ids is not None
                else None,
                jnp.asarray(self._packed[i:end]),
            )

    def __len__(self) -> int:
        return self.num_users


class ImpressionIterator:
    """Iterate impressions one-by-one for evaluation (JAX).

    Single contract across encoders: yields parsed-int news ids for the
    candidates of one impression; the model's TextEncoder handles the
    per-news text lookup.

    Default (single-view, e.g. NRMS):
        ``features`` shape ``(C,)`` int32.

    Multi-view: pass any of ``candidate_category``,
    ``candidate_subcategory``, ``candidate_entity`` (per-impression
    arrays aligned with ``candidate_news_ids``) and ``features``
    becomes ``(C, k)`` with column order
    ``[news_idx | entities | category | subcategory]``.
    """

    def __init__(
        self,
        candidate_news_ids: Any,
        labels: Any,
        impression_ids: Any,
        candidate_ids: Any,
        candidate_category: Any = None,
        candidate_subcategory: Any = None,
        candidate_entity: Any = None,
    ):
        self.candidate_news_ids = candidate_news_ids
        self.labels = labels
        self.impression_ids = impression_ids
        self.candidate_ids = candidate_ids
        self.candidate_category = candidate_category
        self.candidate_subcategory = candidate_subcategory
        self.candidate_entity = candidate_entity
        self._multi_view = (
            candidate_category is not None
            or candidate_subcategory is not None
            or candidate_entity is not None
        )
        self.num_impressions = len(labels)

    def __iter__(self):
        for idx in range(self.num_impressions):
            cand_ids_np = np.asarray(self.candidate_news_ids[idx], dtype=np.int32)
            if self._multi_view:
                # Column order: [news_idx | entities | category | subcategory]
                packed = [cand_ids_np[:, None]]
                if self.candidate_entity is not None:
                    packed.append(
                        np.asarray(self.candidate_entity[idx], dtype=np.int32)
                    )
                if self.candidate_category is not None:
                    packed.append(
                        np.asarray(
                            self.candidate_category[idx], dtype=np.int32
                        ).reshape(-1, 1)
                    )
                if self.candidate_subcategory is not None:
                    packed.append(
                        np.asarray(
                            self.candidate_subcategory[idx], dtype=np.int32
                        ).reshape(-1, 1)
                    )
                features_np = np.concatenate(packed, axis=1)
            else:
                features_np = cand_ids_np
            features = jnp.asarray(features_np)
            labels_arr = jnp.asarray(self.labels[idx], dtype=jnp.float32)
            imp_id = self.impression_ids[idx]
            cand_ids = self.candidate_ids[idx] if idx < len(self.candidate_ids) else []
            yield features, labels_arr, imp_id, cand_ids

    def __len__(self) -> int:
        return self.num_impressions


# ---------------------------------------------------------------------------
# GLORY training dataloader (variable-size subgraphs → JAX arrays)
# ---------------------------------------------------------------------------


# Padding constants for GLORY JAX (fixed shapes → single JIT compilation).
# Profiled on MIND-small with batch_size=32, k_hops=2, num_neighbors=8:
#   per-sample nodes: mean=528, p99=741
#   per-batch total nodes: mean=16.9k, p99=17.8k
#   per-batch total edges: mean=585k, p99=644k
_GLORY_MAX_NODES = 20480
_GLORY_MAX_EDGES = 720896


def _glory_collate_jax(
    samples: list[dict[str, np.ndarray]],
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Concatenate and pad GLORY subgraphs to fixed shapes for JIT.

    Concatenates per-sample subgraphs (same as PyTorch) then pads to
    ``_GLORY_MAX_NODES`` / ``_GLORY_MAX_EDGES`` so every batch has
    identical tensor shapes.  XLA compiles once and reuses the compiled
    kernel for all batches.

    Padded nodes get zero features; padded edges are self-loops on
    node 0 (harmless for scatter-add).  A ``num_real_nodes`` int is
    included so the model can slice or mask if needed.
    """
    sub_xs: list[np.ndarray] = []
    sub_edges: list[np.ndarray] = []
    mappings: list[np.ndarray] = []
    cand_tokens: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    offset = 0
    for s in samples:
        n = s["sub_x"].shape[0]
        sub_xs.append(s["sub_x"])
        if s["sub_edge_index"].size > 0:
            sub_edges.append(s["sub_edge_index"] + offset)
        m = s["mapping_idx"].copy()
        valid = m != -1
        m[valid] = m[valid] + offset
        mappings.append(m)
        cand_tokens.append(s["cand_tokens"])
        labels.append(s["label"])
        offset += n

    # Concatenate real data.
    real_x = np.concatenate(sub_xs, axis=0)  # (N_real, feat)
    real_edges = (
        np.concatenate(sub_edges, axis=1)
        if sub_edges
        else np.zeros((2, 0), dtype=np.int64)
    )  # (2, E_real)
    N_real = real_x.shape[0]
    E_real = real_edges.shape[1]
    feat_dim = real_x.shape[1]

    # Pad nodes.
    if N_real > _GLORY_MAX_NODES:
        # Rare overflow — clip (some subgraphs will index padding,
        # producing slightly noisy gradients for this one batch).
        real_x = real_x[:_GLORY_MAX_NODES]
        N_real = _GLORY_MAX_NODES
    node_pad = _GLORY_MAX_NODES - N_real
    padded_x = (
        np.concatenate(
            [real_x, np.zeros((node_pad, feat_dim), dtype=real_x.dtype)],
            axis=0,
        )
        if node_pad > 0
        else real_x
    )

    # Pad edges — point at the last (isolated) padding node so
    # scatter-add doesn't accumulate messages on real nodes.
    pad_node_idx = _GLORY_MAX_NODES - 1
    if E_real > _GLORY_MAX_EDGES:
        real_edges = real_edges[:, :_GLORY_MAX_EDGES]
        E_real = _GLORY_MAX_EDGES
    edge_pad = _GLORY_MAX_EDGES - E_real
    if edge_pad > 0:
        pad_edges = np.full((2, edge_pad), pad_node_idx, dtype=real_edges.dtype)
        padded_edges = np.concatenate([real_edges, pad_edges], axis=1)
    else:
        padded_edges = real_edges

    batched = {
        "subgraph_x": np.asarray(padded_x, dtype=np.int32),
        "subgraph_edge_index": np.asarray(padded_edges, dtype=np.int64),
        "mapping_idx": np.stack(mappings, axis=0).astype(np.int32),
        "cand_tokens": np.stack(cand_tokens, axis=0).astype(np.int32),
        "num_real_nodes": np.array(N_real, dtype=np.int32),
    }

    # Pass through entity data if present (use_entity=True).
    if "candidate_entity" in samples[0]:
        batched["candidate_entity"] = np.stack(
            [s["candidate_entity"] for s in samples],
            axis=0,
        ).astype(np.int32)
        batched["entity_mask"] = np.stack(
            [s["entity_mask"] for s in samples],
            axis=0,
        ).astype(np.float32)

    labels_out = np.stack(labels, axis=0).astype(np.float32)
    return batched, labels_out


def create_glory_jax_dataloader(
    *,
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """Create a GLORY training dataloader that yields JAX arrays.

    Reuses PyTorch's ``DataLoader`` for multi-process subgraph sampling
    (the dataset ``__getitem__`` returns numpy arrays) and converts
    to fixed-shape JAX arrays in the collate function.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_glory_collate_jax,
        drop_last=True,  # Drop last to keep batch size consistent for JIT
    )
