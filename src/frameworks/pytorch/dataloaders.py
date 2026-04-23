"""PyTorch dataloaders for news recommendation.

Ported from src/datasets/dataloader.py (Keras version).
Wraps numpy arrays from the core dataset and yields torch tensors.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------
# Training dataset / dataloader
# ------------------------------------------------------------------


class NewsRecommenderDataset(Dataset):
    """Wraps numpy feature arrays as a torch Dataset.

    Each sample is a ``(features_dict, label)`` pair where every value
    in the features dict is a numpy array slice converted to a tensor
    on the fly.
    """

    def __init__(self, features: dict[str, np.ndarray], labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.num_samples = len(labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sample_features = {
            k: torch.tensor(
                v[idx],
                dtype=torch.long if v.dtype in (np.int32, np.int64) else torch.float32,
            )
            for k, v in self.features.items()
        }
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_features, label


def create_train_dataloader(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a training DataLoader from numpy arrays.

    Args:
        features: Dict of feature name -> numpy array.
        labels: Numpy label array.
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        num_workers: Dataloader workers.
        pin_memory: Pin memory for CUDA.

    Returns:
        PyTorch DataLoader.
    """
    dataset = NewsRecommenderDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ------------------------------------------------------------------
# Inference iterators (same interface as Keras, but yield torch tensors)
# ------------------------------------------------------------------


class NewsBatchDataloader:
    """Batch dataloader for precomputing news embeddings.

    Mirrors the Keras ``NewsBatchDataloader`` interface but yields
    torch tensors.
    """

    def __init__(
        self,
        news_ids: np.ndarray,
        news_tokens: np.ndarray,
        news_abstract_tokens: np.ndarray,
        news_category_indices: np.ndarray,
        news_subcategory_indices: np.ndarray,
        batch_size: int = 1024,
        device: torch.device | None = None,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        news_entity_indices: np.ndarray | None = None,
    ):
        self.news_ids = news_ids
        self.news_tokens = np.asarray(news_tokens)
        self.news_abstract_tokens = np.asarray(news_abstract_tokens)
        self.news_category_indices = np.asarray(news_category_indices)
        self.news_subcategory_indices = np.asarray(news_subcategory_indices)
        self.news_entity_indices = (
            np.asarray(news_entity_indices) if news_entity_indices is not None else None
        )
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_news = len(news_ids)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(0, self.num_news, self.batch_size):
            end = min(i + self.batch_size, self.num_news)
            batch_ids = self.news_ids[i:end]

            parts: list[np.ndarray] = []
            if self.process_title:
                parts.append(self.news_tokens[i:end])
            if self.process_abstract:
                parts.append(self.news_abstract_tokens[i:end])
            # Entity indices (between title/abstract and category, matching model input order)
            if self.news_entity_indices is not None:
                parts.append(self.news_entity_indices[i:end])
            if self.process_category:
                cat = self.news_category_indices[i:end]
                if cat.ndim == 1:
                    cat = cat[:, np.newaxis]
                parts.append(cat)
            if self.process_subcategory:
                subcat = self.news_subcategory_indices[i:end]
                if subcat.ndim == 1:
                    subcat = subcat[:, np.newaxis]
                parts.append(subcat)

            features_np = np.concatenate(parts, axis=1)
            features_tensor = torch.tensor(
                features_np, dtype=torch.long, device=self.device
            )

            yield {"news_id": batch_ids, "news_features": features_tensor}

    def __len__(self) -> int:
        return self.num_news


class UserHistoryBatchDataloader:
    """Batch dataloader for precomputing user embeddings."""

    def __init__(
        self,
        history_tokens: Any,
        history_abstract_tokens: Any,
        history_category: Any,
        history_subcategory: Any,
        impression_ids: Any,
        user_ids: Any = None,
        batch_size: int = 32,
        device: torch.device | None = None,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        history_entity_indices: Any = None,
    ):
        self.history_tokens = np.asarray(history_tokens)
        self.history_abstract_tokens = np.asarray(history_abstract_tokens)
        self.history_category = np.asarray(history_category)
        self.history_subcategory = np.asarray(history_subcategory)
        self.history_entity_indices = (
            np.asarray(history_entity_indices)
            if history_entity_indices is not None
            else None
        )
        self.impression_ids = np.asarray(impression_ids)
        self.user_ids = np.asarray(user_ids) if user_ids is not None else None
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_users = len(impression_ids)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor | None, torch.Tensor]]:
        for i in range(0, self.num_users, self.batch_size):
            end = min(i + self.batch_size, self.num_users)

            batch_imp_ids = self.impression_ids[i:end]

            batch_user_ids = None
            if self.user_ids is not None:
                batch_user_ids = torch.tensor(
                    self.user_ids[i:end], dtype=torch.long, device=self.device
                )

            parts: list[np.ndarray] = []
            if self.process_title:
                parts.append(self.history_tokens[i:end])
            if self.process_abstract:
                parts.append(self.history_abstract_tokens[i:end])
            # Entity indices (matching model input order: title -> entity -> category)
            if self.history_entity_indices is not None:
                parts.append(self.history_entity_indices[i:end])
            if self.process_category:
                cat = self.history_category[i:end]
                if cat.ndim == 2:
                    cat = cat[:, :, np.newaxis]
                parts.append(cat)
            if self.process_subcategory:
                subcat = self.history_subcategory[i:end]
                if subcat.ndim == 2:
                    subcat = subcat[:, :, np.newaxis]
                parts.append(subcat)

            if len(parts) > 1:
                features_np = np.concatenate(parts, axis=-1)
            else:
                features_np = parts[0]

            features_tensor = torch.tensor(
                features_np, dtype=torch.long, device=self.device
            )

            yield batch_imp_ids, batch_user_ids, features_tensor

    def __len__(self) -> int:
        return self.num_users


class ImpressionIterator:
    """Iterate impressions one-by-one, yielding torch tensors."""

    def __init__(
        self,
        impression_tokens: Any,
        impression_abstract_tokens: Any,
        impression_category: Any,
        impression_subcategory: Any,
        labels: Any,
        impression_ids: Any,
        candidate_ids: Any,
        device: torch.device | None = None,
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
        self.device = device or torch.device("cpu")
        self.num_impressions = len(labels)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, Any]]:
        for idx in range(self.num_impressions):
            parts: list[np.ndarray] = []

            if self.process_title:
                parts.append(np.asarray(self.impression_tokens[idx]))
            if self.process_abstract:
                parts.append(np.asarray(self.impression_abstract_tokens[idx]))
            if self.process_category:
                cat = np.asarray(self.impression_category[idx])
                if cat.ndim == 1:
                    cat = cat[:, np.newaxis]
                parts.append(cat)
            if self.process_subcategory:
                subcat = np.asarray(self.impression_subcategory[idx])
                if subcat.ndim == 1:
                    subcat = subcat[:, np.newaxis]
                parts.append(subcat)

            if len(parts) > 1:
                features_np = np.concatenate(parts, axis=1)
            else:
                features_np = parts[0]

            features = torch.tensor(features_np, dtype=torch.long, device=self.device)
            label = torch.tensor(
                np.asarray(self.labels[idx]), dtype=torch.float32, device=self.device
            )

            impression_id = self.impression_ids[idx]
            cand_ids = self.candidate_ids[idx] if idx < len(self.candidate_ids) else []

            yield features, label, impression_id, cand_ids

    def __len__(self) -> int:
        return self.num_impressions


# ------------------------------------------------------------------
# GLORY training dataloader (on-the-fly subgraph sampling)
# ------------------------------------------------------------------
#
# GLORY samples a k-hop subgraph per behavior on the fly and concatenates
# the per-sample subgraphs into one batched graph via node-id offsets.
# This matches the reference implementation's
# ``torch_geometric.data.Batch.from_data_list`` without pulling in
# ``torch_geometric`` as a dependency.  Each batch yields a
# ``(features_dict, labels)`` pair compatible with the standard training
# loop.  The features dict contains:
#
# - ``subgraph_x``: ``(total_nodes, feature_dim)`` — stacked node
#   features for all subgraphs in the batch.
# - ``subgraph_edge_index``: ``(2, total_edges)`` — edges with per-sample
#   offsets already applied.
# - ``mapping_idx``: ``(B, his_size)`` — node index of each clicked
#   history slot within ``subgraph_x`` (``-1`` = padding).
# - ``cand_tokens``: ``(B, npratio+1, feature_dim)``.


class GLORYTrainDataset(Dataset):
    """Per-sample subgraph builder for GLORY training.

    ``__getitem__`` returns a dict of numpy arrays describing one
    sample's subgraph; the custom :func:`glory_collate` concatenates
    these into a batched tensor dict.
    """

    def __init__(
        self,
        *,
        hist_ids: np.ndarray,           # (N_samples, H) int — history news IDs
        cand_ids: np.ndarray,           # (N_samples, C) int — candidate news IDs
        news_features: np.ndarray,      # (num_news, feat_dim) int — packed features
        graph_edge_index: np.ndarray,   # (2, E) int64 — full graph edges
        graph_edge_attr: np.ndarray,    # (E,) int64 — full graph edge weights
        neighbor_dict: dict[int, list[int]],
        labels: np.ndarray,             # (N_samples, C) bool/int
        his_size: int,
        k_hops: int,
        num_neighbors: int,
    ):
        from src.core.data.processing.glory import (  # noqa: F401 — lazy import
            build_csr_in_adjacency,
            extract_edges_for_subgraph,
            sample_subgraph,
        )

        self.hist_ids = np.asarray(hist_ids).astype(np.int64)
        self.cand_ids = np.asarray(cand_ids).astype(np.int64)
        self.news_features = np.asarray(news_features).astype(np.int32)
        self.edge_index = np.asarray(graph_edge_index).astype(np.int64)
        self.edge_attr = np.asarray(graph_edge_attr).astype(np.int64)
        self.neighbor_dict = neighbor_dict
        self.labels = np.asarray(labels)
        self.his_size = int(his_size)
        self.k_hops = int(k_hops)
        self.num_neighbors = int(num_neighbors)
        self._sample_subgraph = sample_subgraph
        self._extract_edges = extract_edges_for_subgraph
        # Pre-build CSR incoming-adjacency so per-sample subgraph
        # extraction is O(|subgraph| · avg_in_degree) instead of
        # O(|full_graph_edges|).  Without this the dataloader is the
        # training bottleneck.
        self.num_nodes = int(news_features.shape[0])
        self._csr = build_csr_in_adjacency(self.edge_index, self.num_nodes)

    def __len__(self) -> int:
        return self.hist_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        hist = self.hist_ids[idx]
        cand = self.cand_ids[idx]
        label = self.labels[idx]

        node_ids, hist_mapping = self._sample_subgraph(
            hist, self.neighbor_dict, self.k_hops, self.num_neighbors,
        )
        sub_edges, _ = self._extract_edges(
            node_ids, self.edge_index, self.edge_attr,
            csr=self._csr, num_nodes=self.num_nodes,
        )
        sub_x = self.news_features[node_ids]
        cand_features = self.news_features[cand]

        # Left-pad mapping to ``his_size`` with -1 sentinels.
        padded_mapping = np.full(self.his_size, -1, dtype=np.int64)
        hist_valid = hist[hist > 0]
        n_valid = min(hist_valid.size, self.his_size)
        if n_valid > 0:
            padded_mapping[-n_valid:] = hist_mapping[-n_valid:]

        return {
            "sub_x": sub_x,
            "sub_edge_index": sub_edges,
            "mapping_idx": padded_mapping,
            "cand_tokens": cand_features,
            "label": label,
        }


def glory_collate(
    samples: list[dict[str, np.ndarray]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Concatenate per-sample GLORY subgraphs into a single batched graph.

    Node ids are offset per-sample so edges remain valid, and
    ``mapping_idx`` is offset in the same way so each history slot
    indexes the correct node in the concatenated array.
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

    batched = {
        "subgraph_x": torch.from_numpy(np.concatenate(sub_xs, axis=0)).long(),
        "subgraph_edge_index": (
            torch.from_numpy(np.concatenate(sub_edges, axis=1)).long()
            if sub_edges
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "mapping_idx": torch.from_numpy(np.stack(mappings, axis=0)).long(),
        "cand_tokens": torch.from_numpy(np.stack(cand_tokens, axis=0)).long(),
    }
    labels_t = torch.from_numpy(np.stack(labels, axis=0)).float()
    return batched, labels_t


def create_glory_train_dataloader(
    *,
    dataset: GLORYTrainDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create the training DataLoader for GLORY (variable-size subgraphs)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=glory_collate,
        drop_last=False,
    )
