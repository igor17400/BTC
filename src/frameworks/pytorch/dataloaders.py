"""PyTorch dataloaders for news recommendation.

Ported from src/datasets/dataloader.py (Keras version).
Wraps numpy arrays from the core dataset and yields torch tensors.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.data.processing.models.glory import (
    build_csr_in_adjacency,
    extract_edges_for_subgraph,
    sample_subgraph,
)

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
    """Batch news IDs for the news-vector precompute stage of eval.

    Single contract for both GloVe and PLM encoders: each batch yields
    a ``news_features`` tensor of parsed-int news ids. The model's
    :class:`TextEncoder` owns the per-news text lookup, so this
    dataloader doesn't need to know which encoder is downstream.

    Default (single-view, e.g. NRMS):
        Yields ``news_features`` of shape ``(B,)`` int64.

    Multi-view (NAML, PP-Rec, CAUM, TCCM): pass any of
    ``category_indices``, ``subcategory_indices``, ``entity_indices``
    and ``news_features`` becomes a packed ``(B, k)`` int64 tensor
    with column order ``[news_idx | entities | category | subcategory]``.
    """

    def __init__(
        self,
        news_ids_str: np.ndarray,
        parsed_news_ids: np.ndarray,
        batch_size: int = 1024,
        device: torch.device | None = None,
        category_indices: np.ndarray | None = None,
        subcategory_indices: np.ndarray | None = None,
        entity_indices: np.ndarray | None = None,
    ):
        self.news_ids_str = np.asarray(news_ids_str)
        self.parsed_news_ids = np.asarray(parsed_news_ids, dtype=np.int64)
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_news = len(news_ids_str)

        # Pack column order: [news_idx | entities | category | subcategory].
        # Matches the training-time _build_train_features layout.
        packed = [self.parsed_news_ids[:, None]]
        if entity_indices is not None:
            packed.append(np.asarray(entity_indices, dtype=np.int64))
        if category_indices is not None:
            packed.append(np.asarray(category_indices, dtype=np.int64).reshape(-1, 1))
        if subcategory_indices is not None:
            packed.append(
                np.asarray(subcategory_indices, dtype=np.int64).reshape(-1, 1)
            )
        # Squeeze the trailing dim when single-view (NRMS contract).
        self._packed = (
            np.concatenate(packed, axis=1) if len(packed) > 1 else self.parsed_news_ids
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(0, self.num_news, self.batch_size):
            end = min(i + self.batch_size, self.num_news)
            yield {
                "news_id": self.news_ids_str[i:end],
                "news_features": torch.from_numpy(self._packed[i:end]).to(self.device),
            }

    def __len__(self) -> int:
        return self.num_news


class UserHistoryBatchDataloader:
    """Batch user histories for the user-vector precompute stage of eval.

    Single contract for both GloVe and PLM encoders: each batch yields
    a ``history`` tensor of parsed-int news ids. The model's TextEncoder
    handles the per-news text lookup.

    Default (single-view, e.g. NRMS):
        Yields ``history`` of shape ``(B, H)`` int64.

    Multi-view (NAML, PP-Rec, CAUM, TCCM): pass any of
    ``history_category``, ``history_subcategory``, ``history_entity``
    and ``history`` becomes ``(B, H, k)`` with column order
    ``[news_idx | entities | category | subcategory]``.
    """

    def __init__(
        self,
        history_news_ids: np.ndarray,
        impression_ids: np.ndarray,
        user_ids: np.ndarray | None = None,
        batch_size: int = 32,
        device: torch.device | None = None,
        history_category: np.ndarray | None = None,
        history_subcategory: np.ndarray | None = None,
        history_entity: np.ndarray | None = None,
    ):
        self.history_news_ids = np.asarray(history_news_ids, dtype=np.int64)
        self.impression_ids = np.asarray(impression_ids)
        self.user_ids = np.asarray(user_ids) if user_ids is not None else None
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_users = len(impression_ids)

        # Pack column order: [news_idx | entities | category | subcategory].
        packed = [self.history_news_ids[..., None]]
        if history_entity is not None:
            packed.append(np.asarray(history_entity, dtype=np.int64))
        if history_category is not None:
            packed.append(np.asarray(history_category, dtype=np.int64)[..., None])
        if history_subcategory is not None:
            packed.append(np.asarray(history_subcategory, dtype=np.int64)[..., None])
        self._packed = (
            np.concatenate(packed, axis=-1)
            if len(packed) > 1
            else self.history_news_ids
        )

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor | None, torch.Tensor]]:
        for i in range(0, self.num_users, self.batch_size):
            end = min(i + self.batch_size, self.num_users)
            imp_ids = self.impression_ids[i:end]
            user_ids = (
                torch.from_numpy(self.user_ids[i:end]).long().to(self.device)
                if self.user_ids is not None
                else None
            )
            history = torch.from_numpy(self._packed[i:end]).to(self.device)
            yield imp_ids, user_ids, history

    def __len__(self) -> int:
        return self.num_users


class ImpressionIterator:
    """Iterate impressions one-by-one for evaluation.

    Single contract for both GloVe and PLM encoders: each iteration
    yields parsed-int news ids for the candidates of one impression.
    The model's TextEncoder handles the per-news text lookup.

    Default (single-view, e.g. NRMS):
        Yields ``features`` of shape ``(C,)`` int64.

    Multi-view (NAML, PP-Rec, CAUM, TCCM): pass any of
    ``candidate_category``, ``candidate_subcategory``, ``candidate_entity``
    (per-impression arrays aligned with ``candidate_news_ids``) and
    ``features`` becomes ``(C, k)`` with column order
    ``[news_idx | entities | category | subcategory]``.
    """

    def __init__(
        self,
        candidate_news_ids: Any,  # (N, C) parsed-int news ids (one row per impression)
        labels: Any,
        impression_ids: Any,
        candidate_ids: Any,  # same as candidate_news_ids by row, kept for API compat
        device: torch.device | None = None,
        candidate_category: Any = None,
        candidate_subcategory: Any = None,
        candidate_entity: Any = None,
    ):
        self.candidate_news_ids = candidate_news_ids
        self.labels = labels
        self.impression_ids = impression_ids
        self.candidate_ids = candidate_ids
        self.device = device or torch.device("cpu")
        self.num_impressions = len(labels)
        self.candidate_category = candidate_category
        self.candidate_subcategory = candidate_subcategory
        self.candidate_entity = candidate_entity
        self._multi_view = (
            candidate_category is not None
            or candidate_subcategory is not None
            or candidate_entity is not None
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, Any]]:
        for idx in range(self.num_impressions):
            cand_ids_np = np.asarray(self.candidate_news_ids[idx], dtype=np.int64)
            if self._multi_view:
                # Column order: [news_idx | entities | category | subcategory]
                packed = [cand_ids_np[:, None]]
                if self.candidate_entity is not None:
                    packed.append(
                        np.asarray(self.candidate_entity[idx], dtype=np.int64)
                    )
                if self.candidate_category is not None:
                    packed.append(
                        np.asarray(
                            self.candidate_category[idx], dtype=np.int64
                        ).reshape(-1, 1)
                    )
                if self.candidate_subcategory is not None:
                    packed.append(
                        np.asarray(
                            self.candidate_subcategory[idx], dtype=np.int64
                        ).reshape(-1, 1)
                    )
                features_np = np.concatenate(packed, axis=1)
            else:
                features_np = cand_ids_np
            features = torch.from_numpy(features_np).to(self.device)
            label = torch.tensor(
                np.asarray(self.labels[idx]),
                dtype=torch.float32,
                device=self.device,
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
        hist_ids: np.ndarray,  # (N_samples, H) int — history news IDs
        cand_ids: np.ndarray,  # (N_samples, C) int — candidate news IDs
        news_features: np.ndarray,  # (num_news, feat_dim) int — packed features
        graph_edge_index: np.ndarray,  # (2, E) int64 — full graph edges
        graph_edge_attr: np.ndarray,  # (E,) int64 — full graph edge weights
        neighbor_dict: dict[int, list[int]],
        labels: np.ndarray,  # (N_samples, C) bool/int
        his_size: int,
        k_hops: int,
        num_neighbors: int,
        entity_neighbor_dict: dict[int, list[int]] | None = None,
        entity_size: int = 5,
        entity_neighbors: int = 10,
        title_size: int = 30,
    ):
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

        # Entity neighbor data (optional, for use_entity=True).
        self.entity_neighbor_dict = entity_neighbor_dict
        self.entity_size = int(entity_size)
        self.entity_neighbors = int(entity_neighbors)
        self.title_size = int(title_size)

    def __len__(self) -> int:
        return self.hist_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        hist = self.hist_ids[idx]
        cand = self.cand_ids[idx]
        label = self.labels[idx]

        node_ids, hist_mapping = self._sample_subgraph(
            hist,
            self.neighbor_dict,
            self.k_hops,
            self.num_neighbors,
        )
        sub_edges, _ = self._extract_edges(
            node_ids,
            self.edge_index,
            self.edge_attr,
            csr=self._csr,
            num_nodes=self.num_nodes,
        )
        sub_x = self.news_features[node_ids]
        cand_features = self.news_features[cand]

        # Left-pad mapping to ``his_size`` with -1 sentinels.
        padded_mapping = np.full(self.his_size, -1, dtype=np.int64)
        hist_valid = hist[hist > 0]
        n_valid = min(hist_valid.size, self.his_size)
        if n_valid > 0:
            padded_mapping[-n_valid:] = hist_mapping[-n_valid:]

        result = {
            "sub_x": sub_x,
            "sub_edge_index": sub_edges,
            "mapping_idx": padded_mapping,
            "cand_tokens": cand_features,
            "label": label,
        }

        # Entity neighbor lookup (optional).
        if self.entity_neighbor_dict is not None:
            E = self.entity_size
            EN = self.entity_neighbors
            # Extract origin entity IDs from candidate features.
            origin_entity = cand_features[
                :, self.title_size : self.title_size + E
            ]  # (C, E)
            # Look up neighbors for each entity.
            C = cand_features.shape[0]
            neighbor_entity = np.zeros(
                (C * E, EN),
                dtype=np.int64,
            )  # (C*E, EN)
            for cnt, eid in enumerate(origin_entity.flatten()):
                if eid == 0:
                    continue
                nbrs = self.entity_neighbor_dict.get(int(eid), [])
                valid_len = min(len(nbrs), EN)
                if valid_len > 0:
                    neighbor_entity[cnt, :valid_len] = nbrs[:valid_len]
            neighbor_entity = neighbor_entity.reshape(C, E * EN)  # (C, E*EN)
            entity_mask = (neighbor_entity > 0).astype(np.float32)
            # Concat: [origin (C, E) | neighbors (C, E*EN)]
            result["candidate_entity"] = np.concatenate(
                [origin_entity, neighbor_entity],
                axis=-1,
            ).astype(np.int64)
            result["entity_mask"] = entity_mask

        return result


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
    # Without these keys the candidate-side entity branch never fires
    # at training time and ``GLORYCandidateEncoder.attn_pool`` keeps its
    # random init — which silently breaks test eval where the cached
    # entity vectors get pooled by those random weights.
    if "candidate_entity" in samples[0]:
        batched["candidate_entity"] = torch.from_numpy(
            np.stack([s["candidate_entity"] for s in samples], axis=0)
        ).long()
        batched["entity_mask"] = torch.from_numpy(
            np.stack([s["entity_mask"] for s in samples], axis=0)
        ).float()
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
