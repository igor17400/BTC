"""PyTorch dataloaders for news recommendation.

Ported from src/datasets/dataloader.py (Keras version).
Wraps numpy arrays from the core dataset and yields torch tensors.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------------
# Training dataset / dataloader
# ------------------------------------------------------------------

class NewsRecommenderDataset(Dataset):
    """Wraps numpy feature arrays as a torch Dataset.

    Each sample is a ``(features_dict, label)`` pair where every value
    in the features dict is a numpy array slice converted to a tensor
    on the fly.
    """

    def __init__(self, features: Dict[str, np.ndarray], labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.num_samples = len(labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sample_features = {
            k: torch.tensor(v[idx], dtype=torch.long if v.dtype in (np.int32, np.int64) else torch.float32)
            for k, v in self.features.items()
        }
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_features, label


def create_train_dataloader(
    features: Dict[str, np.ndarray],
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
        device: Optional[torch.device] = None,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ):
        self.news_ids = news_ids
        self.news_tokens = np.asarray(news_tokens)
        self.news_abstract_tokens = np.asarray(news_abstract_tokens)
        self.news_category_indices = np.asarray(news_category_indices)
        self.news_subcategory_indices = np.asarray(news_subcategory_indices)
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_news = len(news_ids)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(0, self.num_news, self.batch_size):
            end = min(i + self.batch_size, self.num_news)
            batch_ids = self.news_ids[i:end]

            parts: List[np.ndarray] = []
            if self.process_title:
                parts.append(self.news_tokens[i:end])
            if self.process_abstract:
                parts.append(self.news_abstract_tokens[i:end])
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
            features_tensor = torch.tensor(features_np, dtype=torch.long, device=self.device)

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
        device: Optional[torch.device] = None,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
    ):
        self.history_tokens = np.asarray(history_tokens)
        self.history_abstract_tokens = np.asarray(history_abstract_tokens)
        self.history_category = np.asarray(history_category)
        self.history_subcategory = np.asarray(history_subcategory)
        self.impression_ids = np.asarray(impression_ids)
        self.user_ids = np.asarray(user_ids) if user_ids is not None else None
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.num_users = len(impression_ids)

        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[Tuple[Any, Optional[torch.Tensor], torch.Tensor]]:
        for i in range(0, self.num_users, self.batch_size):
            end = min(i + self.batch_size, self.num_users)

            batch_imp_ids = self.impression_ids[i:end]

            batch_user_ids = None
            if self.user_ids is not None:
                batch_user_ids = torch.tensor(
                    self.user_ids[i:end], dtype=torch.long, device=self.device
                )

            parts: List[np.ndarray] = []
            if self.process_title:
                parts.append(self.history_tokens[i:end])
            if self.process_abstract:
                parts.append(self.history_abstract_tokens[i:end])
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
        device: Optional[torch.device] = None,
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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int, Any]]:
        for idx in range(self.num_impressions):
            parts: List[np.ndarray] = []

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
