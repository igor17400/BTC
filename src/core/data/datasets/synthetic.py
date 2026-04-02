"""Synthetic dataset for smoke-testing the full training pipeline.

Generates tiny in-memory data (no downloads, no files) that exercises
every code path: title, abstract, category, subcategory, user_id.
Runs in <5 seconds per model.
"""

from typing import Any

import numpy as np


class SyntheticDataset:
    """Minimal dataset that satisfies the NewsReX dataset interface contract.

    All data is randomly generated in ``__init__`` — no I/O required.
    """

    VOCAB_SIZE = 500
    NUM_CATEGORIES = 10
    NUM_SUBCATEGORIES = 20
    EMBEDDING_SIZE = 300
    NUM_NEWS = 100
    NUM_USERS = 30
    NUM_TRAIN = 80
    NUM_VAL = 20

    def __init__(
        self,
        max_title_length: int = 32,
        max_abstract_length: int = 50,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        embedding_size: int = 300,
        process_title: bool = True,
        process_abstract: bool = False,
        process_category: bool = False,
        process_subcategory: bool = False,
        process_user_id: bool = False,
        seed: int = 42,
        # Accept and ignore extra kwargs from Hydra
        **kwargs,
    ):
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self._embedding_size = embedding_size
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory
        self.process_user_id = process_user_id

        self._rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        rng = self._rng
        V = self.VOCAB_SIZE
        C = self.NUM_CATEGORIES
        SC = self.NUM_SUBCATEGORIES
        N = self.NUM_NEWS

        # --- processed_news ---
        self._processed_news = {
            "vocab_size": V,
            "num_categories": C,
            "num_subcategories": SC,
            "embeddings": rng.standard_normal((V, self._embedding_size)).astype(
                np.float32
            ),
            "tokens": rng.integers(0, V, (N, self.max_title_length), dtype=np.int32),
            "abstract_tokens": rng.integers(
                0, V, (N, self.max_abstract_length), dtype=np.int32
            ),
            "category_indices": rng.integers(0, C, N, dtype=np.int32),
            "subcategory_indices": rng.integers(0, SC, N, dtype=np.int32),
            "news_ids_original_strings": [f"N{i}" for i in range(N)],
        }

        # --- int↔news_id mapping ---
        self._int_to_news_id = {i: f"N{i}" for i in range(N)}

        # --- behaviours ---
        self.train_behaviors_data = self._make_behaviors(self.NUM_TRAIN)
        self.val_behaviors_data = self._make_behaviors(self.NUM_VAL)
        self.test_behaviors_data = self.val_behaviors_data  # reuse for simplicity

    def _make_behaviors(self, n: int) -> dict[str, Any]:
        rng = self._rng
        V = self.VOCAB_SIZE
        C = self.NUM_CATEGORIES
        SC = self.NUM_SUBCATEGORIES
        K = self.max_impressions_length
        H = self.max_history_length
        TL = self.max_title_length
        AL = self.max_abstract_length

        labels = np.zeros((n, K), dtype=np.float32)
        labels[:, 0] = 1.0  # first candidate is positive

        candidate_news_ids = [
            [f"N{int(nid)}" for nid in rng.integers(0, self.NUM_NEWS, K)]
            for _ in range(n)
        ]

        return {
            "history_news_tokens": rng.integers(0, V, (n, H, TL), dtype=np.int32),
            "history_news_abstract_tokens": rng.integers(
                0, V, (n, H, AL), dtype=np.int32
            ),
            "history_news_categories": rng.integers(0, C, (n, H), dtype=np.int32),
            "history_news_subcategories": rng.integers(0, SC, (n, H), dtype=np.int32),
            "candidate_news_tokens": rng.integers(0, V, (n, K, TL), dtype=np.int32),
            "candidate_news_abstract_tokens": rng.integers(
                0, V, (n, K, AL), dtype=np.int32
            ),
            "candidate_news_categories": rng.integers(0, C, (n, K), dtype=np.int32),
            "candidate_news_subcategories": rng.integers(0, SC, (n, K), dtype=np.int32),
            "labels": labels,
            "user_ids": rng.integers(0, self.NUM_USERS, n, dtype=np.int32),
            "impression_ids": np.arange(n, dtype=np.int32),
            "candidate_news_ids": candidate_news_ids,
        }

    # --- Properties ---

    @property
    def processed_news(self) -> dict[str, Any]:
        return self._processed_news

    @property
    def train_size(self) -> int:
        return len(self.train_behaviors_data["labels"])

    @property
    def val_size(self) -> int:
        return len(self.val_behaviors_data["labels"])

    @property
    def test_size(self) -> int:
        return len(self.test_behaviors_data["labels"])

    def get_int_to_news_id_map(self) -> dict[int, str]:
        return self._int_to_news_id
