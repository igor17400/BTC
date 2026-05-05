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

    def __init__(
        self,
        max_title_length: int = 32,
        max_abstract_length: int = 50,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        max_entities: int = 5,
        embedding_size: int = 300,
        vocab_size: int = 500,
        num_categories: int = 10,
        num_subcategories: int = 20,
        num_entities: int = 50,
        num_news: int = 100,
        num_users: int = 30,
        num_train: int = 80,
        num_val: int = 20,
        popularity_embedding_bins: int = 200,
        recency_embedding_bins: int = 1500,
        process_title: bool = True,
        process_abstract: bool = False,
        process_category: bool = False,
        process_subcategory: bool = False,
        process_user_id: bool = False,
        process_entities: bool = False,
        seed: int = 42,
        # Accept and ignore extra kwargs from Hydra
        **kwargs,
    ):
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.max_entities = max_entities
        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._num_categories = num_categories
        self._num_subcategories = num_subcategories
        self._num_entities = num_entities
        self._num_news = num_news
        self._num_users = num_users
        self._num_train = num_train
        self._num_val = num_val
        self._popularity_embedding_bins = popularity_embedding_bins
        self._recency_embedding_bins = recency_embedding_bins
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory
        self.process_user_id = process_user_id
        self.process_entities = process_entities

        self._rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        rng = self._rng
        V = self._vocab_size
        C = self._num_categories
        SC = self._num_subcategories
        N = self._num_news

        # --- processed_news ---
        # Note: ``entity_indices`` is only emitted when ``process_entities``
        # is True because the dataloader picks it up unconditionally and
        # NAML/LSTUR have no entity slot in their input layout.
        self._processed_news = {
            "vocab_size": V,
            "num_categories": C,
            "num_subcategories": SC,
            "num_users": self._num_users,
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
        if self.process_entities:
            self._processed_news["entity_indices"] = rng.integers(
                0, self._num_entities, (N, self.max_entities), dtype=np.int32
            )
            # PP-Rec needs an entity embedding table to actually take the
            # entity branch in the news encoder. Without it the runner still
            # concatenates entity indices into hist/cand tokens but the
            # model skips entity slicing -> CUDA index OOB on category.
            self._processed_news["entity_vocab_size"] = self._num_entities
            entity_emb_dim = 100  # matches MIND entity_embedding.vec format
            self._processed_news["entity_embeddings"] = rng.standard_normal(
                (self._num_entities, entity_emb_dim)
            ).astype(np.float32)

        # --- int↔news_id mapping ---
        self._int_to_news_id = {i: f"N{i}" for i in range(N)}

        # --- behaviours ---
        self.train_behaviors_data = self._make_behaviors(self._num_train)
        self.val_behaviors_data = self._make_behaviors(self._num_val)
        self.test_behaviors_data = self.val_behaviors_data  # reuse for simplicity

    def _make_behaviors(self, n: int) -> dict[str, Any]:
        rng = self._rng
        V = self._vocab_size
        C = self._num_categories
        SC = self._num_subcategories
        E = self._num_entities
        K = self.max_impressions_length
        H = self.max_history_length
        TL = self.max_title_length
        AL = self.max_abstract_length
        ME = self.max_entities

        labels = np.zeros((n, K), dtype=np.float32)
        labels[:, 0] = 1.0  # first candidate is positive

        candidate_news_ids = [
            [f"N{int(nid)}" for nid in rng.integers(0, self._num_news, K)]
            for _ in range(n)
        ]

        result = {
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
            "user_ids": rng.integers(0, self._num_users, n, dtype=np.int32),
            "impression_ids": np.arange(n, dtype=np.int32),
            "candidate_news_ids": candidate_news_ids,
        }
        # PP-Rec extras (entity / CTR / recency) — only emitted when needed
        # so models that don't expect them aren't tripped up by the dataloader.
        if self.process_entities:
            result["history_news_entities"] = rng.integers(
                0, E, (n, H, ME), dtype=np.int32
            )
            result["candidate_news_entities"] = rng.integers(
                0, E, (n, K, ME), dtype=np.int32
            )
            result["history_news_ctr"] = rng.uniform(0.0, 1.0, (n, H)).astype(
                np.float32
            )
            result["candidate_news_ctr"] = rng.integers(
                0, self._popularity_embedding_bins, (n, K), dtype=np.int32
            )
            result["candidate_news_recency"] = rng.integers(
                0, self._recency_embedding_bins, (n, K), dtype=np.int32
            )
        return result

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
