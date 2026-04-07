"""Orchestrator dataset class for news recommendation.

Delegates heavy processing to standalone modules:
- Download: ``src.core.data.download``
- News processing: ``src.core.data.processing.news``
- Behavior processing: ``src.core.data.processing.behaviors``
- Embedding creation: ``src.core.data.processing.embeddings``
- Sampling: ``src.core.data.processing.sampling``
"""

from __future__ import annotations

import collections
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console

from src.core.data.datasets.base import BaseNewsDataset
from src.core.data.download import download_dataset
from src.core.data.encoders.bpemb import BPEmbManager
from src.core.data.encoders.embeddings import EmbeddingsManager
from src.core.data.loaders.cache import CacheManager
from src.core.data.processing.behaviors import get_test_data, get_train_val_data
from src.core.data.processing.embeddings import create_embeddings
from src.core.data.processing.knowledge_graph import KnowledgeGraphProcessor
from src.core.data.processing.news import process_news as _process_news_pipeline
from src.core.data.processing.news import read_all_news
from src.core.data.processing.sampling import ImpressionSampler
from src.core.data.processing.vocabulary import segment_text_into_words
from src.core.data.stats import (
    apply_data_fraction,
    collect_basic_dataset_info,
    collect_behavior_statistics,
    collect_news_statistics,
    collect_overall_statistics,
    collect_quality_metrics,
    display_statistics,
    log_key_statistics,
    reorder_summary_columns,
    save_unique_users_to_csv,
)
from src.core.io.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
console = Console()


class NewsDatasetBase(BaseNewsDataset):
    """Base class for news recommendation datasets following MIND format.

    Supports three modes of operation:
    1. Pre-split datasets (with train/valid directories)
    2. Auto-splitting of single behaviors.tsv files (auto_split_behaviors=True)
    3. Auto-conversion of custom formats to MIND format (auto_convert_format=True)

    Heavy lifting is delegated to standalone processing functions in
    ``src.core.data.processing`` and ``src.core.data.download``.
    """

    def __init__(
        self,
        name: str,
        version: str,
        data_path: str | None = None,
        urls: dict | None = None,
        language: str = "english",
        max_title_length: int = 30,
        max_abstract_length: int = 50,
        max_history_length: int = 50,
        max_impressions_length: int = 5,
        seed: int = 42,
        embedding_type: str = "glove",
        embedding_size: int = 300,
        sampling: DictConfig | None = None,
        data_fraction_train: float = 1.0,
        data_fraction_val: float = 1.0,
        data_fraction_test: float = 1.0,
        mode: str = "train",
        use_knowledge_graph: bool = False,
        random_train_samples: bool = False,
        validation_split_strategy: str = "chronological",
        validation_split_percentage: float = 0.05,
        validation_split_seed: int | None = None,
        auto_split_behaviors: bool = False,
        auto_convert_format: bool = False,
        word_threshold: int = 3,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        process_user_id: bool = False,
        process_entities: bool = False,
        max_entities: int = 1000,
        max_relations: int = 500,
        download_if_missing: bool = True,
        id_prefix: str = "N",
        user_id_prefix: str = "U",
    ):
        super().__init__()
        self.name = name
        self.version = version
        self.language = language
        self.use_knowledge_graph = use_knowledge_graph
        self.cache_manager = CacheManager()

        # ID prefixes for parsing
        self.id_prefix = id_prefix
        self.user_id_prefix = user_id_prefix

        if data_path:
            self.dataset_path = Path(data_path)
        else:
            self.dataset_path = self.cache_manager.get_dataset_path(
                name.lower().replace(" ", "_"), version
            )

        self.urls = urls
        self.download_if_missing = download_if_missing
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length
        self.max_history_length = max_history_length
        self.max_impressions_length = max_impressions_length
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.embeddings_manager = EmbeddingsManager(self.cache_manager)
        self.bpemb_manager = BPEmbManager(self.cache_manager)
        self.sampler = ImpressionSampler(
            sampling if sampling is not None else DictConfig({})
        )
        self.data_fraction_train = data_fraction_train
        self.data_fraction_val = data_fraction_val
        self.data_fraction_test = data_fraction_test
        self.mode = mode
        self.random_train_samples = random_train_samples
        self.word_threshold = word_threshold
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory
        self.process_user_id = process_user_id
        self.process_entities = process_entities

        self.float_dtype = "float32"

        self.validation_split_strategy = validation_split_strategy
        self.validation_split_percentage = validation_split_percentage
        self.validation_split_seed = (
            validation_split_seed if validation_split_seed is not None else seed
        )
        self.auto_split_behaviors = auto_split_behaviors
        self.auto_convert_format = auto_convert_format

        self.max_entities = max_entities
        self.max_relations = max_relations
        self.entity_embeddings: dict[str, list] = {}
        self.context_embeddings: dict[str, list] = {}
        self.entity_embedding_relation: dict[str, set] = collections.defaultdict(set)

        logger.info(f"Initializing {name} dataset ({version} version)")
        logger.info(f"Language: {language}")
        logger.info(f"Data will be stored in: {self.dataset_path}")

        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.train_val_news_data: dict[str, Any] = {}
        self.train_behaviors_data: dict[str, Any] = {}
        self.val_behaviors_data: dict[str, Any] = {}
        self.test_news_data: dict[str, Any] = {}
        self.test_behaviors_data: dict[str, Any] = {}

        np.random.seed(seed)

        # News ID mappings
        self._news_id_to_int_map: dict[str, int] = {}
        self._int_to_news_id_map: dict[int, str] = {}
        self._next_news_int_id = 0

        # Handle format conversion before processing (if needed)
        if self._check_conversion_needed():
            logger.info(
                "Format conversion needed. Converting custom format to MIND format..."
            )
            conversion_path = self._convert_custom_format()
            logger.info(f"Conversion dataset saved at: {conversion_path}")

        # Handle auto-splitting if needed
        if (
            self.auto_split_behaviors
            and not self._has_pre_split_data()
            and self._has_single_behaviors_file()
        ):
            logger.info(
                "Auto-splitting enabled and single behaviors.tsv found. "
                "Performing automatic split..."
            )
            self._auto_split_behaviors_file()

        self.processed_news = self.process_news()
        if self.process_entities:
            self._compute_news_ctr()
        self._load_data(mode)
        self._compute_num_users()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_size(self) -> int:
        if self.train_behaviors_data and "labels" in self.train_behaviors_data:
            return len(self.train_behaviors_data["labels"])
        return 0

    @property
    def val_size(self) -> int:
        if self.val_behaviors_data and "labels" in self.val_behaviors_data:
            return len(self.val_behaviors_data["labels"])
        return 0

    @property
    def test_size(self) -> int:
        if self.test_behaviors_data and "labels" in self.test_behaviors_data:
            return len(self.test_behaviors_data["labels"])
        return 0

    # ------------------------------------------------------------------
    # ID parsing
    # ------------------------------------------------------------------

    def parse_news_id(self, news_id: str) -> int:
        """Parse news ID to integer, creating mapping for string IDs if needed."""
        if self.id_prefix and news_id.startswith(self.id_prefix):
            try:
                return int(news_id.split(self.id_prefix)[1])
            except ValueError:
                if news_id not in self._news_id_to_int_map:
                    self._news_id_to_int_map[news_id] = self._next_news_int_id
                    self._int_to_news_id_map[self._next_news_int_id] = news_id
                    self._next_news_int_id += 1
                return self._news_id_to_int_map[news_id]

        try:
            return int(news_id)
        except ValueError:
            if news_id not in self._news_id_to_int_map:
                self._news_id_to_int_map[news_id] = self._next_news_int_id
                self._int_to_news_id_map[self._next_news_int_id] = news_id
                self._next_news_int_id += 1
            return self._news_id_to_int_map[news_id]

    def get_int_to_news_id_map(self) -> dict[int, str]:
        """Get the inverse mapping from integer IDs to string news IDs."""
        return self._int_to_news_id_map

    def _rebuild_id_mappings(self) -> None:
        """Rebuild ID mappings from processed news data when loading existing files."""
        if "news_ids_original_strings" not in self.processed_news:
            logger.warning("No news_ids_original_strings found in processed news data")
            return

        logger.info("Rebuilding ID mappings from processed news data...")
        news_ids = self.processed_news["news_ids_original_strings"]

        self._news_id_to_int_map.clear()
        self._int_to_news_id_map.clear()
        self._next_news_int_id = 0

        for original_id in news_ids:
            self.parse_news_id(original_id)

        logger.info(
            f"Rebuilt ID mappings: {len(self._news_id_to_int_map)} news IDs mapped"
        )

    def parse_user_id(self, user_id: str) -> int:
        """Parse user ID to integer, handling optional prefix."""
        if self.user_id_prefix and user_id.startswith(self.user_id_prefix):
            return int(user_id.split(self.user_id_prefix)[1])
        return int(user_id)

    # ------------------------------------------------------------------
    # Auto-split and auto-convert logic
    # ------------------------------------------------------------------

    def _has_pre_split_data(self) -> bool:
        """Check if dataset has pre-split train/valid directories."""
        train_behaviors = self.dataset_path / "train" / "behaviors.tsv"
        valid_behaviors = self.dataset_path / "valid" / "behaviors.tsv"
        return train_behaviors.exists() and valid_behaviors.exists()

    def _has_single_behaviors_file(self) -> bool:
        """Check if dataset has a single behaviors.tsv in the root directory."""
        return (self.dataset_path / "behaviors.tsv").exists()

    def _convert_custom_format(self) -> Path:
        """Convert custom dataset format to MIND format."""
        from src.core.data.processing.custom_format import preprocess_custom_dataset

        logger.info("Auto-converting custom dataset format to MIND format...")
        conversion_path = self.dataset_path / "conversion"

        if (conversion_path / "behaviors.tsv").exists() and (
            conversion_path / "news.tsv"
        ).exists():
            logger.info(
                f"Conversion already exists at {conversion_path}, skipping conversion"
            )
            return conversion_path

        try:
            preprocess_custom_dataset(
                input_dir=self.dataset_path,
                output_dir=conversion_path,
                user_id_prefix=self.user_id_prefix,
                news_id_prefix=self.id_prefix,
                time_format="mind",
            )
            logger.info(
                f"Successfully converted dataset to MIND format at {conversion_path}"
            )
            return conversion_path
        except Exception as e:
            logger.error(f"Failed to convert dataset format: {e}")
            raise RuntimeError(f"Dataset format conversion failed: {e}")

    def _check_conversion_needed(self) -> bool:
        """Check if custom format conversion is needed."""
        if not self.auto_convert_format:
            return False

        original_behaviors = self.dataset_path / "behaviors.tsv"
        original_news = self.dataset_path / "news.tsv"
        conversion_path = self.dataset_path / "conversion"

        has_original = original_behaviors.exists() and original_news.exists()
        has_conversion = (conversion_path / "behaviors.tsv").exists()

        return has_original and not has_conversion

    def _auto_split_behaviors_file(self) -> None:
        """Automatically split a single behaviors.tsv into train/valid splits."""
        logger.info(
            "Auto-splitting single behaviors.tsv file into train/valid/test splits..."
        )

        if self.auto_convert_format:
            root_behaviors_path = self.dataset_path / "conversion" / "behaviors.tsv"
        else:
            root_behaviors_path = self.dataset_path / "behaviors.tsv"

        if not root_behaviors_path.exists():
            raise FileNotFoundError(f"behaviors.tsv not found at {root_behaviors_path}")

        behaviors_df = pd.read_csv(
            root_behaviors_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        behaviors_df["time"] = pd.to_datetime(
            behaviors_df["time"], format="%m/%d/%Y %I:%M:%S %p"
        )
        behaviors_df = behaviors_df.sort_values("time").reset_index(drop=True)

        unique_dates = sorted(behaviors_df["time"].dt.date.unique())
        test_start_date = unique_dates[-1]

        test_behaviors = behaviors_df[behaviors_df["time"].dt.date >= test_start_date]
        remaining_behaviors = behaviors_df[
            behaviors_df["time"].dt.date < test_start_date
        ]

        logger.info(
            f"Test split: {len(test_behaviors):,} behaviors from "
            f"{test_start_date} onwards"
        )

        train_dir = self.dataset_path / "train"
        valid_dir = self.dataset_path / "valid"
        train_dir.mkdir(exist_ok=True)
        valid_dir.mkdir(exist_ok=True)

        remaining_behaviors.to_csv(
            train_dir / "behaviors.tsv", sep="\t", header=False, index=False
        )
        test_behaviors.to_csv(
            valid_dir / "behaviors.tsv", sep="\t", header=False, index=False
        )

        if self.auto_convert_format:
            news_path = self.dataset_path / "conversion" / "news.tsv"
        else:
            news_path = self.dataset_path / "news.tsv"

        if news_path.exists():
            shutil.copy2(news_path, train_dir / "news.tsv")
            shutil.copy2(news_path, valid_dir / "news.tsv")
            logger.info("Copied news.tsv to train and valid directories")

        logger.info(
            "Successfully auto-split behaviors.tsv: "
            "train+val data -> train/, test data -> valid/"
        )

    # ------------------------------------------------------------------
    # News processing
    # ------------------------------------------------------------------

    def process_news(self) -> dict[str, Any]:
        """Process news articles into numerical format.

        Delegates vocabulary building, tokenization, and embedding creation
        to standalone functions and wires up the results.
        """
        if self.use_knowledge_graph:
            all_news_df = read_all_news(self.dataset_path)
            self._process_knowledge_graph(all_news_df)

        def _create_embeddings_fn(vocab=None):
            return create_embeddings(
                vocab=vocab if vocab is not None else self.vocab,
                embedding_size=self.embedding_size,
                embedding_type=self.embedding_type,
                language=self.language,
                embeddings_manager=self.embeddings_manager,
            )

        processed_news_content, self.vocab, self.news_str_id_to_int_idx = (
            _process_news_pipeline(
                dataset_path=self.dataset_path,
                word_threshold=self.word_threshold,
                language=self.language,
                max_title_length=self.max_title_length,
                max_abstract_length=self.max_abstract_length,
                embedding_type=self.embedding_type,
                embedding_size=self.embedding_size,
                download_if_missing=self.download_if_missing,
                download_fn=self.download_dataset,
                segment_text_fn=self._segment_text_into_words,
                create_embeddings_fn=_create_embeddings_fn,
                console=console,
                process_entities=self.process_entities,
                max_entities=self.max_entities,
            )
        )

        if "embeddings" in processed_news_content:
            processed_news_content["embeddings"] = np.asarray(
                processed_news_content["embeddings"], dtype=self.float_dtype
            )

        self.news_id_str_to_tokens: dict[str, np.ndarray] = {
            nid_str: processed_news_content["tokens"][
                self.news_str_id_to_int_idx[nid_str]
            ]
            for nid_str in processed_news_content["news_ids_original_strings"]
        }

        return processed_news_content

    # ------------------------------------------------------------------
    # Download (delegates to src.core.data.download)
    # ------------------------------------------------------------------

    def download_dataset(self) -> None:
        """Download and extract dataset if not already present."""
        download_dataset(
            dataset_path=self.dataset_path,
            urls=self.urls,
            use_knowledge_graph=self.use_knowledge_graph,
        )
        self.cache_manager.add_to_cache(
            self.name.lower().replace(" ", "_"),
            self.version,
            "dataset",
            metadata={
                "splits": list(self.urls.keys()) if self.urls else [],
                "max_title_length": self.max_title_length,
                "max_history_length": self.max_history_length,
                "version": self.version,
                "use_knowledge_graph": self.use_knowledge_graph,
                "language": self.language,
            },
        )

    # ------------------------------------------------------------------
    # Data loading and processing (delegates to processing.behaviors)
    # ------------------------------------------------------------------

    def _load_data(self, mode: str = "train") -> bool:
        """Try to load processed tensor data from disk."""
        processed_path = self.dataset_path / "processed"
        files_exist = (
            (processed_path / "processed_train.pkl").exists()
            and (processed_path / "processed_val.pkl").exists()
            and (processed_path / "processed_test.pkl").exists()
        )

        if not files_exist:
            self._process_data()
        else:
            self._rebuild_id_mappings()

        logger.info("Files have already been processed, loading data...")
        try:
            if mode == "train":
                logger.info("Loading train behaviors data...")
                self.train_behaviors_data = pd.read_pickle(
                    processed_path / "processed_train.pkl"
                )
                logger.info("Loading validation behaviors data...")
                self.val_behaviors_data = pd.read_pickle(
                    processed_path / "processed_val.pkl"
                )

                if self.data_fraction_train < 1.0:
                    self.train_behaviors_data = apply_data_fraction(
                        self.train_behaviors_data, self.data_fraction_train
                    )
                if self.data_fraction_val < 1.0:
                    self.val_behaviors_data = apply_data_fraction(
                        self.val_behaviors_data, self.data_fraction_val
                    )

                self._display_statistics(
                    mode,
                    processed_news=self.processed_news,
                    train_behaviors_data=self.train_behaviors_data,
                    val_behaviors_data=self.val_behaviors_data,
                )
                logger.info("Successfully loaded processed train/val tensors")
                return True
            else:
                logger.info("Loading test behaviors data...")
                self.test_behaviors_data = pd.read_pickle(
                    processed_path / "processed_test.pkl"
                )

                if self.data_fraction_test < 1.0:
                    self.test_behaviors_data = apply_data_fraction(
                        self.test_behaviors_data, self.data_fraction_test
                    )

                self._display_statistics(
                    mode,
                    processed_news=self.processed_news,
                    test_behaviors_data=self.test_behaviors_data,
                )
                logger.info("Successfully loaded processed test tensors")
                return True

        except Exception as e:
            logger.warning(f"Failed to load tensors: {str(e)}")
            return False

    def _compute_num_users(self) -> None:
        """Compute total number of unique users across ALL splits from behaviors.tsv files.

        Scans user IDs directly from the raw behaviors files (train, valid, test)
        so the embedding table covers every user the model will encounter,
        matching standard MIND benchmark practice.
        """
        all_user_ids = set()

        # Scan raw behaviors.tsv files for user IDs across all splits
        for split_dir in ["train", "valid", "test"]:
            behaviors_path = self.dataset_path / split_dir / "behaviors.tsv"
            if behaviors_path.exists():
                df = pd.read_csv(
                    behaviors_path,
                    sep="\t",
                    header=None,
                    usecols=[1],  # user_id column
                    names=["user_id"],
                )
                for uid in df["user_id"].unique():
                    all_user_ids.add(self.parse_user_id(str(uid)))

        if all_user_ids:
            # Use max user ID + 1 as the embedding size to handle sparse ID spaces
            num_users = max(all_user_ids) + 1
            self.processed_news["num_users"] = num_users
            logger.info(
                f"Computed num_users: {num_users} "
                f"(max user ID + 1 from {len(all_user_ids)} unique users across all splits)"
            )

    def _compute_news_ctr(self) -> None:
        """Compute aggregate CTR for each news article from training behaviors.

        Reads the raw training behaviors.tsv, counts clicks and impressions
        per news article, and stores CTR values in ``processed_news["news_ctr"]``.
        Used by PP-Rec for popularity modeling.
        """
        ctr_cache = self.dataset_path / "processed" / "news_ctr.npy"
        if ctr_cache.exists():
            logger.info("Loading cached news CTR data...")
            self.processed_news["news_ctr"] = np.load(ctr_cache)
            return

        train_path = self.dataset_path / "train" / "behaviors.tsv"
        if not train_path.exists():
            logger.warning("No training behaviors found — CTR computation skipped.")
            return

        logger.info("Computing per-news CTR from training behaviors...")
        news_ids_str = self.processed_news["news_ids_original_strings"]
        news_str_to_idx = {nid: i for i, nid in enumerate(news_ids_str)}
        num_news = len(news_ids_str)

        click_counts = np.zeros(num_news, dtype=np.float32)
        impression_counts = np.zeros(num_news, dtype=np.float32)

        df = pd.read_csv(
            train_path, sep="\t", header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        for impressions_str in df["impressions"]:
            if pd.isna(impressions_str):
                continue
            for item in str(impressions_str).split():
                parts = item.split("-")
                if len(parts) >= 2:
                    nid_str, label = parts[0], parts[1]
                    idx = news_str_to_idx.get(nid_str)
                    if idx is not None:
                        impression_counts[idx] += 1.0
                        if label == "1":
                            click_counts[idx] += 1.0

        news_ctr = click_counts / (impression_counts + 0.01)
        self.processed_news["news_ctr"] = news_ctr
        np.save(ctr_cache, news_ctr)
        logger.info(
            f"Computed CTR for {int((impression_counts > 0).sum())} news articles "
            f"(mean CTR={news_ctr[impression_counts > 0].mean():.4f})"
        )

    def _process_data(self) -> None:
        """Process train/val/test data and save to disk."""
        processed_path = self.dataset_path / "processed"

        logger.info("Processing train data...")
        train_behaviors_dict, val_behaviors_dict = self.get_train_val_data()

        logger.info("Saving training data...")
        with open(processed_path / "processed_train.pkl", "wb") as f:
            pickle.dump(train_behaviors_dict, f)

        logger.info("Processing validation data...")
        with open(processed_path / "processed_val.pkl", "wb") as f:
            pickle.dump(val_behaviors_dict, f)

        logger.info("Processing test data...")
        test_behaviors_dict = self.get_test_data()

        logger.info("Saving test data...")
        with open(processed_path / "processed_test.pkl", "wb") as f:
            pickle.dump(test_behaviors_dict, f)

        self.train_behaviors_data = train_behaviors_dict
        self.val_behaviors_data = val_behaviors_dict
        self.test_behaviors_data = test_behaviors_dict

        logger.info("Generating dataset summary...")
        self.generate_dataset_summary()

        logger.info("Preprocessing complete!")

    def get_train_val_data(
        self,
        sampled_user_set: set[str] | None = None,
    ) -> tuple[dict[str, np.ndarray | list], dict[str, np.ndarray | list]]:
        """Load and process training data, splitting into train and validation sets."""
        return get_train_val_data(
            dataset_path=self.dataset_path,
            validation_split_strategy=self.validation_split_strategy,
            validation_split_percentage=self.validation_split_percentage,
            validation_split_seed=self.validation_split_seed,
            processed_news=self.processed_news,
            parse_news_id=self.parse_news_id,
            parse_user_id=self.parse_user_id,
            sampler=self.sampler,
            max_history_length=self.max_history_length,
            max_title_length=self.max_title_length,
            max_abstract_length=self.max_abstract_length,
            random_train_samples=self.random_train_samples,
            float_dtype=self.float_dtype,
            sampled_user_set=sampled_user_set,
            console=console,
        )

    def get_test_data(
        self,
        sampled_user_set: set[str] | None = None,
    ) -> dict[str, np.ndarray | list]:
        """Load and process test data."""
        return get_test_data(
            dataset_path=self.dataset_path,
            processed_news=self.processed_news,
            parse_news_id=self.parse_news_id,
            parse_user_id=self.parse_user_id,
            sampler=self.sampler,
            max_history_length=self.max_history_length,
            max_title_length=self.max_title_length,
            max_abstract_length=self.max_abstract_length,
            random_train_samples=self.random_train_samples,
            float_dtype=self.float_dtype,
            sampled_user_set=sampled_user_set,
            console=console,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _display_statistics(
        self,
        mode: str = "train",
        processed_news: dict[str, Any] | None = None,
        train_behaviors_data: dict[str, Any] | None = None,
        val_behaviors_data: dict[str, Any] | None = None,
        test_behaviors_data: dict[str, Any] | None = None,
    ) -> None:
        if mode == "train":
            data_dict = {
                "news": processed_news,
                "train_behaviors": train_behaviors_data,
                "val_behaviors": val_behaviors_data,
            }
        else:
            data_dict = {
                "news": processed_news,
                "test_behaviors": test_behaviors_data,
            }
        display_statistics(data_dict, mode)

    def generate_dataset_summary(self) -> None:
        """Generate a comprehensive dataset summary CSV file."""
        logger.info("Generating dataset summary...")

        processed_dir = self.dataset_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        try:
            summary_data: dict[str, Any] = {}

            collect_basic_dataset_info(self, summary_data)
            collect_news_statistics(self, summary_data)
            collect_behavior_statistics(self, summary_data)
            collect_overall_statistics(self, summary_data)
            collect_quality_metrics(summary_data)

            summary_df = pd.DataFrame([summary_data])
            summary_df = reorder_summary_columns(summary_df)

            summary_file_path = processed_dir / "datasets_summary.csv"
            summary_df.to_csv(summary_file_path, index=False)

            save_unique_users_to_csv(self)

            logger.info(f"Dataset summary saved to: {summary_file_path}")
            log_key_statistics(summary_data)

            return summary_file_path

        except Exception as e:
            logger.error(f"Error generating dataset summary: {e}")
            return None

    # ------------------------------------------------------------------
    # Knowledge graph
    # ------------------------------------------------------------------

    def _process_knowledge_graph(self, all_news_df: pd.DataFrame) -> None:
        """Process knowledge graph data using KnowledgeGraphProcessor."""
        logger.info("Processing knowledge graph data...")

        kg_processor = KnowledgeGraphProcessor(
            cache_dir=self.dataset_path / "knowledge_graph",
            dataset_path=self.dataset_path,
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )
        kg_processor.process(all_news_df["title"])
        self._load_kg_embeddings()

    def _load_kg_embeddings(self) -> None:
        """Load entity and context embeddings from files."""
        logger.info("Loading entity and context embeddings...")

        for mode in ("train", "dev", "test"):
            entity_file = self.dataset_path / mode / "entity_embedding.vec"
            context_file = self.dataset_path / mode / "context_embedding.vec"

            for filepath, target_dict in (
                (entity_file, self.entity_embeddings),
                (context_file, self.context_embeddings),
            ):
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            if len(line.strip()) > 0:
                                terms = line.strip().split("\t")
                                if len(terms) == 101:
                                    target_dict[terms[0]] = list(map(float, terms[1:]))

    # ------------------------------------------------------------------
    # Text segmentation (overridable by subclasses, e.g. Japanese)
    # ------------------------------------------------------------------

    def _segment_text_into_words(self, sent: str) -> list[str]:
        """Segment a sentence string into a list of word strings."""
        return segment_text_into_words(sent)

    def tokenize_text(
        self, text: str, max_length: int, vocab: dict[str, int] | None = None
    ) -> list[int]:
        """Convert text to token ID sequence with padding/truncation."""
        if vocab is None:
            vocab = self.vocab

        words = self._segment_text_into_words(text)
        token_ids = [
            vocab.get(word, vocab.get("[UNK]", 1)) for word in words[:max_length]
        ]

        # Pad to max_length
        pad_id = vocab.get("[PAD]", 0)
        token_ids.extend([pad_id] * (max_length - len(token_ids)))

        return token_ids
