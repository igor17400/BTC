"""Slim orchestrator dataset class for news recommendation.

Replaces the monolithic ``src.datasets.base_news.NewsDatasetBase`` by delegating
heavy processing to standalone functions in ``src.core.data.processing``.
"""

import collections
import logging
import pickle
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.core.data.datasets.base import BaseNewsDataset
from src.core.data.processing.behaviors import get_test_data, get_train_val_data
from src.core.data.processing.news import process_news as _process_news_pipeline
from src.core.data.processing.news import read_all_news
from src.core.data.processing.vocabulary import segment_text_into_words
from src.core.data.loaders.dataloader import (
    ImpressionIterator,
    NewsBatchDataloader,
    NewsDataLoader,
    UserHistoryBatchDataloader,
)
from src.core.data.processing.knowledge_graph import KnowledgeGraphProcessor
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
from src.core.data.loaders.cache import CacheManager
from src.core.data.encoders.embeddings import EmbeddingsManager
from src.core.data.encoders.bpemb import BPEmbManager
from src.core.data.processing.sampling import ImpressionSampler
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
    ``src.core.data.processing``.
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

        policy = keras.mixed_precision.global_policy()
        if policy.compute_dtype in ("mixed_float16", "float16"):
            self.float_dtype = "float16"
        else:
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
        self._load_data(mode)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_size(self) -> int:
        return (
            len(self.train_behaviors_data["impression_ids"])
            if "impression_ids" in self.train_behaviors_data
            else 0
        )

    @property
    def val_size(self) -> int:
        return (
            len(self.val_behaviors_data["impression_ids"])
            if "impression_ids" in self.val_behaviors_data
            else 0
        )

    @property
    def test_size(self) -> int:
        return (
            len(self.test_behaviors_data["impression_ids"])
            if "impression_ids" in self.test_behaviors_data
            else 0
        )

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
            raise FileNotFoundError(
                f"behaviors.tsv not found at {root_behaviors_path}"
            )

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

        test_behaviors = behaviors_df[
            behaviors_df["time"].dt.date >= test_start_date
        ]
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
    # News processing (delegates to processing.news)
    # ------------------------------------------------------------------

    def process_news(self) -> dict[str, Any]:
        """Process news articles into numerical format.

        Delegates vocabulary building, tokenization, and embedding creation
        to standalone functions and wires up the results.
        """
        # Handle knowledge graph before main processing
        if self.use_knowledge_graph:
            all_news_df = read_all_news(self.dataset_path)
            self._process_knowledge_graph(all_news_df)

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
                create_embeddings_fn=self._create_embeddings,
                console=console,
            )
        )

        # Convert embeddings to framework tensor for model consumption
        if "embeddings" in processed_news_content:
            processed_news_content["embeddings"] = keras.ops.cast(
                keras.ops.convert_to_tensor(processed_news_content["embeddings"]),
                self.float_dtype,
            )

        # Build fast news-id-to-tokens lookup
        self.news_id_str_to_tokens: dict[str, np.ndarray] = {
            nid_str: processed_news_content["tokens"][
                self.news_str_id_to_int_idx[nid_str]
            ]
            for nid_str in processed_news_content["news_ids_original_strings"]
        }

        return processed_news_content

    # ------------------------------------------------------------------
    # Embeddings creation (kept on class for access to managers)
    # ------------------------------------------------------------------

    def _create_embeddings(self, vocab: dict[str, int] | None = None) -> np.ndarray:
        """Create embedding matrix based on language and embedding type."""
        if vocab is not None:
            self.vocab = vocab

        logger.info(
            f"Creating embeddings for language: {self.language}, "
            f"type: {self.embedding_type}..."
        )

        if self.embedding_type == "glove" and self.language == "english":
            return self._create_glove_embeddings()
        elif self.embedding_type == "bpemb":
            return self._create_bpemb_embeddings()
        else:
            return self._create_random_embeddings()

    def _create_glove_embeddings(self) -> np.ndarray:
        """Create embedding matrix using GloVe embeddings."""
        glove_tensor_tf, glove_vocab_map = (
            self.embeddings_manager.load_glove_embeddings_tf_and_vocab_map(
                self.embedding_size
            )
        )
        if glove_tensor_tf is None or glove_vocab_map is None:
            raise ValueError("GloVe embeddings or vocab map could not be loaded.")

        glove_array = keras.ops.convert_to_numpy(glove_tensor_tf)
        glove_mean_np = np.mean(glove_array, axis=0)
        glove_std_np = np.std(glove_array, axis=0)

        embedding_matrix = np.zeros(
            (len(self.vocab), self.embedding_size), dtype=np.float32
        )
        embedding_matrix[self.vocab["[PAD]"]] = np.zeros(
            self.embedding_size, dtype=np.float32
        )
        embedding_matrix[self.vocab["[UNK]"]] = np.random.normal(
            loc=glove_mean_np, scale=glove_std_np, size=self.embedding_size
        ).astype(np.float32)

        if "<NUM>" in self.vocab:
            num_token_id = self.vocab["<NUM>"]
            glove_num_idx = glove_vocab_map.get("<NUM>")
            if glove_num_idx is not None:
                embedding_matrix[num_token_id] = glove_array[glove_num_idx]
            else:
                glove_number_idx = glove_vocab_map.get("number")
                if glove_number_idx is not None:
                    embedding_matrix[num_token_id] = glove_array[glove_number_idx]
                else:
                    embedding_matrix[num_token_id] = np.random.normal(
                        loc=glove_mean_np,
                        scale=glove_std_np,
                        size=self.embedding_size,
                    ).astype(np.float32)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Populating embedding matrix...", total=len(self.vocab)
            )
            for word, idx in self.vocab.items():
                if word in ("[PAD]", "[UNK]", "<NUM>"):
                    progress.advance(task)
                    continue
                glove_word_idx = glove_vocab_map.get(word)
                if glove_word_idx is not None:
                    embedding_matrix[idx] = glove_array[glove_word_idx]
                else:
                    embedding_matrix[idx] = np.random.normal(
                        loc=glove_mean_np,
                        scale=glove_std_np,
                        size=self.embedding_size,
                    ).astype(np.float32)
                progress.advance(task)

        return embedding_matrix

    def _create_bpemb_embeddings(self) -> np.ndarray:
        """Create embedding matrix using BPEmb pre-trained embeddings."""
        logger.info(f"Creating BPEmb embeddings for language: {self.language}")

        lang_map = {
            "japanese": "ja", "german": "de", "french": "fr", "spanish": "es",
            "italian": "it", "portuguese": "pt", "russian": "ru", "korean": "ko",
            "chinese": "zh", "arabic": "ar", "hindi": "hi", "turkish": "tr",
            "polish": "pl", "dutch": "nl", "english": "en",
        }
        lang_code = lang_map.get(self.language.lower(), self.language.lower())

        try:
            logger.info(f"Loading BPEmb embeddings for language: {lang_code}")
            bpemb_embeddings = self.embeddings_manager.get_bpemb_embeddings(
                language=lang_code, vocab_size=200000, dim=self.embedding_size
            )

            if not bpemb_embeddings:
                logger.warning(f"No BPEmb embeddings loaded for {lang_code}")
                return self._create_random_embeddings()

            logger.info(
                f"Creating embedding matrix from {len(bpemb_embeddings):,} BPE tokens"
            )
            embedding_matrix = (
                np.random.randn(len(self.vocab), self.embedding_size).astype(
                    np.float32
                )
                * 0.1
            )
            embedding_matrix[self.vocab["[PAD]"]] = np.zeros(
                self.embedding_size, dtype=np.float32
            )

            matched_words = 0
            for word, idx in self.vocab.items():
                if word in ("[PAD]", "[UNK]", "<NUM>"):
                    continue
                if word in bpemb_embeddings:
                    embedding_matrix[idx] = bpemb_embeddings[word]
                    matched_words += 1
                elif word.lower() in bpemb_embeddings:
                    embedding_matrix[idx] = bpemb_embeddings[word.lower()]
                    matched_words += 1
                else:
                    subword_embeddings = [
                        bpemb_embeddings[bt]
                        for bt in bpemb_embeddings
                        if bt in word and len(bt) > 1
                    ]
                    if subword_embeddings:
                        embedding_matrix[idx] = np.mean(subword_embeddings, axis=0)
                        matched_words += 1

            match_pct = (matched_words / len(self.vocab)) * 100
            logger.info(
                f"Successfully created BPEmb embedding matrix: "
                f"{embedding_matrix.shape}"
            )
            logger.info(
                f"Matched {matched_words}/{len(self.vocab)} words ({match_pct:.1f}%)"
            )
            return embedding_matrix

        except Exception as e:
            logger.error(f"Failed to load BPEmb embeddings for {lang_code}: {e}")
            logger.warning("Falling back to random embeddings")
            return self._create_random_embeddings()

    def _create_random_embeddings(self) -> np.ndarray:
        """Create random embedding matrix as fallback."""
        logger.info(f"Creating random embeddings for language: {self.language}")
        embedding_matrix = (
            np.random.randn(len(self.vocab), self.embedding_size).astype(np.float32)
            * 0.1
        )
        embedding_matrix[self.vocab["[PAD]"]] = np.zeros(
            self.embedding_size, dtype=np.float32
        )
        return embedding_matrix

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
    # Download
    # ------------------------------------------------------------------

    def download_dataset(self) -> None:
        """Download and extract dataset if not already present."""
        if self.dataset_path.exists() and any(self.dataset_path.iterdir()):
            logger.info(f"Found existing dataset at {self.dataset_path}")
            return

        if not self.urls:
            logger.warning(
                f"No URLs provided for downloading dataset. "
                f"Assuming data exists at {self.dataset_path}"
            )
            return

        self.dataset_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            for split, url in self.urls.items():
                zip_path = self.dataset_path / f"{split}.zip"
                extract_path = self.dataset_path / split

                if not extract_path.exists():
                    download_task = progress.add_task(
                        f"Downloading {split} set...", total=None
                    )

                    logger.info(f"Downloading {split} set from {url}")
                    urllib.request.urlretrieve(
                        url,
                        zip_path,
                        reporthook=lambda count, block_size, total_size: progress.update(
                            download_task,
                            total=(
                                total_size // block_size
                                if total_size > 0
                                else None
                            ),
                            completed=count,
                        ),
                    )
                    progress.update(download_task, completed=True)
                    progress.add_task(f"Extracting {split} set...", total=100)

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)

                    zip_path.unlink()
                    logger.info(f"Successfully processed {split} set")
                else:
                    logger.info(f"Found existing {split} set at {extract_path}")

        if self.use_knowledge_graph:
            self._download_knowledge_graph()

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

    def _download_knowledge_graph(self) -> None:
        """Download and extract knowledge graph data."""
        graph_extract_path = self.dataset_path / "download" / "wikidata-graph"

        if not graph_extract_path.exists():
            graph_url = (
                "https://mind201910.blob.core.windows.net/"
                "knowledge-graph/wikidata-graph.zip"
            )
            graph_zip_path = self.dataset_path / "download" / "wikidata-graph.zip"
            self._download_and_unzip_file(
                graph_url, graph_zip_path, graph_extract_path, "knowledge graph"
            )
        else:
            logger.info("Found existing knowledge graph data")

    def _download_and_unzip_file(
        self, url: str, zip_path: Path, extract_path: Path, description: str
    ) -> None:
        """Download a file from a URL and unzip it."""
        logger.info(
            f"Downloading {description} data from {url} to {zip_path}..."
        )
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Extracting {description} to {extract_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        zip_path.unlink()
        logger.info(
            f"Successfully downloaded and extracted {description} data."
        )

    # ------------------------------------------------------------------
    # Dataloader factory methods
    # ------------------------------------------------------------------

    def train_dataloader(self, batch_size: int, model_name: str = "nrms"):
        """Create training dataset with token-based inputs."""
        return NewsDataLoader.create_train_dataset(
            history_news_tokens=self.train_behaviors_data["history_news_tokens"],
            history_news_abstract_tokens=self.train_behaviors_data[
                "history_news_abstract_tokens"
            ],
            history_news_category=self.train_behaviors_data[
                "history_news_categories"
            ],
            history_news_subcategory=self.train_behaviors_data[
                "history_news_subcategories"
            ],
            candidate_news_tokens=self.train_behaviors_data[
                "candidate_news_tokens"
            ],
            candidate_news_abstract_tokens=self.train_behaviors_data[
                "candidate_news_abstract_tokens"
            ],
            candidate_news_category=self.train_behaviors_data[
                "candidate_news_categories"
            ],
            candidate_news_subcategory=self.train_behaviors_data[
                "candidate_news_subcategories"
            ],
            labels=self.train_behaviors_data["labels"],
            user_ids=self.train_behaviors_data["user_ids"],
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
            process_user_id=self.process_user_id,
            model_name=model_name,
        )

    def user_history_dataloader(
        self, mode: str, batch_size: int
    ) -> UserHistoryBatchDataloader:
        """Create dataloader for user history validation/testing."""
        if mode == "val":
            data = self.val_behaviors_data
        elif mode == "test":
            data = self.test_behaviors_data
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return UserHistoryBatchDataloader(
            history_tokens=data["history_news_tokens"],
            history_abstract_tokens=data["history_news_abstract_tokens"],
            history_category=data["history_news_categories"],
            history_subcategory=data["history_news_subcategories"],
            impression_ids=data["impression_ids"],
            user_ids=data["user_ids"],
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
        )

    def impression_dataloader(self, mode: str) -> ImpressionIterator:
        """Create dataloader for impressions validation/testing."""
        if mode == "val":
            data = self.val_behaviors_data
        elif mode == "test":
            data = self.test_behaviors_data
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return ImpressionIterator(
            impression_tokens=data["candidate_news_tokens"],
            impression_abstract_tokens=data["candidate_news_abstract_tokens"],
            impression_category=data["candidate_news_categories"],
            impression_subcategory=data["candidate_news_subcategories"],
            labels=data["labels"],
            impression_ids=data["impression_ids"],
            candidate_ids=data["candidate_news_ids"],
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
        )

    def news_dataloader(self, batch_size: int) -> NewsBatchDataloader:
        """Create dataloader for processed news validation/testing."""
        news_ids = self.processed_news.get(
            "news_ids_original_strings", np.array([])
        )
        news_tokens = self.processed_news.get("tokens", np.array([]))
        news_abstract_tokens = self.processed_news.get(
            "abstract_tokens", np.array([])
        )
        news_category_indices = self.processed_news.get(
            "category_indices", np.array([])
        )
        news_subcategory_indices = self.processed_news.get(
            "subcategory_indices", np.array([])
        )

        return NewsBatchDataloader(
            news_ids=news_ids,
            news_tokens=keras.ops.convert_to_tensor(news_tokens),
            news_abstract_tokens=keras.ops.convert_to_tensor(news_abstract_tokens),
            news_category_indices=keras.ops.convert_to_tensor(
                news_category_indices
            ),
            news_subcategory_indices=keras.ops.convert_to_tensor(
                news_subcategory_indices
            ),
            batch_size=batch_size,
            process_title=self.process_title,
            process_abstract=self.process_abstract,
            process_category=self.process_category,
            process_subcategory=self.process_subcategory,
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

            logger.info("Collecting basic dataset info...")
            collect_basic_dataset_info(self, summary_data)

            logger.info("Collecting news statistics...")
            collect_news_statistics(self, summary_data)

            logger.info("Collecting behavior statistics...")
            collect_behavior_statistics(self, summary_data)

            logger.info("Collecting overall statistics...")
            collect_overall_statistics(self, summary_data)

            logger.info("Collecting quality metrics...")
            collect_quality_metrics(summary_data)

            logger.info("Creating DataFrame and saving to CSV...")
            summary_df = pd.DataFrame([summary_data])
            summary_df = reorder_summary_columns(summary_df)

            summary_file_path = processed_dir / "datasets_summary.csv"
            summary_df.to_csv(summary_file_path, index=False)

            logger.info("Saving unique user IDs to CSV...")
            save_unique_users_to_csv(self)

            logger.info(f"Dataset summary saved to: {summary_file_path}")
            logger.info(f"Summary contains {len(summary_data)} statistics")

            log_key_statistics(summary_data)

            return summary_file_path

        except Exception as e:
            logger.error(f"Error generating dataset summary: {e}")
            logger.error("Continuing without summary generation...")
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
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load entity and context embeddings from files."""
        logger.info("Loading entity and context embeddings...")

        for mode in ("train", "dev", "test"):
            entity_file = self.dataset_path / mode / "entity_embedding.vec"
            context_file = self.dataset_path / mode / "context_embedding.vec"

            if entity_file.exists():
                with open(entity_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split("\t")
                            if len(terms) == 101:
                                self.entity_embeddings[terms[0]] = list(
                                    map(float, terms[1:])
                                )

            if context_file.exists():
                with open(context_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split("\t")
                            if len(terms) == 101:
                                self.context_embeddings[terms[0]] = list(
                                    map(float, terms[1:])
                                )

    # ------------------------------------------------------------------
    # Text segmentation (overridable by subclasses, e.g. Japanese)
    # ------------------------------------------------------------------

    def _segment_text_into_words(self, sent: str) -> list[str]:
        """Segment a sentence string into a list of word strings."""
        return segment_text_into_words(sent)

    def tokenize_text(
        self,
        text: str,
        vocab: dict[str, int],
        max_len: int,
        unk_token_id: int,
        pad_token_id: int,
    ) -> list[int]:
        """Convert a raw text string into a fixed-length token ID sequence."""
        from src.core.data.processing.vocabulary import tokenize_text as _tokenize

        return _tokenize(
            text,
            vocab,
            max_len,
            unk_token_id,
            pad_token_id,
            segment_text_fn=self._segment_text_into_words,
        )
