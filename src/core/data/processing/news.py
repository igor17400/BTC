"""News reading, tokenization, and processing utilities.

Standalone functions extracted from NewsDatasetBase for reading news TSV files,
tokenizing titles/abstracts, and orchestrating the full news processing pipeline.
"""

import json
import logging
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.core.data.processing.vocabulary import (
    build_vocabulary,
    segment_text_into_words,
    tokenize_text,
)

logger = logging.getLogger(__name__)


def read_all_news(dataset_path: Path) -> pd.DataFrame:
    """Read and combine all news data from train/valid/test splits.

    Looks for ``news.tsv`` files in ``train/``, ``valid/``, and ``test/``
    subdirectories of ``dataset_path``. Deduplicates by news ID.

    Args:
        dataset_path: Root path of the dataset.

    Returns:
        Combined DataFrame of all unique news articles.

    Raises:
        FileNotFoundError: If no ``news.tsv`` files are found.
    """
    logger.info("Reading all news (train, dev, test) for vocab building...")
    news_dfs: list[pd.DataFrame] = []

    column_names = [
        "id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    def read_news_file(file_path: Path) -> pd.DataFrame:
        """Read a news.tsv file with auto-detected column format."""
        return pd.read_table(
            file_path,
            header=None,
            names=column_names,
            na_filter=False,
            on_bad_lines="skip",
            engine="python",
        )

    for split in ("train", "valid", "test"):
        news_path = dataset_path / split / "news.tsv"
        if news_path.exists():
            news_dfs.append(read_news_file(news_path))

    if not news_dfs:
        raise FileNotFoundError(
            f"No news.tsv files found in {dataset_path}. "
            f"Expected news.tsv files in train/valid/test directories."
        )

    all_news_df = pd.concat(news_dfs, ignore_index=True).drop_duplicates(
        subset=["id"], keep="first"
    )
    logger.info("All news dataframes joined and duplicates dropped...")
    return all_news_df


def tokenize_all_news(
    all_news_df: pd.DataFrame,
    news_str_id_to_int_idx: dict[str, int],
    vocab: dict[str, int],
    max_title_length: int,
    max_abstract_length: int,
    tokenize_fn: Callable | None = None,
    segment_text_fn: Callable | None = None,
    console: Console | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize all news titles and abstracts into fixed-length integer arrays.

    Args:
        all_news_df: DataFrame with columns ``id``, ``title``, ``abstract``.
        news_str_id_to_int_idx: Mapping from string news IDs to integer indices.
        vocab: Word-to-token-ID vocabulary.
        max_title_length: Maximum number of tokens per title.
        max_abstract_length: Maximum number of tokens per abstract.
        tokenize_fn: Custom tokenize function.  Defaults to
            :func:`tokenize_text`.
        segment_text_fn: Word segmentation function passed through to tokenizer.
        console: Rich Console for progress display.

    Returns:
        Tuple of (tokenized_titles, tokenized_abstracts) as NumPy int32 arrays
        with shapes ``(num_news, max_title_length)`` and
        ``(num_news, max_abstract_length)``.
    """
    if console is None:
        console = Console()

    logger.info("Tokenizing all news titles and abstracts...")
    num_unique_news = len(all_news_df["id"].unique())

    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]

    tokenized_titles_np = np.full(
        (num_unique_news, max_title_length), pad_id, dtype=np.int32
    )
    tokenized_abstracts_np = np.full(
        (num_unique_news, max_abstract_length), pad_id, dtype=np.int32
    )

    def _default_tokenize(text: str, v: dict[str, int], ml: int) -> list[int]:
        return tokenize_text(
            text,
            v,
            ml,
            unk_token_id=unk_id,
            pad_token_id=pad_id,
            segment_text_fn=segment_text_fn,
        )

    _tokenize = tokenize_fn if tokenize_fn is not None else _default_tokenize

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Tokenizing all news titles and abstracts...", total=len(all_news_df)
        )
        for nid_str, title_text, abstract_text in zip(
            all_news_df["id"], all_news_df["title"], all_news_df["abstract"]
        ):
            int_idx = news_str_id_to_int_idx[nid_str]
            tokenized_titles_np[int_idx] = _tokenize(
                str(title_text), vocab, max_title_length
            )
            tokenized_abstracts_np[int_idx] = _tokenize(
                str(abstract_text), vocab, max_abstract_length
            )
            progress.advance(task)

    return tokenized_titles_np, tokenized_abstracts_np


def load_entity_embeddings(dataset_path: Path) -> dict[str, np.ndarray]:
    """Load pre-trained entity embeddings from MIND entity_embedding.vec files.

    Args:
        dataset_path: Root dataset path containing train/valid/test splits.

    Returns:
        Dict mapping WikidataId (e.g. "Q12345") to 100-dim numpy vectors.
    """
    entity_emb_dict: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        emb_path = dataset_path / split / "entity_embedding.vec"
        if not emb_path.exists():
            continue
        with open(emb_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    entity_id = parts[0]
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if entity_id not in entity_emb_dict:
                        entity_emb_dict[entity_id] = vec
    logger.info(f"Loaded {len(entity_emb_dict)} entity embeddings")
    return entity_emb_dict


def parse_entity_indices(
    all_news_df: pd.DataFrame,
    news_str_id_to_int_idx: dict[str, int],
    entity_emb_dict: dict[str, np.ndarray],
    max_entities: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse entity IDs from news title_entities column and build index arrays.

    Args:
        all_news_df: DataFrame with ``title_entities`` column (JSON arrays).
        news_str_id_to_int_idx: News string ID to int index mapping.
        entity_emb_dict: Pre-loaded entity embeddings.
        max_entities: Max entities per news article.

    Returns:
        Tuple of (entity_indices, entity_embedding_matrix, entity_vocab_size).
        entity_indices: (num_news, max_entities) int32 array.
        entity_embedding_matrix: (entity_vocab_size, emb_dim) float32 array.
    """
    # Build entity vocabulary from entities that have embeddings
    entity_to_idx: dict[str, int] = {}  # WikidataId -> index (1-based, 0 = pad)
    emb_dim = 0
    for vec in entity_emb_dict.values():
        emb_dim = len(vec)
        break

    num_news = len(all_news_df["id"].unique())
    entity_indices = np.zeros((num_news, max_entities), dtype=np.int32)

    for nid_str, entities_str in zip(all_news_df["id"], all_news_df["title_entities"]):
        int_idx = news_str_id_to_int_idx.get(nid_str)
        if int_idx is None:
            continue

        try:
            entities = json.loads(str(entities_str)) if entities_str else []
        except (json.JSONDecodeError, TypeError):
            entities = []

        for i, ent in enumerate(entities[:max_entities]):
            wikidata_id = ent.get("WikidataId", "")
            if wikidata_id and wikidata_id in entity_emb_dict:
                if wikidata_id not in entity_to_idx:
                    entity_to_idx[wikidata_id] = len(entity_to_idx) + 1  # 1-based
                entity_indices[int_idx, i] = entity_to_idx[wikidata_id]

    # Build embedding matrix (index 0 = zero padding)
    entity_vocab_size = len(entity_to_idx) + 1
    entity_embedding_matrix = np.zeros((entity_vocab_size, emb_dim), dtype=np.float32)
    for wikidata_id, idx in entity_to_idx.items():
        entity_embedding_matrix[idx] = entity_emb_dict[wikidata_id]

    logger.info(
        f"Built entity vocabulary: {entity_vocab_size} entities "
        f"(embedding dim={emb_dim}), max_entities={max_entities}"
    )
    return entity_indices, entity_embedding_matrix, entity_vocab_size


def process_news(
    dataset_path: Path,
    word_threshold: int = 3,
    language: str = "english",
    max_title_length: int = 30,
    max_abstract_length: int = 50,
    embedding_type: str = "glove",
    embedding_size: int = 300,
    download_if_missing: bool = True,
    download_fn: Callable | None = None,
    segment_text_fn: Callable | None = None,
    create_embeddings_fn: Callable | None = None,
    tokenize_fn: Callable | None = None,
    console: Console | None = None,
    process_entities: bool = False,
    max_entities: int = 5,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    """Orchestrate full news processing: vocab building, tokenization, embeddings.

    This is the main entry point that coordinates vocabulary construction,
    news tokenization, category mapping, and embedding creation.  Results are
    cached to disk so subsequent calls load from pickled files.

    Args:
        dataset_path: Root path of the dataset.
        word_threshold: Minimum word frequency for vocabulary inclusion.
        language: Language identifier (e.g. ``"english"``).
        max_title_length: Maximum title token length.
        max_abstract_length: Maximum abstract token length.
        embedding_type: Embedding type (``"glove"``, ``"bpemb"``, etc.).
        embedding_size: Dimensionality of word embeddings.
        download_if_missing: Whether to trigger download before processing.
        download_fn: Optional callback to download the dataset.
        segment_text_fn: Custom word segmentation function.
        create_embeddings_fn: Callable ``(vocab) -> np.ndarray`` that produces
            the embedding matrix.  If ``None``, no embeddings are created.
        tokenize_fn: Optional custom tokenize function.
        console: Rich Console for progress display.

    Returns:
        Tuple of:
        - ``processed_news_content``: Dict with keys ``tokens``,
          ``abstract_tokens``, ``vocab_size``, ``embeddings`` (NumPy),
          ``category_indices``, ``subcategory_indices``, etc.
        - ``vocab``: The word-to-ID vocabulary dict.
        - ``news_str_id_to_int_idx``: Mapping from string news IDs to row
          indices in the token arrays.
    """
    if segment_text_fn is None:
        segment_text_fn = segment_text_into_words
    if console is None:
        console = Console()

    processed_path = dataset_path / "processed"
    os.makedirs(processed_path, exist_ok=True)

    vocab_file = processed_path / f"vocab_thresh{word_threshold}_{language}.pkl"
    # News tokenization cache depends on title/abstract length.
    processed_news_file = processed_path / (
        f"processed_news_thresh{word_threshold}_{language}"
        f"_t{max_title_length}_a{max_abstract_length}.pkl"
    )
    embeddings_file = (
        processed_path / f"filtered_embeddings_thresh{word_threshold}_{language}.npy"
    )

    if download_if_missing and download_fn is not None:
        download_fn()

    if not (
        processed_news_file.exists()
        and embeddings_file.exists()
        and vocab_file.exists()
    ):
        logger.info(
            f"Processing news data with word_threshold={word_threshold}, "
            f"language={language}..."
        )

        # Read all news data
        all_news_df = read_all_news(dataset_path)

        unique_news_ids_str = all_news_df["id"].unique()
        news_str_id_to_int_idx: dict[str, int] = {
            nid_str: i for i, nid_str in enumerate(unique_news_ids_str)
        }

        # Create category mappings
        unique_categories = sorted(all_news_df["category"].unique())
        unique_subcategories = sorted(all_news_df["subcategory"].unique())
        category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        subcategory_to_idx = {
            subcat: idx for idx, subcat in enumerate(unique_subcategories)
        }

        category_indices = np.zeros(len(unique_news_ids_str), dtype=np.int32)
        subcategory_indices = np.zeros(len(unique_news_ids_str), dtype=np.int32)

        for nid_str, cat, subcat in zip(
            all_news_df["id"],
            all_news_df["category"],
            all_news_df["subcategory"],
        ):
            int_idx = news_str_id_to_int_idx[nid_str]
            category_indices[int_idx] = category_to_idx[cat]
            subcategory_indices[int_idx] = subcategory_to_idx[subcat]

        # Build vocabulary
        vocab = build_vocabulary(
            all_news_df,
            segment_text_fn=segment_text_fn,
            word_threshold=word_threshold,
            console=console,
        )
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)

        # Tokenize all news
        tokenized_titles_np, tokenized_abstracts_np = tokenize_all_news(
            all_news_df,
            news_str_id_to_int_idx,
            vocab,
            max_title_length,
            max_abstract_length,
            tokenize_fn=tokenize_fn,
            segment_text_fn=segment_text_fn,
            console=console,
        )

        # Create embeddings (returned as NumPy array)
        embedding_matrix: np.ndarray | None = None
        if create_embeddings_fn is not None:
            embedding_matrix = create_embeddings_fn(vocab)
            np.save(embeddings_file, embedding_matrix)

        processed_news_content: dict[str, Any] = {
            "news_ids_original_strings": unique_news_ids_str,
            "tokens": tokenized_titles_np,
            "abstract_tokens": tokenized_abstracts_np,
            "vocab_size": len(vocab),
            "category_indices": category_indices,
            "subcategory_indices": subcategory_indices,
            "num_categories": len(unique_categories),
            "num_subcategories": len(unique_subcategories),
        }

        with open(processed_news_file, "wb") as f:
            pickle.dump(processed_news_content, f)
        logger.info("Finished processing news data.")

    else:
        logger.info("Loading processed news data from cache...")
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
        with open(processed_news_file, "rb") as f:
            processed_news_content = pickle.load(f)

        news_str_id_to_int_idx = {
            nid_str: i
            for i, nid_str in enumerate(
                processed_news_content["news_ids_original_strings"]
            )
        }

    # Load embeddings as NumPy array (no framework conversion here)
    if embeddings_file.exists():
        logger.info("Loading filtered embeddings...")
        processed_news_content["embeddings"] = np.load(embeddings_file)

    processed_news_content["vocab"] = vocab

    # --- Entity processing (optional, for PP-Rec) ---
    if process_entities:
        entity_cache = processed_path / f"entity_data_max{max_entities}.pkl"
        if entity_cache.exists():
            logger.info("Loading cached entity data...")
            with open(entity_cache, "rb") as f:
                entity_data = pickle.load(f)
            processed_news_content["entity_indices"] = entity_data["entity_indices"]
            processed_news_content["entity_embeddings"] = entity_data[
                "entity_embeddings"
            ]
            processed_news_content["entity_vocab_size"] = entity_data[
                "entity_vocab_size"
            ]
        else:
            logger.info("Processing entity embeddings...")
            entity_emb_dict = load_entity_embeddings(dataset_path)
            if entity_emb_dict:
                all_news_df = read_all_news(dataset_path)
                entity_indices, entity_emb_matrix, entity_vocab_size = (
                    parse_entity_indices(
                        all_news_df,
                        news_str_id_to_int_idx,
                        entity_emb_dict,
                        max_entities=max_entities,
                    )
                )
                processed_news_content["entity_indices"] = entity_indices
                processed_news_content["entity_embeddings"] = entity_emb_matrix
                processed_news_content["entity_vocab_size"] = entity_vocab_size
                with open(entity_cache, "wb") as f:
                    pickle.dump(
                        {
                            "entity_indices": entity_indices,
                            "entity_embeddings": entity_emb_matrix,
                            "entity_vocab_size": entity_vocab_size,
                        },
                        f,
                    )
            else:
                logger.warning("No entity embeddings found — entity features disabled.")

    return processed_news_content, vocab, news_str_id_to_int_idx
