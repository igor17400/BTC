"""Vocabulary building and text tokenization utilities.

Standalone functions extracted from NewsDatasetBase for building word vocabularies
and tokenizing text into numerical token sequences.
"""

import collections
import logging
import re
from typing import Callable, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.core.data.stats import string_is_number

logger = logging.getLogger(__name__)


def segment_text_into_words(sent: str) -> List[str]:
    """Segment a sentence string into a list of word strings (English default).

    Uses a regex pattern to extract words and common punctuation.

    Args:
        sent: Input sentence string.

    Returns:
        List of lowercased word tokens.
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    return pat.findall(sent.lower()) if isinstance(sent, str) else []


def tokenize_text(
    text: str,
    vocab: Dict[str, int],
    max_len: int,
    unk_token_id: int,
    pad_token_id: int,
    segment_text_fn: Optional[Callable[[str], List[str]]] = None,
) -> List[int]:
    """Convert a raw text string into a fixed-length sequence of numerical token IDs.

    Args:
        text: Raw text string to tokenize.
        vocab: Dictionary mapping words to integer token IDs.
        max_len: Maximum sequence length (truncate or pad to this length).
        unk_token_id: Token ID for unknown words.
        pad_token_id: Token ID for padding.
        segment_text_fn: Optional custom word segmentation function.
            Defaults to ``segment_text_into_words``.

    Returns:
        List of integer token IDs of length ``max_len``.
    """
    if segment_text_fn is None:
        segment_text_fn = segment_text_into_words

    words = segment_text_fn(text)
    tokens = []
    for word in words:
        processed_word = "<NUM>" if string_is_number(word) else word
        tokens.append(vocab.get(processed_word, unk_token_id))

    tokens = tokens[:max_len]
    tokens.extend([pad_token_id] * (max_len - len(tokens)))
    return tokens


def build_vocabulary(
    news_df,
    segment_text_fn: Optional[Callable[[str], List[str]]] = None,
    word_threshold: int = 3,
    console: Optional[Console] = None,
) -> Dict[str, int]:
    """Build vocabulary from news titles and abstracts.

    Words that appear fewer than ``word_threshold`` times are excluded to reduce
    vocabulary size and improve generalisation.

    Args:
        news_df: DataFrame with ``title`` and ``abstract`` columns.
        segment_text_fn: Function to split text into word tokens.
            Defaults to ``segment_text_into_words``.
        word_threshold: Minimum word frequency to be included in the vocabulary.
        console: Rich Console instance for progress display.

    Returns:
        Dictionary mapping word strings to integer token IDs.
    """
    if segment_text_fn is None:
        segment_text_fn = segment_text_into_words
    if console is None:
        console = Console()

    logger.info("Building vocabulary from news titles and abstracts...")
    word_counter: collections.Counter = collections.Counter()

    total_items = len(news_df) * 2  # titles + abstracts
    logger.info(f"Counting words from {len(news_df):,} news titles and abstracts...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Counting words in titles and abstracts...", total=total_items
        )

        # Process titles
        for title_text in news_df["title"].values:
            words = segment_text_fn(str(title_text))
            for word in words:
                processed_word = "<NUM>" if string_is_number(word) else word
                word_counter[processed_word] += 1
            progress.advance(task)

        # Process abstracts
        for abstract_text in news_df["abstract"].values:
            words = segment_text_fn(str(abstract_text))
            for word in words:
                processed_word = "<NUM>" if string_is_number(word) else word
                word_counter[processed_word] += 1
            progress.advance(task)

    # Build vocabulary with special tokens
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    token_id_counter = 2

    if "<NUM>" in word_counter and word_counter["<NUM>"] >= word_threshold:
        vocab["<NUM>"] = token_id_counter
        token_id_counter += 1

    sorted_word_counts = sorted(
        word_counter.items(), key=lambda item: item[1], reverse=True
    )

    for word, count in sorted_word_counts:
        if word in vocab:
            continue
        if count >= word_threshold:
            vocab[word] = token_id_counter
            token_id_counter += 1

    logger.info(f"Vocabulary size (after threshold {word_threshold}): {len(vocab)}")
    return vocab
