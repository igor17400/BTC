"""Embedding matrix creation for news recommendation models.

Supports GloVe, BPEmb, and random initialization strategies.
"""

from __future__ import annotations

import logging

import numpy as np

from src.core.io.logging import console
from src.core.io.progress import create_progress

logger = logging.getLogger(__name__)


def create_embeddings(
    vocab: dict[str, int],
    embedding_size: int,
    embedding_type: str,
    language: str,
    embeddings_manager,
) -> np.ndarray:
    """Create embedding matrix based on language and embedding type.

    Args:
        vocab: Word-to-index mapping.
        embedding_size: Dimension of embeddings.
        embedding_type: One of "glove", "bpemb", or "random".
        language: Dataset language (e.g., "english", "japanese").
        embeddings_manager: EmbeddingsManager instance for loading pretrained vectors.

    Returns:
        Embedding matrix of shape (vocab_size, embedding_size).
    """
    logger.info(
        f"Creating embeddings for language: {language}, type: {embedding_type}..."
    )

    if embedding_type == "glove" and language == "english":
        return _create_glove_embeddings(vocab, embedding_size, embeddings_manager)
    elif embedding_type == "bpemb":
        return _create_bpemb_embeddings(
            vocab, embedding_size, language, embeddings_manager
        )
    else:
        return _create_random_embeddings(vocab, embedding_size, language)


def _create_glove_embeddings(
    vocab: dict[str, int],
    embedding_size: int,
    embeddings_manager,
) -> np.ndarray:
    """Create embedding matrix using GloVe embeddings."""
    glove_tensor_tf, glove_vocab_map = embeddings_manager.load_glove_embeddings(
        embedding_size
    )
    if glove_tensor_tf is None or glove_vocab_map is None:
        raise ValueError("GloVe embeddings or vocab map could not be loaded.")

    glove_array = np.asarray(glove_tensor_tf)
    glove_mean_np = np.mean(glove_array, axis=0)
    glove_std_np = np.std(glove_array, axis=0)

    embedding_matrix = np.zeros((len(vocab), embedding_size), dtype=np.float32)
    embedding_matrix[vocab["[PAD]"]] = np.zeros(embedding_size, dtype=np.float32)
    embedding_matrix[vocab["[UNK]"]] = np.random.normal(
        loc=glove_mean_np, scale=glove_std_np, size=embedding_size
    ).astype(np.float32)

    if "<NUM>" in vocab:
        num_token_id = vocab["<NUM>"]
        glove_num_idx = glove_vocab_map.get("<NUM>")
        if glove_num_idx is not None:
            embedding_matrix[num_token_id] = glove_array[glove_num_idx]
        else:
            glove_number_idx = glove_vocab_map.get("number")
            if glove_number_idx is not None:
                embedding_matrix[num_token_id] = glove_array[glove_number_idx]
            else:
                embedding_matrix[num_token_id] = np.random.normal(
                    loc=glove_mean_np, scale=glove_std_np, size=embedding_size
                ).astype(np.float32)

    with create_progress(console=console) as progress:
        task = progress.add_task("Populating embedding matrix...", total=len(vocab))
        for word, idx in vocab.items():
            if word in ("[PAD]", "[UNK]", "<NUM>"):
                progress.advance(task)
                continue
            glove_word_idx = glove_vocab_map.get(word)
            if glove_word_idx is not None:
                embedding_matrix[idx] = glove_array[glove_word_idx]
            else:
                embedding_matrix[idx] = np.random.normal(
                    loc=glove_mean_np, scale=glove_std_np, size=embedding_size
                ).astype(np.float32)
            progress.advance(task)

    return embedding_matrix


def _create_bpemb_embeddings(
    vocab: dict[str, int],
    embedding_size: int,
    language: str,
    embeddings_manager,
) -> np.ndarray:
    """Create embedding matrix using BPEmb pre-trained embeddings."""
    logger.info(f"Creating BPEmb embeddings for language: {language}")

    lang_map = {
        "japanese": "ja",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "korean": "ko",
        "chinese": "zh",
        "arabic": "ar",
        "hindi": "hi",
        "turkish": "tr",
        "polish": "pl",
        "dutch": "nl",
        "english": "en",
    }
    lang_code = lang_map.get(language.lower(), language.lower())

    try:
        logger.info(f"Loading BPEmb embeddings for language: {lang_code}")
        bpemb_embeddings = embeddings_manager.get_bpemb_embeddings(
            language=lang_code, vocab_size=200000, dim=embedding_size
        )

        if not bpemb_embeddings:
            logger.warning(f"No BPEmb embeddings loaded for {lang_code}")
            return _create_random_embeddings(vocab, embedding_size, language)

        logger.info(
            f"Creating embedding matrix from {len(bpemb_embeddings):,} BPE tokens"
        )
        embedding_matrix = (
            np.random.randn(len(vocab), embedding_size).astype(np.float32) * 0.1
        )
        embedding_matrix[vocab["[PAD]"]] = np.zeros(embedding_size, dtype=np.float32)

        matched_words = 0
        for word, idx in vocab.items():
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

        match_pct = (matched_words / len(vocab)) * 100
        logger.info(
            f"Successfully created BPEmb embedding matrix: {embedding_matrix.shape}"
        )
        logger.info(f"Matched {matched_words}/{len(vocab)} words ({match_pct:.1f}%)")
        return embedding_matrix

    except Exception as e:
        logger.error(f"Failed to load BPEmb embeddings for {lang_code}: {e}")
        logger.warning("Falling back to random embeddings")
        return _create_random_embeddings(vocab, embedding_size, language)


def _create_random_embeddings(
    vocab: dict[str, int],
    embedding_size: int,
    language: str,
) -> np.ndarray:
    """Create random embedding matrix as fallback."""
    logger.info(f"Creating random embeddings for language: {language}")
    embedding_matrix = (
        np.random.randn(len(vocab), embedding_size).astype(np.float32) * 0.1
    )
    embedding_matrix[vocab["[PAD]"]] = np.zeros(embedding_size, dtype=np.float32)
    return embedding_matrix
