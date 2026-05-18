"""Pretrained Language Model (PLM) frozen-encoder utility.

Encodes a news corpus once with a HuggingFace model and caches the
resulting per-news vectors to disk.  Subsequent training runs read the
cached numpy array — no PLM forward at training time.

This is the v1 path (frozen, single-vector-per-news).  Future variants:
fine-tuned PLM, token-level embeddings.

Cache layout:
    .cache/plm_embeds/{plm_name}_{max_length}_{pooling}_{corpus_hash}.npz

Each cache file stores:
    - ``embeddings`` (num_news, plm_dim) float32
    - ``news_ids``   (num_news,) object — original news_id strings in row order

Loading round-trips through :func:`encode_corpus`; users get a
``(num_news, plm_dim)`` matrix plus the same ``news_ids`` list so the
caller can verify alignment against ``processed_news``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — keep names canonical so cache keys stay stable.
# ---------------------------------------------------------------------------

PLM_REGISTRY: dict[str, str] = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    # Sentence-BERT default: small & fast, designed for sentence-level pooling.
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
}


def resolve_plm_name(name_or_alias: str) -> str:
    """Map a short alias (``"bert"``) or a full HF identifier to the HF id."""
    return PLM_REGISTRY.get(name_or_alias, name_or_alias)


# ---------------------------------------------------------------------------
# Cache key construction
# ---------------------------------------------------------------------------


def _corpus_hash(news_ids: list[str], texts: list[str]) -> str:
    """Short stable hash over the corpus contents so different datasets get
    different cache files even if model + params are identical."""
    h = hashlib.sha256()
    h.update(str(len(news_ids)).encode())
    # Use first/last 32 ids + their texts — enough to disambiguate without
    # hashing the full corpus (which is fast anyway but redundant).
    for nid, txt in zip(news_ids[:32], texts[:32]):
        h.update(nid.encode())
        h.update(txt.encode("utf-8", errors="ignore"))
    for nid, txt in zip(news_ids[-32:], texts[-32:]):
        h.update(nid.encode())
        h.update(txt.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:10]


def cache_path(
    plm_name: str,
    max_length: int,
    pooling: str,
    corpus_hash: str,
    cache_root: Path | str = ".cache/plm_embeds",
) -> Path:
    """Canonical cache filename for a given (PLM, params, corpus) tuple."""
    safe_plm = plm_name.replace("/", "__")
    return Path(cache_root) / f"{safe_plm}_max{max_length}_{pooling}_{corpus_hash}.npz"


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def encode_corpus(
    *,
    news_ids: list[str],
    texts: list[str],
    plm_name: str,
    max_length: int = 32,
    pooling: Literal["mean", "cls"] = "mean",
    batch_size: int = 64,
    device: str = "cuda",
    cache_root: Path | str = ".cache/plm_embeds",
    force_recompute: bool = False,
) -> np.ndarray:
    """Encode a list of news texts into per-news frozen-PLM SENTENCE-level vectors.

    Sentence-level v1 path — does the pooling offline so the cache is
    compact ``(num_news, plm_dim)``. For configurable training-time
    pooling (mean / cls / learnable attention), see
    :func:`encode_corpus_tokens`.

    Args:
        news_ids: One identifier per text, stable across runs. Used for
            cache invalidation and alignment checks.
        texts: News texts to encode (one string per news item).
        plm_name: HF model identifier OR an alias from :data:`PLM_REGISTRY`
            (e.g. ``"bert"``, ``"roberta"``, ``"distilbert"``, ``"sbert"``).
        max_length: Tokenizer truncation length. 32 covers the median MIND
            title; bump to 64–128 if encoding title+abstract.
        pooling: ``"mean"`` (recommended, robust for non-fine-tuned models
            and required for Sentence-BERT) or ``"cls"`` (BERT-NSP style).
        batch_size: Inference batch size.
        device: ``"cuda"`` if available else ``"cpu"``.
        cache_root: Root directory for cache files.
        force_recompute: Skip the cache and recompute from scratch.

    Returns:
        ``(num_news, plm_dim)`` float32 numpy array, with rows in the same
        order as ``news_ids``.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    plm_hf_id = resolve_plm_name(plm_name)
    corpus_hash = _corpus_hash(news_ids, texts)
    path = cache_path(plm_hf_id, max_length, pooling, corpus_hash, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_recompute:
        logger.info(f"PLM cache hit: {path}")
        data = np.load(path, allow_pickle=True)
        cached_ids = data["news_ids"].tolist()
        if cached_ids != news_ids:
            logger.warning("PLM cache news_ids mismatch — falling back to recompute")
        else:
            return data["embeddings"].astype(np.float32)

    logger.info(
        f"PLM cache miss: encoding {len(news_ids)} news with {plm_hf_id} "
        f"(max_length={max_length}, pooling={pooling})..."
    )

    # Load model + tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(plm_hf_id)
    model = AutoModel.from_pretrained(plm_hf_id)
    model = model.to(device).eval()

    plm_dim = model.config.hidden_size
    out = np.zeros((len(news_ids), plm_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            hidden = outputs.last_hidden_state  # (B, T, D)

            if pooling == "cls":
                vecs = hidden[:, 0, :]  # (B, D)
            elif pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            out[start:end] = vecs.detach().cpu().numpy()

            if (start // batch_size) % 50 == 0:
                logger.info(
                    f"  Encoded {end}/{len(texts)} ({100 * end / len(texts):.1f}%)"
                )

    np.savez(
        path,
        embeddings=out,
        news_ids=np.array(news_ids, dtype=object),
    )
    logger.info(f"PLM cache written: {path} ({out.nbytes / 1e6:.1f} MB)")
    return out


def encode_corpus_tokens(
    *,
    news_ids: list[str],
    texts: list[str],
    plm_name: str,
    max_length: int = 32,
    batch_size: int = 64,
    device: str = "cuda",
    cache_root: Path | str = ".cache/plm_embeds",
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a corpus into per-news TOKEN-level frozen-PLM hidden states.

    Returns the full ``last_hidden_state`` for each news article, plus
    the tokenizer's ``attention_mask``. This lets downstream code apply
    a training-time pooler (mean / cls / learnable attention / gate)
    over the per-token vectors — the structure IP2 and PLM-NR use.

    Cache file: ``.cache/plm_embeds/{plm}_max{T}_TOKENS_{hash}.npz``
    (kept separate from the sentence-level cache to avoid collisions).

    Args:
        news_ids, texts, plm_name, max_length, batch_size, device,
            cache_root, force_recompute: As in :func:`encode_corpus`.

    Returns:
        Tuple of:
            - ``token_embeddings``: ``(num_news, max_length, plm_dim)`` float32.
            - ``attention_mask``: ``(num_news, max_length)`` int8 (0/1).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    plm_hf_id = resolve_plm_name(plm_name)
    corpus_hash = _corpus_hash(news_ids, texts)
    safe_plm = plm_hf_id.replace("/", "__")
    path = Path(cache_root) / f"{safe_plm}_max{max_length}_TOKENS_{corpus_hash}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_recompute:
        logger.info(f"PLM token-level cache hit: {path}")
        data = np.load(path, allow_pickle=True)
        cached_ids = data["news_ids"].tolist()
        if cached_ids != news_ids:
            logger.warning(
                "PLM token cache news_ids mismatch — falling back to recompute"
            )
        else:
            return (
                data["embeddings"].astype(np.float32),
                data["attention_mask"].astype(np.int8),
            )

    logger.info(
        f"PLM token-level cache miss: encoding {len(news_ids)} news with "
        f"{plm_hf_id} (max_length={max_length})..."
    )

    tokenizer = AutoTokenizer.from_pretrained(plm_hf_id)
    model = AutoModel.from_pretrained(plm_hf_id)
    model = model.to(device).eval()

    plm_dim = model.config.hidden_size
    embeddings = np.zeros((len(news_ids), max_length, plm_dim), dtype=np.float32)
    attention_mask = np.zeros((len(news_ids), max_length), dtype=np.int8)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            enc = tokenizer(
                texts[start:end],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            embeddings[start:end] = (
                outputs.last_hidden_state.detach().cpu().numpy()
            )  # (B, T, D)
            attention_mask[start:end] = (
                enc["attention_mask"].detach().cpu().numpy().astype(np.int8)
            )

            if (start // batch_size) % 50 == 0:
                logger.info(
                    f"  Encoded {end}/{len(texts)} ({100 * end / len(texts):.1f}%)"
                )

    np.savez(
        path,
        embeddings=embeddings,
        attention_mask=attention_mask,
        news_ids=np.array(news_ids, dtype=object),
    )
    logger.info(f"PLM token cache written: {path} ({embeddings.nbytes / 1e9:.2f} GB)")
    return embeddings, attention_mask


# ---------------------------------------------------------------------------
# Convenience: bind a cache file into a ``processed_news`` dict
# ---------------------------------------------------------------------------


def _parse_news_id(news_id_str: str, id_prefix: str = "N") -> int:
    """MIND-style parsing: ``"N123"`` → 123. Mirrors
    :meth:`NewsDatasetBase.parse_news_id` for prefix-stripping IDs.
    """
    if id_prefix and news_id_str.startswith(id_prefix):
        return int(news_id_str.split(id_prefix)[1])
    return int(news_id_str)


def attach_plm_embeddings(
    processed_news: dict,
    *,
    plm_name: str,
    text_field: str = "title",
    max_length: int = 32,
    pooling: Literal["mean", "cls"] = "mean",
    batch_size: int = 64,
    device: str = "cuda",
    cache_root: Path | str = ".cache/plm_embeds",
    id_prefix: str = "N",
    level: Literal["sentence", "token"] = "sentence",
) -> None:
    """Encode the given text field for every news item and attach a
    dense ``(max_parsed_id + 1, plm_dim)`` lookup table to
    ``processed_news`` under ``plm_embeddings_by_id``.

    The lookup table is indexed by the **parsed integer news id** (the
    numeric suffix of ``"N123"``), matching the convention used in
    ``train_behaviors_data["histories_news_ids"]`` and
    ``["candidate_news_ids"]``. Row ``0`` is reserved for padding (zero
    vector) since MIND parsed ids start at 1.

    Also stores:
        - ``plm_embeddings``: the compact ``(num_news, plm_dim)`` matrix
          in ``news_ids_original_strings`` row order, kept for any
          downstream code that wants the per-news view.
        - ``plm_dim``: integer hidden dim.

    Args:
        id_prefix: ID prefix to strip when parsing news_id strings.
            Defaults to ``"N"`` (MIND). Pass ``""`` for prefix-less
            datasets like the Japanese corpus.
    """
    news_ids = list(processed_news["news_ids_original_strings"])

    raw_key = f"raw_{text_field}s"
    if raw_key not in processed_news:
        raise KeyError(
            f"processed_news is missing '{raw_key}'. The data pipeline "
            f"needs to keep the raw text for PLM encoding. "
            f"See src.core.data.processing.text.news.process_news."
        )
    texts = list(processed_news[raw_key])
    if len(texts) != len(news_ids):
        raise ValueError(
            f"Length mismatch: {len(news_ids)} news_ids vs {len(texts)} {raw_key}"
        )

    parsed_ids = [_parse_news_id(nid, id_prefix) for nid in news_ids]
    max_parsed = max(parsed_ids)

    if level == "sentence":
        compact = encode_corpus(
            news_ids=news_ids,
            texts=texts,
            plm_name=plm_name,
            max_length=max_length,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            cache_root=cache_root,
        )
        plm_dim = compact.shape[1]
        by_id = np.zeros((max_parsed + 1, plm_dim), dtype=np.float32)
        for row, pid in enumerate(parsed_ids):
            by_id[pid] = compact[row]
        processed_news["plm_embeddings"] = compact
        processed_news["plm_embeddings_by_id"] = by_id
        processed_news["plm_dim"] = int(plm_dim)
        logger.info(
            f"Built PLM sentence-level lookup: ({by_id.shape[0]:,}, {plm_dim}) "
            f"= {by_id.nbytes / 1e6:.1f} MB (parsed_id 0 = padding)"
        )
    elif level == "token":
        token_emb, attn_mask = encode_corpus_tokens(
            news_ids=news_ids,
            texts=texts,
            plm_name=plm_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            cache_root=cache_root,
        )
        plm_dim = token_emb.shape[-1]
        # Dense lookup indexed by parsed news id; row 0 is padding (zeros).
        by_id = np.zeros((max_parsed + 1, max_length, plm_dim), dtype=np.float32)
        mask_by_id = np.zeros((max_parsed + 1, max_length), dtype=np.int8)
        for row, pid in enumerate(parsed_ids):
            by_id[pid] = token_emb[row]
            mask_by_id[pid] = attn_mask[row]
        processed_news["plm_token_embeddings_by_id"] = by_id
        processed_news["plm_attention_mask_by_id"] = mask_by_id
        processed_news["plm_dim"] = int(plm_dim)
        processed_news["plm_max_length"] = int(max_length)
        logger.info(
            f"Built PLM token-level lookup: "
            f"({by_id.shape[0]:,}, {max_length}, {plm_dim}) "
            f"= {by_id.nbytes / 1e9:.2f} GB (parsed_id 0 = padding)"
        )
    else:
        raise ValueError(f"Unknown level: {level!r}. Use 'sentence' or 'token'.")


__all__ = [
    "PLM_REGISTRY",
    "attach_plm_embeddings",
    "cache_path",
    "encode_corpus",
    "resolve_plm_name",
]
