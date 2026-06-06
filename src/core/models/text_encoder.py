"""Factory that builds a framework-specific :class:`TextEncoder`.

The factory is the single place that decides — based on the Hydra
``cfg.encoder`` block — whether a model receives a GloVe word-lookup
encoder or a frozen-cached PLM encoder. Models call this factory once
in their constructor and from then on operate against a uniform
``text_encoder(news_idx) -> (tokens, mask)`` interface.

Per-framework implementations live in
:mod:`src.frameworks.pytorch.layers.text_encoder` and
:mod:`src.frameworks.jax.layers.text_encoder`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from omegaconf import DictConfig

# Per-news word-id matrices written by
# :func:`src.core.data.processing.text.news.process_news_data`. These
# are ROW-indexed by position in ``news_ids_original_strings``; we
# re-index them by parsed news id below so the GloVe TextEncoder shares
# the same lookup convention as the PLM cache.
_GLOVE_WORDS_KEY = {"title": "tokens", "abstract": "abstract_tokens"}

# PLM cache keys written by
# :func:`src.core.data.encoders.plm.attach_plm_embeddings`. The
# ``output_prefix`` arg is inserted between ``plm_`` and the rest of
# the key, so ``output_prefix="abstract_"`` produces
# ``plm_abstract_token_embeddings_by_id`` (NOT
# ``abstract_plm_token_embeddings_by_id``).
_PLM_TOKENS_KEY = {
    "title": "plm_token_embeddings_by_id",
    "abstract": "plm_abstract_token_embeddings_by_id",
}
_PLM_MASK_KEY = {
    "title": "plm_attention_mask_by_id",
    "abstract": "plm_abstract_attention_mask_by_id",
}


def build_text_encoder(
    framework: Literal["pytorch", "jax"],
    encoder_cfg: DictConfig | None,
    processed_news: dict[str, Any],
    *,
    kind: Literal["title", "abstract"] = "title",
    rngs: Any = None,
):
    """Build a TextEncoder (GloVe or PLM) for the requested text kind.

    The returned encoder's ``output_dim`` is the **native** text dim:
      - GloVe: ``encoder_cfg.embedding_size`` (300 by default).
      - PLM:   ``plm_dim`` from the cache shape (768 for BERT-base, ...).

    Models read ``text_encoder.output_dim`` to size their per-news
    MHSA, then apply their own post-pool projection if their internal
    ``news_dim`` differs from the text dim. This is the IP2-style
    design used by the reference NRMS+PLM literature.

    Args:
        framework: ``"pytorch"`` or ``"jax"``.
        encoder_cfg: Top-level ``cfg.encoder``. ``None`` falls back to
            GloVe with default ``embedding_size=300``.
        processed_news: Dataset-init output. Must contain
            ``"tokens"`` / ``"abstract_tokens"`` for GloVe and
            ``"plm_token_embeddings_by_id"`` /
            ``"abstract_plm_token_embeddings_by_id"`` (+ matching mask
            keys) for PLM.
        kind: ``"title"`` or ``"abstract"`` — which cached view to read.
        rngs: ``flax.nnx.Rngs`` for the JAX path. Ignored by PyTorch.
    """
    enc_type = encoder_cfg.type if encoder_cfg is not None else "glove"
    framework = framework.lower()
    if framework not in ("pytorch", "jax"):
        raise ValueError(
            f"Unsupported framework {framework!r}. Use 'pytorch' or 'jax'."
        )

    if enc_type == "glove":
        embedding_dim = (
            int(encoder_cfg.embedding_size) if encoder_cfg is not None else 300
        )
        return _build_glove(framework, processed_news, embedding_dim, kind, rngs)

    # All non-glove types share the same frozen-cache path; output_dim
    # is the cache's native plm_dim (no projection in the encoder).
    return _build_plm(framework, processed_news, kind, rngs)


def _build_glove(
    framework: str,
    processed_news: dict[str, Any],
    embedding_dim: int,
    kind: str,
    rngs: Any,
):
    key = _GLOVE_WORDS_KEY[kind]
    if key not in processed_news:
        raise KeyError(
            f"GloVe text encoder for kind={kind!r} requires "
            f"processed_news[{key!r}] (per-news word-id matrix)."
        )
    word_id_by_id = _word_ids_by_parsed_id(processed_news, key)
    vocab_size = int(processed_news["vocab_size"])
    # Title and abstract share the same word vocabulary, so both views must be
    # initialised from the pretrained GloVe matrix.
    pretrained = processed_news.get("embeddings")

    if framework == "pytorch":
        from src.frameworks.pytorch.layers.text_encoder import GloveTextEncoder

        return GloveTextEncoder(
            word_id_by_id,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_embeddings=pretrained,
        )

    from src.frameworks.jax.layers.text_encoder import GloveTextEncoder as JaxGlove

    return JaxGlove(
        word_id_by_id,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        pretrained_embeddings=pretrained,
        rngs=rngs,
    )


def _word_ids_by_parsed_id(processed_news: dict[str, Any], key: str) -> np.ndarray:
    """Re-index a row-keyed word-id matrix by parsed news id.

    ``processed_news[key]`` is ``(num_news, T)`` int, indexed by row
    position in ``news_ids_original_strings``. The model receives
    parsed news ids (e.g. 123 for "N123") and the PLM cache is already
    parsed-id-indexed, so this function expands the row-keyed table to
    a parsed-id-keyed table ``(max_parsed_id + 1, T)`` to match. Row 0
    stays all-zero (padding) and any unused parsed ids are zero rows.
    """
    row_table = np.asarray(processed_news[key])
    num_news, T = row_table.shape
    news_ids_str = processed_news["news_ids_original_strings"]
    parsed_ids = np.array(
        [
            int(s[1:]) if isinstance(s, str) and s.startswith("N") else int(s)
            for s in news_ids_str
        ],
        dtype=np.int64,
    )
    if len(parsed_ids) != num_news:
        raise ValueError(
            f"news_ids_original_strings length {len(parsed_ids)} != num_news {num_news}"
        )
    max_id = int(parsed_ids.max()) if num_news > 0 else 0
    by_id = np.zeros((max_id + 1, T), dtype=row_table.dtype)
    by_id[parsed_ids] = row_table
    return by_id


def _build_plm(
    framework: str,
    processed_news: dict[str, Any],
    kind: str,
    rngs: Any,
):
    tokens_key = _PLM_TOKENS_KEY[kind]
    mask_key = _PLM_MASK_KEY[kind]
    if tokens_key not in processed_news or mask_key not in processed_news:
        raise KeyError(
            f"PLM text encoder for kind={kind!r} requires "
            f"processed_news[{tokens_key!r}] and processed_news[{mask_key!r}]. "
            f"Call attach_plm_embeddings(..., level='token'"
            + (", output_prefix='abstract_'" if kind == "abstract" else "")
            + ") in the runner."
        )
    cache = processed_news[tokens_key]
    mask = processed_news[mask_key]

    if framework == "pytorch":
        from src.frameworks.pytorch.layers.text_encoder import PLMTextEncoder

        return PLMTextEncoder(cache, mask)

    from src.frameworks.jax.layers.text_encoder import PLMTextEncoder as JaxPLM

    return JaxPLM(cache, mask)


__all__ = ["build_text_encoder"]
