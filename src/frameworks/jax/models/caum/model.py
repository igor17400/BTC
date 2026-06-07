"""CAUM top-level model (Flax NNX).

Mirror of the PyTorch CAUM. Composes :class:`NewsEncoder`,
:class:`UserEncoder` (fallback mean-pool) and :class:`InterModel`
(candidate-aware scoring) with an injected :class:`TextEncoder`.

During training, ``score_training_batch`` encodes clicked + candidate
news once, then loops over each impression slot to score it via the
candidate-aware ``inter_model``. During evaluation, a custom evaluator
(see :mod:`src.core.models.evaluations.custom.caum`) precomputes news
vectors and runs ``inter_model`` per candidate.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import CAUMConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .inter_model import InterModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class CAUM(BaseModel):
    """CAUM — Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CAUMConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = CAUMConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=getattr(config, "encoder", None),
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )

        num_entities = _resolve_num_entities(processed_news)
        num_categories = int(processed_news.get("num_categories", 0))

        self.news_encoder = NewsEncoder(
            config,
            text_encoder,
            num_entities=num_entities,
            num_categories=num_categories,
            rngs=rngs,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)
        self.inter_model = InterModel(config, rngs=rngs)

    def score_training_batch(
        self,
        hist_packed: jax.Array,
        cand_packed: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score history × candidates with the candidate-aware inter_model.

        Args:
            hist_packed: ``(B, H, k)`` packed history features.
            cand_packed: ``(B, C, k)`` packed candidate features.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        clicked_vecs = self.news_encoder(
            hist_packed, training=training
        )  # (B, H, news_dim)
        cand_vecs = self.news_encoder(
            cand_packed, training=training
        )  # (B, C, news_dim)

        scores = []
        for i in range(self.config.max_impressions_length):
            cand_i = cand_vecs[:, i, :]
            score_i = self.inter_model(cand_i, clicked_vecs, training=training)
            scores.append(score_i)

        return jnp.stack(scores, axis=1)  # (B, C)

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        return self.score_training_batch(
            inputs["hist_tokens"], inputs["cand_tokens"], training=training
        )


def _resolve_num_entities(processed_news: dict[str, Any]) -> int:
    """Recover the entity vocab size from processed_news."""
    n = processed_news.get("num_entities")
    if n is not None:
        return int(n)
    entity_indices = processed_news.get("entity_indices")
    if entity_indices is not None:
        return int(np.max(entity_indices)) + 1
    return 1
