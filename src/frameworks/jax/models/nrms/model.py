"""NRMS top-level model (Flax NNX).

Reference: Wu et al., "Neural News Recommendation with Multi-Head
Self-Attention", EMNLP 2019.

Composes :class:`NewsEncoder` and :class:`UserEncoder` with an injected
:class:`TextEncoder` (GloVe or PLM, picked by the runner-provided
``encoder`` config). Single code path regardless of encoder type.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import NRMSConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class NRMS(BaseModel):
    """NRMS — Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NRMSConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = NRMSConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=getattr(config, "encoder", None),
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )

        self.news_encoder = NewsEncoder(config, text_encoder, rngs=rngs)
        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)

    def score_training_batch(
        self,
        history_features: jax.Array,
        candidate_features: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score a training batch.

        Args:
            history_features: ``(B, H)`` parsed news ids.
            candidate_features: ``(B, C)`` parsed news ids.
            training: Enable dropout.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        user_repr = self.user_encoder(history_features, training=training)  # (B, E)
        cand_repr = self.news_encoder(
            candidate_features, training=training
        )  # (B, C, E)
        return jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Training forward; inference uses ``news_encoder`` and
        ``user_encoder`` directly via the shared evaluator."""
        return self.score_training_batch(
            inputs["hist_features"], inputs["cand_features"], training=training
        )
