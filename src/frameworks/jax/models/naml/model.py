"""NAML top-level model (Flax NNX).

Mirror of the PyTorch NAML. Composes :class:`NewsEncoder` and
:class:`UserEncoder` with injected :class:`TextEncoder` instances
(one per text view). Single code path regardless of encoder type.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import NAMLConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class NAML(BaseModel):
    """NAML — Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: NAMLConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = NAMLConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        encoder_cfg = getattr(config, "encoder", None)
        title_text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )
        # Abstract view is optional: when process_abstract is False the data
        # pipeline produces no abstract cache, so skip the encoder entirely
        # (the news encoder then runs on title + category + subcategory).
        abstract_text_encoder = (
            build_text_encoder(
                framework="jax",
                encoder_cfg=encoder_cfg,
                processed_news=processed_news,
                kind="abstract",
                rngs=rngs,
            )
            if config.process_abstract
            else None
        )

        self.news_encoder = NewsEncoder(
            config,
            title_text_encoder,
            abstract_text_encoder,
            num_categories=int(processed_news["num_categories"]),
            num_subcategories=int(processed_news["num_subcategories"]),
            rngs=rngs,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)

    def score_training_batch(
        self,
        hist_packed: jax.Array,
        cand_packed: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score history × candidates.

        Args:
            hist_packed: ``(B, H, 3)`` packed history features.
            cand_packed: ``(B, C, 3)`` packed candidate features.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        user_repr = self.user_encoder(hist_packed, training=training)  # (B, D)
        cand_repr = self.news_encoder(cand_packed, training=training)  # (B, C, D)
        return jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Training forward; inference uses ``news_encoder`` and
        ``user_encoder`` directly via the shared evaluator."""
        hist_packed = _pack(inputs, "hist")
        cand_packed = _pack(inputs, "cand")
        return self.score_training_batch(hist_packed, cand_packed, training=training)


def _pack(inputs: dict[str, jax.Array], prefix: str) -> jax.Array:
    """Pack [news_idx | category | subcategory] columns from the inputs dict."""
    news_idx = inputs[f"{prefix}_features"]
    category = inputs[f"{prefix}_category"]
    subcategory = inputs[f"{prefix}_subcategory"]
    return jnp.stack([news_idx, category, subcategory], axis=-1)
