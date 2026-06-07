"""LSTUR top-level model (Flax NNX).

Mirror of the PyTorch LSTUR. Composes :class:`NewsEncoder` and
:class:`UserEncoder` with an injected :class:`TextEncoder` (GloVe or
PLM, picked by the runner-provided ``encoder`` config). Single code
path regardless of encoder type.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import LSTURConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class LSTUR(BaseModel):
    """LSTUR — Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        num_users: int,
        config: LSTURConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = LSTURConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id
        self.num_users = num_users

        text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=getattr(config, "encoder", None),
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )

        self.news_encoder = NewsEncoder(
            config,
            text_encoder,
            num_categories=int(processed_news.get("num_categories", 0)),
            num_subcategories=int(processed_news.get("num_subcategories", 0)),
            rngs=rngs,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder, num_users, rngs=rngs)

    def score_training_batch(
        self,
        history_packed: jax.Array,
        user_ids: jax.Array,
        candidate_packed: jax.Array,
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score history × candidates.

        Args:
            history_packed: ``(B, H)`` plain news_idx or ``(B, H, k)`` packed.
            user_ids: ``(B,)`` or ``(B, 1)`` int.
            candidate_packed: ``(B, C)`` plain news_idx or ``(B, C, k)`` packed.

        Returns:
            ``(B, C)`` raw logit scores.
        """
        user_repr = self.user_encoder(
            history_packed, user_ids, training=training
        )  # (B, D)
        cand_repr = self.news_encoder(candidate_packed, training=training)  # (B, C, D)
        return jnp.sum(cand_repr * user_repr[:, None, :], axis=-1)

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Training forward; inference uses ``news_encoder`` and
        ``user_encoder`` directly via the shared evaluator."""
        history_packed = _pack(inputs, "hist", self.config)
        candidate_packed = _pack(inputs, "cand", self.config)
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))
        return self.score_training_batch(
            history_packed, user_ids, candidate_packed, training=training
        )


def _pack(inputs: dict[str, jax.Array], prefix: str, config: LSTURConfig) -> jax.Array:
    """Pack ``[news_idx | category | subcategory]`` columns from the inputs dict.

    Returns plain ``news_idx`` when both category flags are disabled —
    matching the eval dataloader's single-view shape so the news encoder
    receives the same layout at train and eval time.
    """
    news_idx = inputs[f"{prefix}_features"]
    if not config.use_category and not config.use_subcategory:
        return news_idx

    parts: list[jax.Array] = [jnp.expand_dims(news_idx, axis=-1)]
    if config.use_category:
        parts.append(jnp.expand_dims(inputs[f"{prefix}_category"], axis=-1))
    if config.use_subcategory:
        parts.append(jnp.expand_dims(inputs[f"{prefix}_subcategory"], axis=-1))
    return jnp.concatenate(parts, axis=-1)
