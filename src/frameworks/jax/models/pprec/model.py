"""PP-Rec top-level model (Flax NNX).

Mirror of the PyTorch PP-Rec.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import PPRecConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .popularity import ActivityGater, PopularityPredictor
from .user_encoder import UserEncoder


class PPRec(BaseModel):
    """PP-Rec Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: PPRecConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = PPRecConfig(**config_overrides)
        self.config = config
        self.process_user_id = config.process_user_id

        encoder_cfg = getattr(config, "encoder", None)

        rel_text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )
        bias_text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )

        entity_emb, category_emb = _build_aux_embeddings(
            processed_news, config, rngs=rngs
        )

        self.news_encoder = NewsEncoder(
            config, rel_text_encoder, entity_emb, category_emb, rngs=rngs
        )
        self.bias_news_encoder = NewsEncoder(
            config, bias_text_encoder, entity_emb, category_emb, rngs=rngs
        )
        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)
        self.popularity_predictor = PopularityPredictor(config, rngs=rngs)
        self.activity_gater = (
            ActivityGater(config, rngs=rngs) if config.use_activity_gate else None
        )

    def score_training_batch(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = True,
    ) -> jax.Array:
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_ctr = inputs.get("cand_ctr")
        cand_recency = inputs.get("cand_recency")

        user_vec = self.user_encoder(hist_features, hist_ctr, training=training)

        rel_cand_vecs = self.news_encoder(cand_features, training=training)
        bias_cand_vecs = self.bias_news_encoder(cand_features, training=training)
        rel_scores = jnp.sum(rel_cand_vecs * user_vec[:, None, :], axis=-1)

        B, C = cand_features.shape[:2]
        bias_flat = bias_cand_vecs.reshape(B * C, self.config.news_dim)
        recency_flat = cand_recency.reshape(B * C) if cand_recency is not None else None
        ctr_flat = cand_ctr.reshape(B * C) if cand_ctr is not None else None
        pop_scores = self.popularity_predictor(
            bias_flat, recency_indices=recency_flat, ctr_values=ctr_flat
        ).reshape(B, C)

        if self.activity_gater is not None:
            eta = self.activity_gater(user_vec)[:, None]
            scores = eta * rel_scores + (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores
        return scores

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        return self.score_training_batch(inputs, training=training)


def _build_aux_embeddings(
    processed_news: dict[str, Any],
    config: PPRecConfig,
    *,
    rngs: nnx.Rngs,
) -> tuple[nnx.Embed | None, nnx.Embed | None]:
    """Build entity + category embedding layers (or ``None`` when disabled)."""
    entity_emb: nnx.Embed | None = None
    if config.use_entity and "entity_embeddings" in processed_news:
        entity_matrix = np.asarray(processed_news["entity_embeddings"])
        entity_emb = nnx.Embed(
            num_embeddings=int(processed_news["entity_vocab_size"]),
            features=config.entity_embedding_dim,
            rngs=rngs,
        )
        entity_emb.embedding.value = jnp.asarray(entity_matrix)

    category_emb: nnx.Embed | None = None
    if "num_categories" in processed_news:
        category_emb = nnx.Embed(
            num_embeddings=int(processed_news["num_categories"]) + 1,
            features=config.category_embedding_dim,
            rngs=rngs,
        )

    return entity_emb, category_emb
