"""CROWN top-level model (Flax NNX).

Reference: Ko et al., "CROWN: A Novel Approach to Comprehending Users'
Preferences for Accurate Personalized News Recommendation", WWW 2025.

Mirror of the PyTorch CROWN. Each text view (title, abstract) gets its
own :class:`TextEncoder` instance via :func:`build_text_encoder`; no
encoder-type branches inside the model body.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import CROWNConfig
from src.core.models.text_encoder import build_text_encoder

from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class CROWN(BaseModel):
    """CROWN: intent disentanglement + bipartite GNN user encoder."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CROWNConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if config is None:
            config = CROWNConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        num_subcategories = int(processed_news.get("num_subcategories", 100))
        encoder_cfg = getattr(config, "encoder", None)

        title_text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="title",
            rngs=rngs,
        )
        abstract_text_encoder = build_text_encoder(
            framework="jax",
            encoder_cfg=encoder_cfg,
            processed_news=processed_news,
            kind="abstract",
            rngs=rngs,
        )

        self.category_embedding = nnx.Embed(
            num_embeddings=num_categories + 1,
            features=config.category_embedding_dim,
            rngs=rngs,
        )
        self.category_embedding.embedding.value = jax.random.uniform(
            rngs.params(),
            (num_categories + 1, config.category_embedding_dim),
            minval=-0.1,
            maxval=0.1,
        )
        self.subcategory_embedding = nnx.Embed(
            num_embeddings=num_subcategories + 1,
            features=config.subcategory_embedding_dim,
            rngs=rngs,
        )
        self.subcategory_embedding.embedding.value = jax.random.uniform(
            rngs.params(),
            (num_subcategories + 1, config.subcategory_embedding_dim),
            minval=-0.1,
            maxval=0.1,
        )

        self.news_encoder = NewsEncoder(
            config,
            title_text_encoder,
            abstract_text_encoder,
            self.category_embedding,
            self.subcategory_embedding,
            rngs=rngs,
        )
        # Resize the category predictor now that we know num_categories.
        self.news_encoder.category_predictor = nnx.Linear(
            config.intent_embedding_dim,
            num_categories + 1,
            rngs=rngs,
        )

        self.user_encoder = UserEncoder(config, self.news_encoder, rngs=rngs)
        self.alpha = config.alpha

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        hist_packed = _pack(inputs, "hist")
        cand_packed = _pack(inputs, "cand")

        cand_full = self.news_encoder(
            cand_packed, compute_aux_loss=training, training=training
        )

        history_mask = self.news_encoder.valid_mask(hist_packed)

        user_repr = self.user_encoder.forward_with_candidates(
            history_packed=hist_packed,
            history_mask=history_mask,
            candidate_repr=cand_full,
            training=training,
        )

        return jnp.sum(user_repr[:, None, :] * cand_full, axis=-1)

    def get_auxiliary_loss(self) -> jax.Array:
        return self.alpha * self.news_encoder._aux_loss.value


def _pack(inputs: dict[str, jax.Array], prefix: str) -> jax.Array:
    """Pack [news_idx | category | subcategory] columns into ``(*, 3)``."""
    news_idx = inputs[f"{prefix}_features"]
    category = inputs[f"{prefix}_category"]
    subcategory = inputs[f"{prefix}_subcategory"]
    return jnp.stack([news_idx, category, subcategory], axis=-1)
