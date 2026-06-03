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

from ...layers.plm_token import FrozenCache
from ..base import BaseModel
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class _FrozenEmbed(nnx.Module):
    """Drop-in for ``nnx.Embed`` with a non-trainable embedding table.

    Holds the table as :class:`FrozenCache` (an ``nnx.Variable``
    subclass that is NOT an ``nnx.Param``), so the JAX optimizer —
    which uses ``wrt=nnx.Param`` — never updates it. Matches the
    PyTorch ``requires_grad=False`` semantics for CROWN's category
    embeddings (reference ``crown-www25/newsEncoders.py:33-35``).
    """

    def __init__(self, table: jax.Array):
        self.embedding = FrozenCache(table)

    def __call__(self, ids: jax.Array) -> jax.Array:
        return self.embedding.value[ids]


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

        # Category / subcategory embeddings — uniform-init then FROZEN
        # to match the reference repo
        # (``crown-www25/newsEncoders.py:33-35`` sets ``requires_grad =
        # False`` after a U(-0.1, 0.1) init, with ``subCategory[0]``
        # zeroed). _FrozenEmbed holds the table as :class:`FrozenCache`,
        # which the JAX optimizer (``wrt=nnx.Param``) skips.
        cat_table = jax.random.uniform(
            rngs.params(),
            (num_categories + 1, config.category_embedding_dim),
            minval=-0.1,
            maxval=0.1,
        )
        subcat_table = jax.random.uniform(
            rngs.params(),
            (num_subcategories + 1, config.subcategory_embedding_dim),
            minval=-0.1,
            maxval=0.1,
        )
        subcat_table = subcat_table.at[0].set(0.0)
        self.category_embedding = _FrozenEmbed(cat_table)
        self.subcategory_embedding = _FrozenEmbed(subcat_table)

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

        # Candidate-aware user representations ``(B, C, D)`` — every
        # candidate produces a different attended-history vector
        # (reference ``userEncoders.py:CROWN.forward``).
        user_per_cand = self.user_encoder.forward_with_candidates(
            history_packed=hist_packed,
            history_mask=history_mask,
            candidate_repr=cand_full,
            training=training,
        )

        return jnp.sum(user_per_cand * cand_full, axis=-1)

    def get_auxiliary_loss(self) -> jax.Array:
        return self.alpha * self.news_encoder._aux_loss.value


def _pack(inputs: dict[str, jax.Array], prefix: str) -> jax.Array:
    """Pack [news_idx | category | subcategory] columns into ``(*, 3)``."""
    news_idx = inputs[f"{prefix}_features"]
    category = inputs[f"{prefix}_category"]
    subcategory = inputs[f"{prefix}_subcategory"]
    return jnp.stack([news_idx, category, subcategory], axis=-1)
