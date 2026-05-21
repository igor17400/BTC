"""DIGAT (EMNLP 2022 Findings) — JAX/Flax NNX implementation.

Dual Interactive Graph Attention Networks for news recommendation.
Two interacting graph channels (news-SAG + user-topic) co-evolve across
``graph_depth`` layers.  The SAG (Semantic Augmented Graph) is precomputed
offline; the user-topic graph is built per behavior from category
assignments.

Structurally identical to the PyTorch version in
:mod:`src.frameworks.pytorch.models.digat.model` — the two ports share
identical module layout, method names, and forward-pass shapes so
evaluation results are reproducible across frameworks.

Reference: Mao et al., "DIGAT: Modeling News Recommendation with
Dual-Graph Interaction", EMNLP 2022 Findings.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.core.models.configs import DIGATConfig

from ...layers import PLMTokenLookup
from ..base import BaseModel
from .graph_encoder import GraphEncoder
from .news_encoder import NewsEncoder


class DIGAT(BaseModel):
    """DIGAT: Dual Interactive Graph Attention Networks.

    Training: encode candidates (with SAG) + history → dual graph
    interaction → dot-product scoring.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: DIGATConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if config is None:
            config = DIGATConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        self.num_categories = num_categories + 1  # +1 for padding category
        encoder_type = getattr(getattr(config, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        if encoder_type == "glove":
            vocab_size = int(processed_news["vocab_size"])
            embeddings_matrix = np.asarray(processed_news["embeddings"])
            self.word_embedding: nnx.Module = nnx.Embed(
                num_embeddings=vocab_size,
                features=config.embedding_size,
                rngs=rngs,
            )
            self.word_embedding.embedding.value = jnp.asarray(embeddings_matrix)
        else:
            if "plm_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"DIGAT with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id']."
                )
            plm_dim = int(processed_news["plm_dim"])
            self.word_embedding = PLMTokenLookup(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                plm_dim=plm_dim,
                output_dim=config.embedding_size,
                rngs=rngs,
            )

        self.news_encoder = NewsEncoder(
            config, self.word_embedding, rngs=rngs, encoder_type=encoder_type
        )

        self.graph_encoder = GraphEncoder(
            config,
            self.num_categories,
            rngs=rngs,
        )

        # No user encoder in the evaluation adapter sense — DIGAT's
        # evaluation goes through ``score_digat_impression`` directly.
        # The BaseModel contract fields are set to satisfy attribute
        # access; the standard evaluator is not used for DIGAT.
        self.user_encoder = None

        self.news_graph_size = config.news_graph_size
        self.max_history = config.max_history_length
        self.D = config.news_embedding_dim

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        """Training forward pass.

        Expected input keys:
            hist_tokens: ``(batch_size, hist_len, title_len)``
            hist_mask: ``(batch_size, hist_len, title_len)``
            user_graph: ``(batch_size, user_graph_size, user_graph_size)``
            user_category_mask: ``(batch_size, num_categories)``
            user_category_indices: ``(batch_size, hist_len)``
            cand_tokens: ``(batch_size, num_cands, sag_size, title_len)``
            cand_mask: ``(batch_size, num_cands, sag_size, title_len)``
            cand_graph: ``(batch_size, num_cands, sag_size, sag_size)``
            cand_graph_mask: ``(batch_size, num_cands, sag_size)``

        Returns:
            ``(batch_size, num_cands)`` raw logits.
        """
        batch_size, num_cands = inputs["cand_graph"].shape[:2]
        sag_size = self.news_graph_size
        user_graph_size = self.max_history + self.num_categories
        batch_cands = batch_size * num_cands

        # Flatten candidates: shape depends on encoder.
        # GloVe: ``(B, C, sag_size, T)`` ids → ``(B*C, sag_size, T)``.
        # PLM:   ``(B, C, sag_size)`` parsed news_idx → ``(B*C, sag_size)``.
        if self.encoder_type == "glove":
            cand_tokens = inputs["cand_tokens"].reshape(batch_cands, sag_size, -1)
            cand_mask = (
                inputs["cand_mask"].reshape(batch_cands, sag_size, -1)
                if "cand_mask" in inputs
                else None
            )
        else:
            cand_tokens = inputs["cand_tokens"].reshape(batch_cands, sag_size)
            cand_mask = None
        cand_graph = inputs["cand_graph"].reshape(batch_cands, sag_size, sag_size)
        cand_graph_mask = inputs["cand_graph_mask"].reshape(batch_cands, sag_size)

        # Expand user data: (batch_size, ...) → (batch_cands, ...)
        user_graph = jnp.broadcast_to(
            inputs["user_graph"][:, None],
            (batch_size, num_cands, user_graph_size, user_graph_size),
        ).reshape(batch_cands, user_graph_size, user_graph_size)
        user_cat_mask = jnp.broadcast_to(
            inputs["user_category_mask"][:, None],
            (batch_size, num_cands, self.num_categories),
        ).reshape(batch_cands, self.num_categories)
        user_cat_indices = jnp.broadcast_to(
            inputs["user_category_indices"][:, None],
            (batch_size, num_cands, self.max_history),
        ).reshape(batch_cands, self.max_history)

        # Encode candidate news (each with SAG neighbors)
        cand_emb = self.news_encoder(cand_tokens, cand_mask, training=training)

        # Encode user history
        user_news_emb = self.news_encoder(
            inputs["hist_tokens"],
            inputs.get("hist_mask"),
            training=training,
        )
        user_news_emb = jnp.broadcast_to(
            user_news_emb[:, None],
            (batch_size, num_cands, self.max_history, self.D),
        ).reshape(batch_cands, self.max_history, self.D)

        # Dual graph interaction
        news_ctx, user_ctx = self.graph_encoder(
            cand_emb,
            cand_graph,
            cand_graph_mask,
            user_news_emb,
            user_graph,
            user_cat_mask,
            user_cat_indices,
            self.num_categories,
            training=training,
        )

        # Dot-product scoring
        news_ctx = news_ctx.reshape(batch_size, num_cands, self.D)
        user_ctx = user_ctx.reshape(batch_size, num_cands, self.D)
        return jnp.sum(news_ctx * user_ctx, axis=-1)
