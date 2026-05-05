"""TCCM main model class (JAX/Flax NNX).

Reuses PP-Rec's ``co1`` news encoder and popularity-aware user encoder
from the JAX PP-Rec implementation. The popularity encoder and activity
gater are TCCM-specific.

Final fusion (paper reference)::

    train: scores = 2 · η · rel + 2 · (1-η) · pop
    eval : scores = η · rel + (1-η) · pop
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import PPRecConfig, TCCMConfig

from ..base import BaseModel
from ..pprec import PPRecNewsEncoder, PPRecUserEncoder
from .layers import TCCMActivityGater, TCCMPopularityEncoder


def _tccm_to_pprec_config(cfg: TCCMConfig) -> PPRecConfig:
    """Forward TCCM's news/user-encoder hyperparameters into a PPRecConfig."""
    return PPRecConfig(
        embedding_size=cfg.embedding_size,
        news_dim=cfg.news_dim,
        entity_embedding_dim=cfg.entity_embedding_dim,
        category_embedding_dim=cfg.category_embedding_dim,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        co_num_heads=cfg.co_num_heads,
        co_head_dim=cfg.co_head_dim,
        attention_hidden_dim=cfg.attention_hidden_dim,
        popularity_embedding_bins=cfg.popularity_embedding_bins,
        popularity_embedding_dim=cfg.popularity_embedding_dim,
        use_entity=cfg.use_entity,
        use_recency=False,
        use_ctr=False,
        use_activity_gate=cfg.use_activity_gate,
        dropout_rate=cfg.dropout_rate,
        seed=cfg.seed,
        max_title_length=cfg.max_title_length,
        max_history_length=cfg.max_history_length,
        max_impressions_length=cfg.max_impressions_length,
        max_entities=cfg.max_entities,
        process_user_id=cfg.process_user_id,
    )


class TCCM(BaseModel):
    """TCCM (CIKM 2023) — JAX/Flax NNX implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: TCCMConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **config_overrides,
    ):
        if config is None:
            config = TCCMConfig(**config_overrides)
        self.config = config
        self._pprec_config = _tccm_to_pprec_config(config)
        self.process_user_id = config.process_user_id
        self._max_title_length = config.max_title_length
        self._max_entities = config.max_entities

        pn = processed_news

        word_emb = nnx.Embed(
            num_embeddings=pn["vocab_size"],
            features=config.embedding_size,
            rngs=rngs,
        )
        word_emb.embedding.value = jnp.asarray(pn["embeddings"])

        if not (config.use_entity and "entity_embeddings" in pn):
            raise ValueError(
                "TCCM requires entity embeddings (use_entity=True and "
                "'entity_embeddings' in processed_news)."
            )
        entity_emb = nnx.Embed(
            num_embeddings=pn["entity_vocab_size"],
            features=config.entity_embedding_dim,
            rngs=rngs,
        )
        entity_emb.embedding.value = jnp.asarray(pn["entity_embeddings"])

        # PP-Rec ``co1`` news encoder (no category branch)
        self.news_encoder = PPRecNewsEncoder(
            self._pprec_config,
            word_emb,
            entity_emb,
            None,
            rngs=rngs,
        )
        self.user_encoder = PPRecUserEncoder(
            self._pprec_config,
            self.news_encoder,
            rngs=rngs,
        )
        self.popularity_encoder = TCCMPopularityEncoder(config, rngs=rngs)
        self.activity_gater = (
            TCCMActivityGater(config, rngs=rngs) if config.use_activity_gate else None
        )

    def __call__(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = False,
    ) -> jax.Array:
        return self.score_training_batch(inputs, training=training)

    def score_training_batch(
        self,
        inputs: dict[str, jax.Array],
        *,
        training: bool = True,
    ) -> jax.Array:
        """Score a training batch.

        Required keys in ``inputs``:
        * ``hist_tokens`` ``(B, H, F)``
        * ``cand_tokens`` ``(B, C, F)``
        * ``hist_ctr`` ``(B, H)`` int — discretised history-news CTR
        * ``cand_pop_buckets`` ``(B, C, T+E)`` int — per-token CTR buckets
        * ``cand_news_exist_time`` ``(B, C)`` int — clamped age in hours
        """
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_pop_buckets = inputs["cand_pop_buckets"]
        cand_exist_time = inputs["cand_news_exist_time"]

        user_vec = self.user_encoder(
            hist_features,
            hist_ctr,
            training=training,
        )

        B, C, F = cand_features.shape
        flat_cand = cand_features.reshape(B * C, F)
        cand_vecs = self.news_encoder(flat_cand, training=training).reshape(B, C, -1)
        rel_scores = jnp.sum(cand_vecs * user_vec[:, None, :], axis=-1)

        T_plus_E = cand_pop_buckets.shape[2]
        flat_buckets = cand_pop_buckets.reshape(B * C, T_plus_E)
        flat_time = cand_exist_time.reshape(B * C)
        pop_scores = self.popularity_encoder(
            flat_buckets,
            flat_time,
            title_len=self._max_title_length,
            training=training,
        ).reshape(B, C)

        if self.config.use_activity_gate and self.activity_gater is not None:
            eta = self.activity_gater(user_vec)[:, None]
            scores = 2.0 * eta * rel_scores + 2.0 * (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores
        return scores
