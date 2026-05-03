"""TCCM main model class (Keras) — TCCM-faithful.

Now uses the TCCM-specific encoders (:class:`TCCMNewsEncoder`,
:class:`TCCMPopularityEncoder`, :class:`TCCMActivityGater`) instead of
PP-Rec's. The user encoder is reused from PP-Rec because it already
matches the reference's ``popularity_user_modeling`` branch
(MHSA(20,20) over history news vectors + CPJA over [news ⨁ pop_emb]).

Final fusion (paper reference)::

    train: scores = 2 · η · rel + 2 · (1-η) · pop
    eval : scores = η · rel + (1-η) · pop

Causal intervention (``do(P)``) is applied at inference only when
``config.use_causal_intervention`` is set.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import keras
from keras import layers, ops

from src.core.models.configs import PPRecConfig, TCCMConfig
from src.frameworks.keras.models.base import BaseModel
from src.frameworks.keras.models.pprec import (
    ActivityGater,
    PPRecNewsEncoder,
    PPRecUserEncoder,
)

from .layers import TCCMPopularityEncoder


def _tccm_to_pprec_config(cfg: TCCMConfig) -> PPRecConfig:
    """Forward TCCM's news/user-encoder hyperparameters into a PPRecConfig.

    TCCM reuses PP-Rec's ``co1`` news encoder (5×40 cross-attn, Concat +
    Dense fusion) and popularity-aware user encoder (CPJA over [news ⨁
    pop_emb]) — these consistently outperformed the literally-faithful
    Add-residual / 20×20 cross-attn variant on MIND-small val/test.
    """
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
    """TCCM (CIKM 2023) — Keras implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: TCCMConfig | None = None,
        name: str = "tccm",
        **config_overrides,
    ):
        super().__init__(name=name)

        if config is None:
            config = TCCMConfig(**config_overrides)
        self.config = config
        self._pprec_config = _tccm_to_pprec_config(config)

        self.processed_news = processed_news
        self._validate_processed_news()
        self.process_user_id = config.process_user_id

        # TCCM follows the paper reference's ``attrs = ['title', 'entity']``
        # — no category branch.
        self._news_feature_dim = config.max_title_length
        if config.use_entity and "entity_indices" in processed_news:
            self._news_feature_dim += config.max_entities
        self._max_title_length = config.max_title_length
        self._max_entities = config.max_entities

        self.news_encoder = None
        self.user_encoder = None
        self.popularity_encoder = None
        self.activity_gater = None

        self.build(None)

    def build(self, input_shape) -> None:
        cfg = self.config
        pn = self.processed_news

        word_emb = layers.Embedding(
            input_dim=pn["vocab_size"],
            output_dim=cfg.embedding_size,
            embeddings_initializer=keras.initializers.Constant(pn["embeddings"]),
            trainable=True,
            name="word_embedding",
        )

        if not (cfg.use_entity and "entity_embeddings" in pn):
            raise ValueError(
                "TCCM requires entity embeddings (use_entity=True and "
                "'entity_embeddings' in processed_news)."
            )
        entity_emb = layers.Embedding(
            input_dim=pn["entity_vocab_size"],
            output_dim=cfg.entity_embedding_dim,
            embeddings_initializer=keras.initializers.Constant(
                pn["entity_embeddings"]
            ),
            trainable=True,
            name="entity_embedding",
        )

        # PP-Rec ``co1`` news encoder (no category branch — TCCM
        # follows the reference's ``attrs = ['title', 'entity']``).
        self.news_encoder = PPRecNewsEncoder(
            self._pprec_config, word_emb, entity_emb, None, name="news_encoder"
        )
        self.user_encoder = PPRecUserEncoder(self._pprec_config, self.news_encoder)
        self.popularity_encoder = TCCMPopularityEncoder(cfg)
        if cfg.use_activity_gate:
            self.activity_gater = ActivityGater(self._pprec_config)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(self, inputs, training=None):
        """Training forward pass — returns raw logits ``(B, C)``.

        Inference uses :attr:`news_encoder` / :attr:`user_encoder` /
        :attr:`popularity_encoder` directly via the TCCM evaluator.
        """
        return self.score_training_batch(inputs, training=training)

    def score_training_batch(self, inputs, training=None):
        """Score a training batch.

        Required keys in ``inputs``:

        * ``hist_tokens`` ``(B, H, F)``
        * ``cand_tokens`` ``(B, C, F)``
        * ``hist_ctr`` ``(B, H)`` int32 — discretised history-news CTR
        * ``cand_pop_buckets`` ``(B, C, T+E)`` int32 — per-token CTR buckets
        * ``cand_news_exist_time`` ``(B, C)`` int32 — clamped age in hours
        """
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_pop_buckets = inputs["cand_pop_buckets"]
        cand_exist_time = inputs["cand_news_exist_time"]

        if hist_ctr is not None:
            user_vec = self.user_encoder([hist_features, hist_ctr], training=training)
        else:
            user_vec = self.user_encoder(hist_features, training=training)

        B = ops.shape(cand_features)[0]
        C = ops.shape(cand_features)[1]
        F = ops.shape(cand_features)[2]
        flat_cand = ops.reshape(cand_features, (B * C, F))
        cand_vecs = ops.reshape(
            self.news_encoder(flat_cand, training=training), (B, C, -1)
        )

        user_expanded = ops.expand_dims(user_vec, axis=1)
        rel_scores = ops.sum(cand_vecs * user_expanded, axis=-1)

        T_plus_E = ops.shape(cand_pop_buckets)[2]
        flat_buckets = ops.reshape(cand_pop_buckets, (B * C, T_plus_E))
        flat_time = ops.reshape(cand_exist_time, (B * C,))
        pop_scores_flat = self.popularity_encoder(
            flat_buckets,
            flat_time,
            title_len=self._max_title_length,
            training=training,
        )
        pop_scores = ops.reshape(pop_scores_flat, (B, C))

        if self.config.use_activity_gate and self.activity_gater is not None:
            eta = self.activity_gater(user_vec, training=training)
            eta = ops.expand_dims(eta, axis=-1)
            scores = 2.0 * eta * rel_scores + 2.0 * (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores
        return scores

    def get_config(self):
        base_config = super().get_config()
        if is_dataclass(self.config):
            base_config.update(asdict(self.config))
        return base_config
