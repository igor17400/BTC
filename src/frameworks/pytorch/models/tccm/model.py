"""TCCM main model class (PyTorch) — TCCM-faithful.

Now uses the TCCM-specific encoders (:class:`TCCMNewsEncoder`,
:class:`TCCMPopularityEncoder`, :class:`TCCMActivityGater`) instead of
PP-Rec's. The user encoder is reused from PP-Rec because it already
matches the reference's ``popularity_user_modeling`` branch.

Final fusion (paper reference)::

    train: scores = 2 · η · rel + 2 · (1-η) · pop
    eval : scores = η · rel + (1-η) · pop
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from src.core.models.configs import PPRecConfig, TCCMConfig

from ...layers import PLMTokenLookup
from ..base import BaseModel
from ..pprec import PPRecNewsEncoder, PPRecUserEncoder
from .layers import TCCMActivityGater, TCCMPopularityEncoder


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
    """TCCM (CIKM 2023) — PyTorch implementation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: TCCMConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = TCCMConfig(**config_overrides)
        self.config = config
        self._pprec_config = _tccm_to_pprec_config(config)
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id
        self._max_title_length = config.max_title_length
        self._max_entities = config.max_entities

        cfg = config
        pn = processed_news
        encoder_type = getattr(getattr(cfg, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        if encoder_type == "glove":
            word_emb: nn.Module = nn.Embedding(pn["vocab_size"], cfg.embedding_size)
            word_emb.weight = nn.Parameter(
                torch.tensor(pn["embeddings"], dtype=torch.float32)
            )
        else:
            if "plm_token_embeddings_by_id" not in pn:
                raise KeyError(
                    f"TCCM with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id']."
                )
            plm_dim = int(pn["plm_dim"])
            word_emb = PLMTokenLookup(
                pn["plm_token_embeddings_by_id"],
                pn["plm_attention_mask_by_id"],
                plm_dim=plm_dim,
                output_dim=cfg.embedding_size,
            )

        if not (cfg.use_entity and "entity_embeddings" in pn):
            raise ValueError(
                "TCCM requires entity embeddings (use_entity=True and "
                "'entity_embeddings' in processed_news)."
            )
        entity_emb = nn.Embedding(pn["entity_vocab_size"], cfg.entity_embedding_dim)
        entity_emb.weight = nn.Parameter(
            torch.tensor(pn["entity_embeddings"], dtype=torch.float32)
        )

        # PP-Rec ``co1`` news encoder (no category branch — TCCM
        # follows the reference's ``attrs = ['title', 'entity']``).
        # Popularity branch (below) stays keyed on GloVe-token-CTR
        # regardless of encoder.type — the relevance encoder is the
        # only branch that consumes PLM features.
        self.news_encoder = PPRecNewsEncoder(
            self._pprec_config, word_emb, entity_emb, None, encoder_type=encoder_type
        )
        self.user_encoder = PPRecUserEncoder(self._pprec_config, self.news_encoder)
        self.popularity_encoder = TCCMPopularityEncoder(cfg)
        self.activity_gater = TCCMActivityGater(cfg) if cfg.use_activity_gate else None

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        return self.score_training_batch(inputs)

    def score_training_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score a training batch.

        Required keys in ``inputs``:

        * ``hist_tokens`` ``(B, H, F)``
        * ``cand_tokens`` ``(B, C, F)``
        * ``hist_ctr`` ``(B, H)`` long — discretised history-news CTR
        * ``cand_pop_buckets`` ``(B, C, T+E)`` long — per-token CTR buckets
        * ``cand_news_exist_time`` ``(B, C)`` long — clamped age in hours
        """
        hist_features = inputs["hist_tokens"]
        cand_features = inputs["cand_tokens"]
        hist_ctr = inputs.get("hist_ctr")
        cand_pop_buckets = inputs["cand_pop_buckets"]
        cand_exist_time = inputs["cand_news_exist_time"]

        user_vec = self.user_encoder(hist_features, hist_ctr, training=True)

        B, C, F = cand_features.shape
        flat_cand = cand_features.reshape(B * C, F)
        cand_vecs = self.news_encoder(flat_cand, training=True).reshape(B, C, -1)
        rel_scores = torch.sum(cand_vecs * user_vec.unsqueeze(1), dim=-1)

        T_plus_E = cand_pop_buckets.shape[2]
        flat_buckets = cand_pop_buckets.reshape(B * C, T_plus_E)
        flat_time = cand_exist_time.reshape(B * C)
        pop_scores = self.popularity_encoder(
            flat_buckets, flat_time, title_len=self._max_title_length
        ).reshape(B, C)

        if self.config.use_activity_gate and self.activity_gater is not None:
            eta = self.activity_gater(user_vec).unsqueeze(-1)
            scores = 2.0 * eta * rel_scores + 2.0 * (1.0 - eta) * pop_scores
        else:
            scores = rel_scores + pop_scores
        return scores

    def get_config(self) -> dict:
        return asdict(self.config)
