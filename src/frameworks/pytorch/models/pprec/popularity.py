"""PP-Rec popularity predictor + activity gater (PyTorch).

Two modules used at the score-fusion stage:

* :class:`PopularityPredictor` — content + recency + CTR scorer driven
  by the *bias* news encoder. Computes ``content_score`` from the bias
  news vector, ``recency_score`` from a recency embedding, then a gate
  blends them. A separate linear scaler converts raw CTR into an
  additive score term.
* :class:`ActivityGater` — per-user gate ``η ∈ [0, 1]`` that balances
  relevance vs popularity in the final fused score.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import PPRecConfig


class PopularityPredictor(nn.Module):
    """Time-aware news popularity predictor.

    Heads:
        content: news_dim -> pop_content_dims (3 hidden) -> 1
        recency: recency_emb_dim -> pop_recency_dims (2 hidden) -> 1
        gate:    concat[news_vec, recency_emb] -> pop_gate_dims (2 hidden)
                 -> 1 (sigmoid)
        ctr:     Linear(1, 1, bias=False) initialised to ``ctr_scaler_init``
    """

    def __init__(self, config: PPRecConfig):
        super().__init__()
        self.config = config

        c1, c2, c3 = config.pop_content_dims
        self.content_dense1 = nn.Linear(config.news_dim, c1)
        self.content_dense2 = nn.Linear(c1, c2)
        self.content_dense3 = nn.Linear(c2, c3)
        self.content_out = nn.Linear(c3, 1, bias=False)

        if config.use_recency:
            r1, r2 = config.pop_recency_dims
            g1, g2 = config.pop_gate_dims
            self.recency_embedding = nn.Embedding(
                config.recency_embedding_bins, config.recency_embedding_dim
            )
            self.recency_dense1 = nn.Linear(config.recency_embedding_dim, r1)
            self.recency_dense2 = nn.Linear(r1, r2)
            self.recency_out = nn.Linear(r2, 1, bias=False)
            gate_in = config.news_dim + config.recency_embedding_dim
            self.gate_dense1 = nn.Linear(gate_in, g1)
            self.gate_dense2 = nn.Linear(g1, g2)
            self.gate_out = nn.Linear(g2, 1)

        if config.use_ctr:
            self.ctr_scaler = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.ctr_scaler.weight.fill_(config.ctr_scaler_init)

    def forward(
        self,
        bias_news_vec: torch.Tensor,
        recency_indices: torch.Tensor | None = None,
        ctr_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute popularity score.

        Args:
            bias_news_vec: ``(batch, news_dim)`` from the bias news encoder.
            recency_indices: ``(batch,)`` int recency bucket indices.
            ctr_values: ``(batch,)`` float CTR values.

        Returns:
            ``(batch,)`` popularity score.
        """
        x = torch.tanh(self.content_dense1(bias_news_vec))
        x = torch.tanh(self.content_dense2(x))
        x = self.content_dense3(x)
        content_score = self.content_out(x).squeeze(-1)

        pop_score = content_score

        if self.config.use_recency and recency_indices is not None:
            recency_emb = self.recency_embedding(recency_indices.long())
            r = torch.tanh(self.recency_dense1(recency_emb))
            r = torch.tanh(self.recency_dense2(r))
            recency_score = self.recency_out(r).squeeze(-1)

            gate_input = torch.cat([bias_news_vec, recency_emb], dim=-1)
            g = torch.tanh(self.gate_dense1(gate_input))
            g = torch.tanh(self.gate_dense2(g))
            gate = torch.sigmoid(self.gate_out(g)).squeeze(-1)

            pop_score = (1.0 - gate) * content_score + gate * recency_score

        if self.config.use_ctr and ctr_values is not None:
            ctr_input = ctr_values.unsqueeze(-1).float()
            ctr_score = self.ctr_scaler(ctr_input).squeeze(-1)
            pop_score = pop_score + ctr_score

        return pop_score


class ActivityGater(nn.Module):
    """Per-user gate ``η ∈ [0, 1]`` balancing relevance vs popularity."""

    def __init__(self, config: PPRecConfig):
        super().__init__()
        h = config.activity_gate_hidden_dim
        self.dense1 = nn.Linear(config.news_dim, h)
        self.dense2 = nn.Linear(h, 1)

    def forward(self, user_vec: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense1(user_vec))
        return torch.sigmoid(self.dense2(x)).squeeze(-1)
