"""CAUM candidate-aware interaction model (PyTorch).

Three candidate-aware modules combine clicked-news vectors with a single
candidate vector to produce a scalar matching score:

    1. Candi-CNN     — circular-shift window over clicks + candidate -> Dense
    2. Candi-SelfAtt — concat candidate with each click -> Dense -> MHSA
    3. Candi-Att     — DNN scorer over [fused, candidate] -> softmax
                       -> weighted sum (user_vec)
    4. Score         — dot(user_vec, candidate)

Because the user representation depends on the candidate, the standard
"precompute user vec once → dot product" evaluator cannot be used; CAUM
uses a custom evaluator (see :mod:`src.core.models.evaluations.custom.caum`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.models.configs import CAUMConfig


class DenseAttentionScorer(nn.Module):
    """Two-layer MLP scorer for Candi-Att."""

    def __init__(self, hidden_dim: int, mid_dim: int):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, mid_dim)
        self.dense3 = nn.Linear(mid_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: ``(*, 2 * hidden_dim)`` → ``(*, 1)``."""
        return self.dense3(torch.tanh(self.dense2(torch.tanh(self.dense1(inputs)))))


class InterModel(nn.Module):
    """Candidate-aware user interest model.

    Takes pre-encoded clicked-news vectors and a single candidate news
    vector, produces a scalar matching score. Operates entirely in the
    ``news_dim`` space so it does not need to know whether GloVe or PLM
    was used upstream.
    """

    def __init__(self, config: CAUMConfig):
        super().__init__()
        self.config = config
        D = config.news_dim

        self.dropout_cand = nn.Dropout(config.dropout_rate)
        self.dropout_clicks = nn.Dropout(config.dropout_rate)

        # Candi-CNN: projects [left, center, right, candidate] -> D
        self.cnn_projection = nn.Linear(4 * D, D)

        # Candi-SelfAtt: projects [candidate, click] -> D, then MHSA
        self.selfatt_input_projection = nn.Linear(2 * D, D)
        self.selfatt_mha = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=config.candi_selfatt_num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )

        # Fusion
        self.fusion_dropout = nn.Dropout(config.dropout_rate)
        self.fusion_projection = nn.Linear(2 * D, D)

        # Candi-Att (DNN scorer applied per click)
        self.dense_att = DenseAttentionScorer(
            hidden_dim=config.candi_att_hidden_dim,
            mid_dim=config.candi_att_mid_dim,
        )

    def forward(
        self,
        cand_vec: torch.Tensor,
        clicked_vecs: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Score a candidate against clicked news.

        Args:
            cand_vec: ``(B, D)`` single candidate news vector.
            clicked_vecs: ``(B, H, D)`` encoded clicked news vectors.

        Returns:
            ``(B,)`` scalar matching scores.
        """
        D = self.config.news_dim
        B, H, _ = clicked_vecs.shape

        can_vec_dropped = self.dropout_cand(cand_vec)
        user_vecs = self.dropout_clicks(clicked_vecs)

        cand_repeated = cand_vec.unsqueeze(1).expand(-1, H, -1)

        # Candi-CNN — circular-shift window.
        left = torch.cat([user_vecs[:, -1:, :], user_vecs[:, :-1, :]], dim=1)
        right = torch.cat([user_vecs[:, 1:, :], user_vecs[:, :1, :]], dim=1)
        cnn_input = torch.cat([left, user_vecs, right, cand_repeated], dim=-1)
        cnn_out = self.cnn_projection(cnn_input)

        # Candi-SelfAtt — concat candidate with each click, project, MHSA.
        selfatt_input = torch.cat([cand_repeated, user_vecs], dim=-1)
        selfatt_input = self.selfatt_input_projection(selfatt_input)

        history_mask = clicked_vecs.any(dim=-1)  # (B, H) — True where valid.
        key_padding_mask = ~history_mask

        fully_empty = key_padding_mask.all(dim=-1)
        if fully_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_empty, 0] = False

        selfatt_out, _ = self.selfatt_mha(
            selfatt_input,
            selfatt_input,
            selfatt_input,
            key_padding_mask=key_padding_mask,
        )

        # Fusion
        fused = torch.cat([cnn_out, selfatt_out], dim=-1)
        fused = self.fusion_dropout(fused)
        fused = self.fusion_projection(fused)

        # Candi-Att — score each fused click against the candidate.
        att_input = torch.cat([fused, cand_repeated], dim=-1)
        flat_att = att_input.reshape(B * H, 2 * D)
        flat_scores = self.dense_att(flat_att)
        att_scores = flat_scores.reshape(B, H)

        att_scores = att_scores.masked_fill(~history_mask, -1e9)
        att_weights = torch.softmax(att_scores, dim=-1)

        user_vec = (fused * att_weights.unsqueeze(-1)).sum(dim=1)
        return (user_vec * can_vec_dropped).sum(dim=-1)
