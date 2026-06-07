"""Pluggable poolers over per-news token sequences (PyTorch).

Used by :class:`PLMTokenNewsEncoder` to collapse a ``(B, T, D)`` token
sequence into a per-news ``(B, D)`` vector. Mirrors the choices in
IP2's :class:`Pooler` (``src/sub_module.py``):

- ``mean`` — attention-mask-weighted mean over tokens
- ``cls``  — first token's vector (``[CLS]``)
- ``avg_first_last`` — mean of first+last hidden layers, mask-weighted
  (not supported here since we cache only the last layer; falls back to
  ``mean`` and logs a warning)
- ``attention`` — learnable MHA + AdditiveAttention pool (IP2 style)
- ``gate`` — gated pool

The frozen poolers (``mean``, ``cls``) introduce no trainable params.
The learnable poolers (``attention``, ``gate``) carry their own params.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .attention_layers import AdditiveAttention

logger = logging.getLogger(__name__)


class Pooler(nn.Module):
    """Pools ``(B, T, D)`` token features into ``(B, D)`` per-news vectors.

    Args:
        plm_dim: PLM hidden size (last-axis size of the input).
        pooler_type: One of ``"mean"``, ``"cls"``, ``"attention"``,
            ``"gate"``, ``"avg_first_last"`` (treated as ``"mean"``).
        attention_query_dim: Hidden dim of the learnable poolers'
            internal MLPs (unused by the frozen poolers).
        num_heads: Number of attention heads for the ``"attention"``
            pooler (IP2 default = 6).
        head_dim: Per-head dim for the ``"attention"`` pooler (IP2
            default = 128).
        dropout_rate: Dropout in the learnable poolers.
    """

    SUPPORTED = ("mean", "cls", "attention", "gate", "avg_first_last")

    def __init__(
        self,
        plm_dim: int,
        pooler_type: str = "mean",
        *,
        attention_query_dim: int = 200,
        num_heads: int = 6,
        head_dim: int = 128,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if pooler_type not in self.SUPPORTED:
            raise ValueError(
                f"Unknown pooler_type: {pooler_type!r}. Supported: {self.SUPPORTED}"
            )
        self.pooler_type = pooler_type
        self.plm_dim = plm_dim

        if pooler_type == "attention":
            # IP2 uses MHA followed by AdditiveAttention with a residual+LN.
            self.norm = nn.LayerNorm(plm_dim)
            # nn.MultiheadAttention forces head_dim = embed_dim/num_heads,
            # so to honor a separate head_dim we go through an explicit
            # bottleneck: project to num_heads*head_dim, do MHA there,
            # project back.  Keeps params identical to a "real" MHA at
            # the configured head dim.
            inner_dim = num_heads * head_dim
            self.attn_in = nn.Linear(plm_dim, inner_dim)
            self.attn_mha = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.attn_out = nn.Linear(inner_dim, plm_dim)
            self.additive = AdditiveAttention(
                input_dim=plm_dim,
                query_vec_dim=attention_query_dim,
            )
        elif pooler_type == "gate":
            self.norm = nn.LayerNorm(plm_dim)
            inner_dim = num_heads * head_dim
            self.attn_in = nn.Linear(plm_dim, inner_dim)
            self.attn_mha = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.attn_out = nn.Linear(inner_dim, plm_dim)
            self.fc1 = nn.Linear(plm_dim, 300)
            self.fc2 = nn.Linear(300, 1, bias=False)
        elif pooler_type == "avg_first_last":
            logger.warning(
                "avg_first_last pooler requires caching the first layer "
                "too; falling back to 'mean' pooling on the last layer."
            )

    def forward(
        self, token_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token features.

        Args:
            token_features: ``(B, T, plm_dim)`` per-token vectors.
            attention_mask: ``(B, T)`` 0/1 mask (1 = real token).

        Returns:
            ``(B, plm_dim)`` per-news vectors.
        """
        mask_f = attention_mask.to(token_features.dtype).unsqueeze(-1)  # (B, T, 1)

        if self.pooler_type in ("mean", "avg_first_last"):
            denom = (
                attention_mask.to(token_features.dtype)
                .sum(dim=1, keepdim=True)
                .clamp(min=1.0)
            )
            return (token_features * mask_f).sum(dim=1) / denom

        if self.pooler_type == "cls":
            return token_features[:, 0, :]

        if self.pooler_type == "attention":
            # MHA over the token sequence (key_padding_mask uses True=pad).
            key_padding_mask = attention_mask == 0  # (B, T)
            # Unmask one row for fully-empty sequences to avoid all-`-inf`
            # softmax → NaN. Output for those rows will be zero anyway
            # since downstream attention pool sees mask=False.
            fully_padded = key_padding_mask.all(dim=-1)
            if fully_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_padded, 0] = False

            x = self.attn_in(token_features)  # (B, T, inner_dim)
            attn_out, _ = self.attn_mha(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=False
            )
            attn_out = self.attn_out(attn_out)  # (B, T, plm_dim)
            normed = self.norm(token_features + attn_out)  # residual+LN
            return self.additive(normed, mask=attention_mask.bool())

        if self.pooler_type == "gate":
            key_padding_mask = attention_mask == 0
            fully_padded = key_padding_mask.all(dim=-1)
            if fully_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_padded, 0] = False
            x = self.attn_in(token_features)
            attn_out, _ = self.attn_mha(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=False
            )
            attn_out = self.attn_out(attn_out)
            h_tilde = self.norm(token_features + attn_out)  # (B, T, plm_dim)
            r = torch.softmax(
                self.fc2(torch.tanh(self.fc1(h_tilde))), dim=1
            )  # (B, T, 1)
            # Mask out padded positions in the gate weights
            r = r * mask_f
            r = r / r.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return (r * h_tilde).sum(dim=1)  # (B, plm_dim)

        raise RuntimeError(f"Unreachable: pooler_type={self.pooler_type}")
