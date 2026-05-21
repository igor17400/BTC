"""DIGAT dual interactive graph encoder (PyTorch).

Maintains a news-graph channel (SAG subgraph per candidate) and a
user-graph channel (history + topic nodes). The two channels interact
across ``graph_depth`` layers: each graph's attention incorporates the
other channel's context vector.

This file also hosts the scatter utilities (``scatter_softmax``,
``scatter_sum``) used by the user-graph context, and the shared
``ScaledDotProductAttention`` used in both context extractors. No
``torch_scatter`` dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import DIGATConfig

# ======================================================================
# Scatter utilities (replace torch_scatter)
# ======================================================================


def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Per-group softmax without ``torch_scatter``.

    Args:
        src: ``(B, N)`` scores.
        index: ``(B, N)`` int group assignments in ``[0, num_groups)``.
        num_groups: Total number of groups.

    Returns:
        ``(B, N)`` softmax-normalised within each group.
    """
    batch_size, _ = src.shape
    idx = index.long()

    group_max = torch.full(
        (batch_size, num_groups), float("-inf"), device=src.device, dtype=src.dtype
    )
    group_max.scatter_reduce_(1, idx, src, reduce="amax", include_self=True)
    element_max = group_max.gather(1, idx)  # (batch_size, num_items)

    exp_src = torch.exp(src - element_max)

    group_sum = torch.zeros(batch_size, num_groups, device=src.device, dtype=src.dtype)
    group_sum.scatter_add_(1, idx, exp_src)
    element_sum = group_sum.gather(1, idx)  # (batch_size, num_items)

    return exp_src / (element_sum + 1e-10)


def scatter_sum(
    src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int
) -> torch.Tensor:
    """Grouped sum without ``torch_scatter``.

    Args:
        src: ``(B, N, D)`` values.
        index: ``(B, N)`` int group assignments in ``[0, dim_size)``.
        dim: Dimension to scatter along (must be 1).
        dim_size: Output size along the scatter dimension.

    Returns:
        ``(B, dim_size, D)`` summed values per group.
    """
    batch_size, _, feat_dim = src.shape
    out = torch.zeros(
        batch_size, dim_size, feat_dim, device=src.device, dtype=src.dtype
    )
    idx = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(dim, idx, src)


# ======================================================================
# Shared attention primitive
# ======================================================================


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: K/Q projections → softmax → pool."""

    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super().__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.scale = math.sqrt(float(attention_dim))

    def forward(
        self,
        features: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = (
            torch.bmm(self.K(features), self.Q(query).unsqueeze(2)).squeeze(2)
            / self.scale
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


# ======================================================================
# Dual graph encoder
# ======================================================================


class GraphEncoder(nn.Module):
    """Dual Interactive Graph Attention encoder."""

    def __init__(self, config: DIGATConfig):
        super().__init__()
        D = config.news_embedding_dim
        depth = config.graph_depth
        self.graph_depth = depth
        self.max_history = config.max_history_length
        self.D = D
        self.scale = math.sqrt(float(D))
        # topic_dropout — full rate, applied to topic embeddings after relu+residual.
        # attn_dropout  — full rate, applied to attention weight matrices.
        # input_dropout — half rate, applied to node embeddings before graph update
        #                 layers and to the gate linear output in news context;
        #                 mirrors reference ``dropout__ = Dropout(rate/2)``.
        self.topic_dropout = nn.Dropout(config.dropout_rate)
        self.attn_dropout = nn.Dropout(config.dropout_rate)
        self.input_dropout = nn.Dropout(config.dropout_rate / 2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # --- News graph context ---
        self.news_ctx_attn = ScaledDotProductAttention(D, D, D)
        self.news_ctx_gate = nn.Linear(D * 2, D)

        # --- User graph context (topic-level scatter + attention) ---
        self.user_news_K = nn.Linear(D, D, bias=False)
        self.user_news_Q = nn.Linear(D, D, bias=True)
        self.topic_affine = nn.Linear(D, D)
        self.user_ctx_attn = ScaledDotProductAttention(D, D, D)

        # --- Per-depth news graph update layers ---
        self.n_W = nn.ModuleList([nn.Linear(D, D) for _ in range(depth)])
        self.n_ffn1 = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(depth)])
        self.n_ffn2 = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(depth)])
        self.n_ffn3 = nn.ModuleList([nn.Linear(D, D) for _ in range(depth)])
        self.n_a = nn.ModuleList([nn.Linear(D, 1, bias=False) for _ in range(depth)])

        # --- Per-depth user graph update layers ---
        self.u_W = nn.ModuleList([nn.Linear(D, D) for _ in range(depth)])
        self.u_ffn1 = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(depth)])
        self.u_ffn2 = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(depth)])
        self.u_ffn3 = nn.ModuleList([nn.Linear(D, D) for _ in range(depth)])
        self.u_a = nn.ModuleList([nn.Linear(D, 1, bias=False) for _ in range(depth)])

        # Learnable topic node embeddings — assigned by ``DIGAT.__init__``
        # once ``num_categories`` is known.
        self.topic_node_emb: nn.Parameter

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier uniform init matching the reference DIGAT ``initialize`` method."""
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.n_W[i].weight)
            nn.init.zeros_(self.n_W[i].bias)
            nn.init.xavier_uniform_(
                self.n_a[i].weight, gain=nn.init.calculate_gain("leaky_relu", 0.2)
            )
            nn.init.xavier_uniform_(
                self.n_ffn1[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.xavier_uniform_(
                self.n_ffn2[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.xavier_uniform_(
                self.n_ffn3[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.zeros_(self.n_ffn3[i].bias)

            nn.init.xavier_uniform_(self.u_W[i].weight)
            nn.init.zeros_(self.u_W[i].bias)
            nn.init.xavier_uniform_(
                self.u_a[i].weight, gain=nn.init.calculate_gain("leaky_relu", 0.2)
            )
            nn.init.xavier_uniform_(
                self.u_ffn1[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.xavier_uniform_(
                self.u_ffn2[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.xavier_uniform_(
                self.u_ffn3[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.zeros_(self.u_ffn3[i].bias)

        # news context gate
        nn.init.xavier_uniform_(self.news_ctx_gate.weight)
        nn.init.zeros_(self.news_ctx_gate.bias)

        # user context: K/Q projections and topic affine
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(
            self.topic_affine.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.zeros_(self.topic_affine.bias)

        # ScaledDotProductAttention layers
        for attn in (self.news_ctx_attn, self.user_ctx_attn):
            nn.init.xavier_uniform_(attn.K.weight)
            nn.init.xavier_uniform_(attn.Q.weight)
            nn.init.zeros_(attn.Q.bias)

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def _news_graph_context(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Gated local/global context from a news SAG subgraph.

        Args:
            emb: ``(B, G, D)`` node embeddings for the SAG subgraph.
            mask: ``(B, G)`` valid-node mask.

        Returns:
            ``(B, D)`` context vector.
        """
        local = emb[:, 0, :]
        glob = self.news_ctx_attn(emb, local, mask=mask)
        gate = torch.sigmoid(
            self.input_dropout(self.news_ctx_gate(torch.cat([local, glob], dim=-1)))
        )
        return gate * local + (1 - gate) * glob

    def _user_graph_context(
        self,
        emb: torch.Tensor,
        cat_mask: torch.Tensor,
        cat_indices: torch.Tensor,
        news_ctx: torch.Tensor,
        num_categories: int,
    ) -> torch.Tensor:
        """Two-level user context: topic-level scatter → user-level attention.

        Args:
            emb: ``(B, H+C, D)`` history news + topic node embeddings.
            cat_mask: ``(B, C+1)`` which categories are active.
            cat_indices: ``(B, H)`` topic assignment per history news.
            news_ctx: ``(B, D)`` current news context.
            num_categories: ``C`` number of category/topic nodes.

        Returns:
            ``(B, D)`` user context vector.
        """
        hist = emb[:, : self.max_history, :]  # (B, H, D)

        K = self.user_news_K(hist)
        Q = self.user_news_Q(news_ctx).unsqueeze(2)
        scores = torch.bmm(K, Q).squeeze(2) / self.scale  # (B, H)

        alpha = scatter_softmax(scores, cat_indices, num_categories)  # (B, H)
        weighted = alpha.unsqueeze(-1) * hist  # (B, H, D)
        topic_emb = scatter_sum(weighted, cat_indices, dim=1, dim_size=num_categories)
        topic_emb = self.topic_dropout(F.relu(self.topic_affine(topic_emb)) + topic_emb)

        return self.user_ctx_attn(topic_emb, news_ctx, mask=cat_mask)

    # ------------------------------------------------------------------
    # Graph update (one layer)
    # ------------------------------------------------------------------

    def _update_graph(
        self,
        idx: int,
        emb: torch.Tensor,
        adj: torch.Tensor,
        cross_ctx: torch.Tensor,
        W: nn.ModuleList,
        ffn1: nn.ModuleList,
        ffn2: nn.ModuleList,
        ffn3: nn.ModuleList,
        a: nn.ModuleList,
    ) -> torch.Tensor:
        """Single cross-interactive graph attention update."""
        batch_size, num_nodes, _ = emb.shape
        emb = self.input_dropout(emb)
        h = W[idx](emb)
        K1 = ffn1[idx](emb).unsqueeze(1)  # (B, 1, N, D)
        K2 = ffn2[idx](emb).unsqueeze(2)  # (B, N, 1, D)
        K3 = ffn3[idx](cross_ctx).view(batch_size, 1, 1, self.D)
        scores = a[idx](F.relu(K1 + K2 + K3)).squeeze(-1)  # (B, N, N)
        scores = self.leaky_relu(scores)
        scores = scores.masked_fill(adj == 0, -1e9)
        alpha = self.attn_dropout(torch.softmax(scores, dim=2))
        return F.relu(torch.bmm(alpha, h)) + emb

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        news_emb: torch.Tensor,
        news_graph: torch.Tensor,
        news_mask: torch.Tensor,
        user_news_emb: torch.Tensor,
        user_graph: torch.Tensor,
        user_cat_mask: torch.Tensor,
        user_cat_indices: torch.Tensor,
        num_categories: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            news_emb: ``(B, G_n, D)`` SAG node embeddings per candidate.
            news_graph: ``(B, G_n, G_n)`` SAG adjacency.
            news_mask: ``(B, G_n)`` valid SAG nodes.
            user_news_emb: ``(B, H, D)`` history news embeddings.
            user_graph: ``(B, G_u, G_u)`` user graph adjacency.
            user_cat_mask: ``(B, C+1)`` active categories.
            user_cat_indices: ``(B, H)`` topic per history item.
            num_categories: ``C`` number of topic nodes.

        Returns:
            ``(news_ctx, user_ctx)`` — each ``(B, D)``.
        """
        batch_size = news_emb.size(0)
        topic_nodes = self.topic_node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        user_emb = torch.cat([user_news_emb, self.input_dropout(topic_nodes)], dim=1)

        news_ctx = self._news_graph_context(news_emb, news_mask)
        user_ctx = self._user_graph_context(
            user_emb, user_cat_mask, user_cat_indices, news_ctx, num_categories
        )

        for i in range(self.graph_depth):
            news_emb = self._update_graph(
                i,
                news_emb,
                news_graph,
                user_ctx,
                self.n_W,
                self.n_ffn1,
                self.n_ffn2,
                self.n_ffn3,
                self.n_a,
            )
            user_emb = self._update_graph(
                i,
                user_emb,
                user_graph,
                news_ctx,
                self.u_W,
                self.u_ffn1,
                self.u_ffn2,
                self.u_ffn3,
                self.u_a,
            )
            news_ctx = news_ctx + self._news_graph_context(news_emb, news_mask)
            user_ctx = user_ctx + self._user_graph_context(
                user_emb, user_cat_mask, user_cat_indices, news_ctx, num_categories
            )

        return news_ctx, user_ctx
