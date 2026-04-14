"""DIGAT (EMNLP 2022 Findings) — PyTorch implementation.

Dual Interactive Graph Attention Networks for news recommendation.
Two interacting graph channels (news-SAG + user-topic) co-evolve across
``graph_depth`` layers.  The SAG (Semantic Augmented Graph) is precomputed
offline; the user-topic graph is built per behavior from category
assignments.

No ``torch_scatter`` dependency — scatter operations are implemented
with standard PyTorch ops.

Reference: Mao et al., "DIGAT: Modeling News Recommendation with
Dual-Graph Interaction", EMNLP 2022 Findings.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import DIGATConfig

from .base import BaseModel

# ======================================================================
# Scatter utilities (replace torch_scatter)
# ======================================================================


def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Per-group softmax without torch_scatter.

    Args:
        src: (B, N) scores.
        index: (B, N) int group assignments in [0, num_groups).
        num_groups: Total number of groups.

    Returns:
        (B, N) softmax-normalised within each group.
    """
    B, N = src.shape
    idx = index.long()

    # Max per group (for numerical stability)
    group_max = torch.full((B, num_groups), float("-inf"), device=src.device, dtype=src.dtype)
    group_max.scatter_reduce_(1, idx, src, reduce="amax", include_self=True)
    element_max = group_max.gather(1, idx)  # (B, N)

    exp_src = torch.exp(src - element_max)

    # Sum per group
    group_sum = torch.zeros(B, num_groups, device=src.device, dtype=src.dtype)
    group_sum.scatter_add_(1, idx, exp_src)
    element_sum = group_sum.gather(1, idx)  # (B, N)

    return exp_src / (element_sum + 1e-10)


def scatter_sum(
    src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int
) -> torch.Tensor:
    """Grouped sum without torch_scatter.

    Args:
        src: (B, N, D) values.
        index: (B, N) int group assignments in [0, dim_size).
        dim: Dimension to scatter along (must be 1).
        dim_size: Output size along the scatter dimension.

    Returns:
        (B, dim_size, D) summed values per group.
    """
    B, _, D = src.shape
    out = torch.zeros(B, dim_size, D, device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(dim, idx, src)


# ======================================================================
# Attention layers
# ======================================================================


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: K/Q projections → softmax → pool."""

    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super().__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.scale = math.sqrt(float(attention_dim))

    def forward(
        self, features: torch.Tensor, query: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = torch.bmm(self.K(features), self.Q(query).unsqueeze(2)).squeeze(2) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention for sequence pooling."""

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.affine = nn.Linear(feature_dim, attention_dim, bias=True)
        self.project = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.project(torch.tanh(self.affine(features))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), features).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (news encoder MSA)."""

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = num_heads * head_dim
        self.scale = math.sqrt(float(head_dim))
        self.W_Q = nn.Linear(d_model, self.out_dim, bias=True)
        self.W_K = nn.Linear(d_model, self.out_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.out_dim)
        return out


# ======================================================================
# News encoder
# ======================================================================


class DIGATNewsEncoder(nn.Module):
    """MSA-based news encoder: embedding → MHA → additive attention."""

    def __init__(
        self,
        config: DIGATConfig,
        word_embedding: nn.Embedding,
    ):
        super().__init__()
        self.word_embedding = word_embedding
        self.dropout = nn.Dropout(config.dropout_rate)
        self.msa = MultiHeadSelfAttention(
            config.embedding_size, config.msa_head_num, config.msa_head_dim,
        )
        self.news_embedding_dim = config.news_embedding_dim
        self.attention = AdditiveAttention(self.news_embedding_dim, config.attention_dim)

    def forward(
        self, title_text: torch.Tensor, title_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            title_text: (B, N, T) token ids — B batches, N news per batch, T title length.
            title_mask: (B, N, T) bool mask (1 = valid token).

        Returns:
            (B, N, news_embedding_dim) news representations.
        """
        B, N, T = title_text.shape
        flat_text = title_text.reshape(B * N, T)
        flat_mask = title_mask.reshape(B * N, T) if title_mask is not None else None

        w = self.dropout(self.word_embedding(flat_text))
        h = F.relu(self.msa(w))
        news_repr = self.attention(h, mask=flat_mask)
        return news_repr.view(B, N, self.news_embedding_dim)


# ======================================================================
# Dual graph encoder
# ======================================================================


class DIGATGraphEncoder(nn.Module):
    """Dual Interactive Graph Attention encoder.

    Maintains a news-graph channel (SAG subgraph per candidate) and a
    user-graph channel (history + topic nodes).  The two channels
    interact across ``graph_depth`` layers: each graph's attention
    incorporates the other channel's context vector.
    """

    def __init__(self, config: DIGATConfig):
        super().__init__()
        D = config.news_embedding_dim
        depth = config.graph_depth
        self.graph_depth = depth
        self.max_history = config.max_history_length
        self.D = D
        self.scale = math.sqrt(float(D))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dropout_ = nn.Dropout(config.dropout_rate)
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

        # Learnable topic node embeddings
        self.topic_node_emb: nn.Parameter  # set in DIGAT.__init__ once category_num is known

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def _news_graph_context(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Gated local/global context from a news SAG subgraph.

        Args:
            emb: (B, G, D) — node embeddings for the SAG subgraph.
            mask: (B, G) — valid-node mask.

        Returns:
            (B, D) context vector.
        """
        local = emb[:, 0, :]
        glob = self.news_ctx_attn(emb, local, mask=mask)
        gate = torch.sigmoid(self.news_ctx_gate(torch.cat([local, glob], dim=-1)))
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
            emb: (B, H+C, D) — history news + topic node embeddings.
            cat_mask: (B, C+1) — which categories are active.
            cat_indices: (B, H) — topic assignment per history news.
            news_ctx: (B, D) — current news context.
            num_categories: C — number of category/topic nodes.

        Returns:
            (B, D) user context vector.
        """
        hist = emb[:, : self.max_history, :]  # (B, H, D)

        K = self.user_news_K(hist)
        Q = self.user_news_Q(news_ctx).unsqueeze(2)
        scores = torch.bmm(K, Q).squeeze(2) / self.scale  # (B, H)

        alpha = scatter_softmax(scores, cat_indices, num_categories)  # (B, H)
        weighted = alpha.unsqueeze(-1) * hist  # (B, H, D)
        topic_emb = scatter_sum(weighted, cat_indices, dim=1, dim_size=num_categories)
        topic_emb = self.dropout(F.relu(self.topic_affine(topic_emb)) + topic_emb)

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
        B = emb.size(0)
        emb = self.dropout(emb)
        h = W[idx](emb)
        K1 = ffn1[idx](emb).unsqueeze(1)  # (B, 1, N, D)
        K2 = ffn2[idx](emb).unsqueeze(2)  # (B, N, 1, D)
        K3 = ffn3[idx](cross_ctx).view(B, 1, 1, self.D)
        scores = a[idx](F.relu(K1 + K2 + K3)).squeeze(-1)  # (B, N, N)
        scores = self.leaky_relu(scores)
        scores = scores.masked_fill(adj == 0, -1e9)
        alpha = self.dropout_(torch.softmax(scores, dim=2))
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
            news_emb: (B, G_n, D) — SAG node embeddings per candidate.
            news_graph: (B, G_n, G_n) — SAG adjacency.
            news_mask: (B, G_n) — valid SAG nodes.
            user_news_emb: (B, H, D) — history news embeddings.
            user_graph: (B, G_u, G_u) — user graph adjacency.
            user_cat_mask: (B, C+1) — active categories.
            user_cat_indices: (B, H) — topic per history item.
            num_categories: int — number of topic nodes.

        Returns:
            (news_ctx, user_ctx) — each (B, D).
        """
        B = news_emb.size(0)
        topic_nodes = self.topic_node_emb.unsqueeze(0).expand(B, -1, -1)
        user_emb = torch.cat([user_news_emb, self.dropout(topic_nodes)], dim=1)

        news_ctx = self._news_graph_context(news_emb, news_mask)
        user_ctx = self._user_graph_context(
            user_emb, user_cat_mask, user_cat_indices, news_ctx, num_categories
        )

        for i in range(self.graph_depth):
            news_emb = self._update_graph(
                i, news_emb, news_graph, user_ctx,
                self.n_W, self.n_ffn1, self.n_ffn2, self.n_ffn3, self.n_a,
            )
            user_emb = self._update_graph(
                i, user_emb, user_graph, news_ctx,
                self.u_W, self.u_ffn1, self.u_ffn2, self.u_ffn3, self.u_a,
            )
            news_ctx = news_ctx + self._news_graph_context(news_emb, news_mask)
            user_ctx = user_ctx + self._user_graph_context(
                user_emb, user_cat_mask, user_cat_indices, news_ctx, num_categories
            )

        return news_ctx, user_ctx


# ======================================================================
# Full DIGAT model
# ======================================================================


class DIGAT(BaseModel):
    """DIGAT: Dual Interactive Graph Attention Networks.

    Training: encode candidates (with SAG) + history → dual graph
    interaction → dot-product scoring.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: DIGATConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = DIGATConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        self.num_categories = num_categories + 1  # +1 for padding category

        vocab_size = int(processed_news["vocab_size"])
        embeddings_matrix = processed_news["embeddings"]
        self.word_embedding = nn.Embedding(vocab_size, config.embedding_size)
        self.word_embedding.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        self.news_encoder = DIGATNewsEncoder(config, self.word_embedding)

        self.graph_encoder = DIGATGraphEncoder(config)
        self.graph_encoder.topic_node_emb = nn.Parameter(
            torch.zeros(self.num_categories, config.news_embedding_dim)
        )
        nn.init.uniform_(self.graph_encoder.topic_node_emb, -0.1, 0.1)

        self.news_graph_size = config.news_graph_size
        self.max_history = config.max_history_length
        self.D = config.news_embedding_dim

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Training forward pass.

        Expected input keys:
            hist_tokens: (B, H, T)
            hist_mask: (B, H, T)
            user_graph: (B, G_u, G_u)
            user_category_mask: (B, C+1)
            user_category_indices: (B, H)
            cand_tokens: (B, C, G_n, T) — candidates with SAG neighbors
            cand_mask: (B, C, G_n, T)
            cand_graph: (B, C, G_n, G_n)
            cand_graph_mask: (B, C, G_n)

        Returns:
            (B, C) raw logits.
        """
        B = inputs["cand_graph"].size(0)
        C = inputs["cand_graph"].size(1)
        G_n = self.news_graph_size
        G_u = self.max_history + self.num_categories
        BC = B * C

        # Flatten candidates: (B, C, G_n, T) → (B*C, G_n, T)
        cand_tokens = inputs["cand_tokens"].view(BC, G_n, -1)
        cand_mask = inputs["cand_mask"].view(BC, G_n, -1) if "cand_mask" in inputs else None
        cand_graph = inputs["cand_graph"].view(BC, G_n, G_n)
        cand_graph_mask = inputs["cand_graph_mask"].view(BC, G_n)

        # Expand user data: (B, ...) → (B*C, ...)
        user_graph = inputs["user_graph"].unsqueeze(1).expand(-1, C, -1, -1).reshape(BC, G_u, G_u)
        user_cat_mask = inputs["user_category_mask"].unsqueeze(1).expand(-1, C, -1).reshape(BC, self.num_categories)
        user_cat_indices = inputs["user_category_indices"].unsqueeze(1).expand(-1, C, -1).reshape(BC, self.max_history)

        # Encode candidate news (each with SAG neighbors)
        cand_emb = self.news_encoder(cand_tokens, cand_mask)  # (BC, G_n, D)

        # Encode user history
        user_news_emb = self.news_encoder(
            inputs["hist_tokens"],
            inputs.get("hist_mask"),
        )  # (B, H, D)
        user_news_emb = user_news_emb.unsqueeze(1).expand(-1, C, -1, -1).reshape(BC, self.max_history, self.D)

        # Dual graph interaction
        news_ctx, user_ctx = self.graph_encoder(
            cand_emb, cand_graph, cand_graph_mask,
            user_news_emb, user_graph, user_cat_mask, user_cat_indices,
            self.num_categories,
        )

        # Dot-product scoring
        news_ctx = news_ctx.view(B, C, self.D)
        user_ctx = user_ctx.view(B, C, self.D)
        return (news_ctx * user_ctx).sum(dim=-1)  # (B, C)
