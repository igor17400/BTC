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

from ..base import BaseModel
from .layers import (
    AdditiveAttention,
    MultiHeadSelfAttention,
    ScaledDotProductAttention,
    scatter_softmax,
    scatter_sum,
)


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
            title_text: (batch_size, num_news, title_len) token ids.
            title_mask: (batch_size, num_news, title_len) bool mask (1 = valid token).

        Returns:
            (batch_size, num_news, news_embedding_dim) news representations.
        """
        batch_size, num_news, title_len = title_text.shape
        flat_text = title_text.reshape(batch_size * num_news, title_len)
        flat_mask = title_mask.reshape(batch_size * num_news, title_len) if title_mask is not None else None

        w = self.dropout(self.word_embedding(flat_text))
        h = F.relu(self.msa(w))
        news_repr = self.attention(h, mask=flat_mask)
        return news_repr.view(batch_size, num_news, self.news_embedding_dim)


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
        # topic_dropout  — full rate, applied to topic embeddings after relu+residual
        # attn_dropout   — full rate, applied to attention weight matrices (must be inplace=False)
        # input_dropout  — half rate, applied to node embeddings before graph update layers
        #                  and to the gate linear output in news context; mirrors reference
        #                  dropout__ = Dropout(rate/2) from the original DIGAT code.
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

        # Learnable topic node embeddings
        self.topic_node_emb: nn.Parameter  # set in DIGAT.__init__ once category_num is known

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier uniform init matching the reference DIGAT initialize() method."""
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.n_W[i].weight)
            nn.init.zeros_(self.n_W[i].bias)
            nn.init.xavier_uniform_(self.n_a[i].weight, gain=nn.init.calculate_gain("leaky_relu", 0.2))
            nn.init.xavier_uniform_(self.n_ffn1[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.n_ffn2[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.n_ffn3[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.n_ffn3[i].bias)

            nn.init.xavier_uniform_(self.u_W[i].weight)
            nn.init.zeros_(self.u_W[i].bias)
            nn.init.xavier_uniform_(self.u_a[i].weight, gain=nn.init.calculate_gain("leaky_relu", 0.2))
            nn.init.xavier_uniform_(self.u_ffn1[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.u_ffn2[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.u_ffn3[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.u_ffn3[i].bias)

        # news context gate
        nn.init.xavier_uniform_(self.news_ctx_gate.weight)
        nn.init.zeros_(self.news_ctx_gate.bias)

        # user context: K/Q projections and topic affine
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.topic_affine.weight, gain=nn.init.calculate_gain("relu"))
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
            emb: (B, G, D) — node embeddings for the SAG subgraph.
            mask: (B, G) — valid-node mask.

        Returns:
            (B, D) context vector.
        """
        local = emb[:, 0, :]
        glob = self.news_ctx_attn(emb, local, mask=mask)
        gate = torch.sigmoid(self.input_dropout(self.news_ctx_gate(torch.cat([local, glob], dim=-1))))
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
        K1 = ffn1[idx](emb).unsqueeze(1)  # (batch_size, 1, num_nodes, feat_dim)
        K2 = ffn2[idx](emb).unsqueeze(2)  # (batch_size, num_nodes, 1, feat_dim)
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
        batch_size = news_emb.size(0)
        topic_nodes = self.topic_node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        user_emb = torch.cat([user_news_emb, self.input_dropout(topic_nodes)], dim=1)

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
            hist_tokens: (batch_size, hist_len, title_len)
            hist_mask: (batch_size, hist_len, title_len)
            user_graph: (batch_size, user_graph_size, user_graph_size)
            user_category_mask: (batch_size, num_categories)
            user_category_indices: (batch_size, hist_len)
            cand_tokens: (batch_size, num_cands, sag_size, title_len)
            cand_mask: (batch_size, num_cands, sag_size, title_len)
            cand_graph: (batch_size, num_cands, sag_size, sag_size)
            cand_graph_mask: (batch_size, num_cands, sag_size)

        Returns:
            (batch_size, num_cands) raw logits.
        """
        batch_size, num_cands = inputs["cand_graph"].shape[:2]
        sag_size = self.news_graph_size
        user_graph_size = self.max_history + self.num_categories
        batch_cands = batch_size * num_cands

        # Flatten candidates: (batch_size, num_cands, sag_size, title_len) → (batch_cands, sag_size, title_len)
        cand_tokens = inputs["cand_tokens"].view(batch_cands, sag_size, -1)
        cand_mask = inputs["cand_mask"].view(batch_cands, sag_size, -1) if "cand_mask" in inputs else None
        cand_graph = inputs["cand_graph"].view(batch_cands, sag_size, sag_size)
        cand_graph_mask = inputs["cand_graph_mask"].view(batch_cands, sag_size)

        # Expand user data: (batch_size, ...) → (batch_cands, ...)
        user_graph = inputs["user_graph"].unsqueeze(1).expand(-1, num_cands, -1, -1).reshape(batch_cands, user_graph_size, user_graph_size)
        user_cat_mask = inputs["user_category_mask"].unsqueeze(1).expand(-1, num_cands, -1).reshape(batch_cands, self.num_categories)
        user_cat_indices = inputs["user_category_indices"].unsqueeze(1).expand(-1, num_cands, -1).reshape(batch_cands, self.max_history)

        # Encode candidate news (each with SAG neighbors)
        cand_emb = self.news_encoder(cand_tokens, cand_mask)  # (batch_cands, sag_size, feat_dim)

        # Encode user history
        user_news_emb = self.news_encoder(
            inputs["hist_tokens"],
            inputs.get("hist_mask"),
        )  # (batch_size, hist_len, feat_dim)
        user_news_emb = user_news_emb.unsqueeze(1).expand(-1, num_cands, -1, -1).reshape(batch_cands, self.max_history, self.D)

        # Dual graph interaction
        news_ctx, user_ctx = self.graph_encoder(
            cand_emb, cand_graph, cand_graph_mask,
            user_news_emb, user_graph, user_cat_mask, user_cat_indices,
            self.num_categories,
        )

        # Dot-product scoring
        news_ctx = news_ctx.view(batch_size, num_cands, self.D)
        user_ctx = user_ctx.view(batch_size, num_cands, self.D)
        return (news_ctx * user_ctx).sum(dim=-1)  # (batch_size, num_cands)
