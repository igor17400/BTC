"""CROWN user encoder (PyTorch) — encoder-agnostic, packed-input.

Pipeline (paper §3.3):
    1. Encode each history slot via shared news encoder.
    2. Initialise a learnable user proxy node.
    3. ``graph_num_layers`` layers of bipartite GNN (GAT or GraphSAGE)
       over the user<->news graph — returns updated user proxy + news.
    4. Additive attention (paper eq. 9) with the updated user proxy as
       query, pooling the updated news nodes into a single user vector.
       Candidate-independent — same mechanism in training and eval.

Accepts packed history ``(B, H, 3)`` = ``[news_idx | cat | subcat]``
per slot (same shape used by both training and eval).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import CROWNConfig

from .news_encoder import NewsEncoder


class UserQueryAttention(nn.Module):
    """Additive attention with the GNN-updated user proxy as query.

    Paper eq. 9 — candidate-independent.
    """

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.W_key = nn.Linear(feature_dim, attention_dim, bias=True)
        self.W_query = nn.Linear(feature_dim, attention_dim, bias=False)

    def forward(
        self,
        news: torch.Tensor,
        user_node: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        keys = torch.tanh(self.W_key(news))  # (B, H, A)
        query = self.W_query(user_node).unsqueeze(1)  # (B, 1, A)
        scores = (keys * query).sum(dim=-1)  # (B, H)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, H, 1)
        return (news * weights).sum(dim=1)


class BipartiteGATLayer(nn.Module):
    """1-layer GAT on a user<->news bipartite graph with self-loops."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        alpha: float = 0.2,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.alpha = alpha

        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        user: torch.Tensor,
        news: torch.Tensor,
        news_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, D = news.shape
        NH, HD = self.num_heads, self.head_dim

        Wu = self.W(user).view(B, NH, HD)
        Wn = self.W(news).view(B, H, NH, HD)

        src_u = (Wu * self.a_src).sum(dim=-1)
        dst_u = (Wu * self.a_dst).sum(dim=-1)
        src_n = (Wn * self.a_src).sum(dim=-1)
        dst_n = (Wn * self.a_dst).sum(dim=-1)

        neg_inf = torch.finfo(src_u.dtype).min

        # User update: attend over {self, all news}
        score_u_n = F.leaky_relu(src_u.unsqueeze(1) + dst_n, self.alpha)
        score_u_u = F.leaky_relu(src_u + dst_u, self.alpha)
        score_u_n = score_u_n.masked_fill(~news_mask.unsqueeze(-1).bool(), neg_inf)

        all_u = torch.cat([score_u_n, score_u_u.unsqueeze(1)], dim=1)
        attn_u = self.dropout(torch.softmax(all_u, dim=1))
        u_from_n = (attn_u[:, :H].unsqueeze(-1) * Wn).sum(dim=1)
        u_self = attn_u[:, H].unsqueeze(-1) * Wu
        user_new = F.elu(u_from_n + u_self).reshape(B, D)

        # News update: each news attends over {self, user}
        score_n_u = F.leaky_relu(src_n + dst_u.unsqueeze(1), self.alpha)
        score_n_n = F.leaky_relu(src_n + dst_n, self.alpha)
        stacked = torch.stack([score_n_u, score_n_n], dim=-2)
        attn_n = self.dropout(torch.softmax(stacked, dim=-2))
        n_from_u = attn_n[:, :, 0].unsqueeze(-1) * Wu.unsqueeze(1)
        n_self = attn_n[:, :, 1].unsqueeze(-1) * Wn
        news_new = F.elu(n_from_u + n_self).reshape(B, H, D)

        return user_new, news_new


class BipartiteSAGELayer(nn.Module):
    """1-layer GraphSAGE on a user<->news bipartite graph (mean aggregator)."""

    def __init__(self, dim: int, dropout_rate: float):
        super().__init__()
        self.W_self = nn.Linear(dim, dim)
        self.W_neigh = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        user: torch.Tensor,
        news: torch.Tensor,
        news_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = news_mask.unsqueeze(-1).to(news.dtype)
        news_count = m.sum(dim=1).clamp(min=1.0)
        news_mean = (news * m).sum(dim=1) / news_count

        user_new = F.relu(self.W_self(user) + self.W_neigh(news_mean))
        news_new = F.relu(
            self.W_self(news) + self.W_neigh(user.unsqueeze(1).expand_as(news))
        )
        user_new = F.normalize(user_new, p=2, dim=-1)
        news_new = F.normalize(news_new, p=2, dim=-1)
        return self.dropout(user_new), self.dropout(news_new)


class UserEncoder(nn.Module):
    """CROWN user encoder: bipartite GNN + paper eq. 9 user-query attention."""

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: NewsEncoder,
    ):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        news_emb_dim = news_encoder.news_embedding_dim

        # Learnable user proxy node (paper: randomly initialised).
        self.user_node = nn.Parameter(torch.empty(news_emb_dim))
        nn.init.uniform_(self.user_node, -0.1, 0.1)

        if config.gnn_type == "gat":
            self.gnn_layers = nn.ModuleList(
                [
                    BipartiteGATLayer(
                        dim=news_emb_dim,
                        num_heads=config.gat_num_heads,
                        dropout_rate=config.dropout_rate,
                        alpha=config.gat_alpha,
                    )
                    for _ in range(config.graph_num_layers)
                ]
            )
        elif config.gnn_type == "graphsage":
            self.gnn_layers = nn.ModuleList(
                [
                    BipartiteSAGELayer(
                        dim=news_emb_dim, dropout_rate=config.dropout_rate
                    )
                    for _ in range(config.graph_num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown gnn_type: {config.gnn_type!r}")

        self.user_attention = UserQueryAttention(
            feature_dim=news_emb_dim,
            attention_dim=config.user_attention_dim,
        )

    def _encode_history_graph(
        self,
        history_packed: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode history -> bipartite GNN -> ``(user_node, news_nodes)``."""
        B, H = history_packed.shape[:2]
        # news_encoder takes ``(*, 3)`` and returns ``(*, news_embedding_dim)``.
        news = self.news_encoder(history_packed, compute_aux_loss=False)  # (B, H, D)

        user = self.user_node.unsqueeze(0).expand(B, -1).contiguous()
        for gnn in self.gnn_layers:
            user, news = gnn(user, news, history_mask)
        return user, news

    def forward_with_candidates(
        self,
        history_packed: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Training: one user representation per behavior.

        ``candidate_repr`` is accepted for API compatibility but not used —
        paper eq. 9 is candidate-independent.
        """
        user_node, news = self._encode_history_graph(history_packed, history_mask)
        return self.user_attention(news, user_node, mask=history_mask)

    def forward(self, packed_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluation: packed history -> single user vector.

        Input: ``(B, H, 3)`` = ``[news_idx | category | subcategory]`` per slot.
        """
        history_mask = self.news_encoder.valid_mask(packed_features)
        user_node, news = self._encode_history_graph(packed_features, history_mask)
        return self.user_attention(news, user_node, mask=history_mask)
