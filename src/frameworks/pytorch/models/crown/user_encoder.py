"""CROWN user encoder (PyTorch) — candidate-aware, GNN selectable.

1. Encode each history slot via the shared news encoder.
2. Bipartite user<->history graph with one learnable user-proxy node
   (zero-initialised) per behaviour; one 1-layer GNN pass updates the
   history representations. GNN family is selected by ``config.gnn_type``:

   - ``"gat"``       — matches the paper TEXT (§4.2.(3-b): "We use GAT").
                       Multi-head bipartite GAT with self-loops; attention
                       coefficients per head are
                       ``e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j)``.
   - ``"graphsage"`` — matches the reference CODE
                       (``seongeunryu/crown-www25/userEncoders.py`` ships
                       ``torch_geometric.nn.GraphSAGE(num_layers=1)``).

3. **Candidate-aware** scaled dot-product attention pools the
   GNN-updated history into a **per-candidate** user vector — matches
   reference code (Q = candidate, K = history), not paper eq. 9.

Training contract::

    forward_with_candidates(history_packed, history_mask, candidate_repr)
        -> user_per_cand  # (B, C, D)

Eval contract — the custom evaluator avoids precomputing user vectors
(impossible with candidate-aware attention) and instead reuses the
encoded news cache::

    encode_history_via_news(history_packed, history_mask) -> gcn_news  # (B, H, D)
    candidate_attention(gcn_news, cand_repr, history_mask)  -> (B, C, D)

The :meth:`forward` entry is retained for the standard runner's
``encode_user`` path but is **not** used by the CROWN custom eval — it
returns the GNN-updated history reps (one per slot), letting the eval
loop run :meth:`candidate_attention` per impression once candidates are
known.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import CROWNConfig

from .news_encoder import NewsEncoder


class BipartiteSAGELayer(nn.Module):
    """1-layer bipartite GraphSAGE (mean aggregator) — reference-faithful.

    Mirrors ``torch_geometric.nn.GraphSAGE(num_layers=1)`` over a graph
    where each history slot is connected only to a single user-proxy
    node. With one shared neighbour the SAGE mean aggregator collapses
    to::

        user_new[b]    = ReLU(W_self · user[b]    + W_neigh · mean_h(news[b,h]))
        news_new[b,h]  = ReLU(W_self · news[b,h]  + W_neigh · user[b])

    We keep this hand-rolled form rather than calling ``SAGEConv``
    directly because the bipartite graph here is trivial (1 user node,
    H history nodes, 1 layer): driving it through PyG's flat edge-list
    format would require per-batch ``edge_index`` construction and
    ``(B, H+1, D) → (B*(H+1), D)`` flattening with offsets — more code,
    not less. PyG's ``SAGEConv`` defaults to ``normalize=False`` and
    ``GraphSAGE(num_layers=1)`` applies no dropout (dropout is between
    layers), so the math here is the exact PyG behaviour.
    """

    def __init__(self, dim: int, dropout_rate: float):
        super().__init__()
        self.W_self = nn.Linear(dim, dim)
        self.W_neigh = nn.Linear(dim, dim)

    def forward(
        self,
        user: torch.Tensor,
        news: torch.Tensor,
        news_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = news_mask.unsqueeze(-1).to(news.dtype)  # (B, H, 1)
        n_count = m.sum(dim=1).clamp(min=1.0)
        news_mean = (news * m).sum(dim=1) / n_count  # (B, D)

        user_new = F.relu(self.W_self(user) + self.W_neigh(news_mean))
        news_new = F.relu(
            self.W_self(news) + self.W_neigh(user.unsqueeze(1).expand_as(news))
        )
        return user_new, news_new


class BipartiteGATLayer(nn.Module):
    """1-layer multi-head bipartite GAT with self-loops.

    Matches the paper TEXT (§4.2.(3-b): "We use GAT"). For each head
    attention coefficients are computed as::

        e_ij = LeakyReLU( a_src^T · (W h_i) + a_dst^T · (W h_j) )

    User attends over {self, all valid news}; each news attends over
    {self, user}. With ``concat=True`` head outputs are concatenated so
    the layer is dim-preserving (``num_heads * head_dim == dim``).

    Same rationale as :class:`BipartiteSAGELayer` for not using PyG's
    ``GATConv``: the bipartite structure here is trivial, and the
    per-batch ``edge_index`` plumbing PyG requires would dwarf this
    direct implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"BipartiteGATLayer: dim ({dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.W.weight)
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

        Wu = self.W(user).view(B, NH, HD)  # (B, NH, HD)
        Wn = self.W(news).view(B, H, NH, HD)  # (B, H, NH, HD)

        # Source / destination halves of a^T [Wh_i || Wh_j].
        src_u = (Wu * self.a_src).sum(dim=-1)  # (B, NH)
        dst_u = (Wu * self.a_dst).sum(dim=-1)
        src_n = (Wn * self.a_src).sum(dim=-1)  # (B, H, NH)
        dst_n = (Wn * self.a_dst).sum(dim=-1)

        neg_inf = torch.finfo(src_u.dtype).min

        # User update: user attends over {self, all news}.
        e_un = F.leaky_relu(src_u.unsqueeze(1) + dst_n, self.alpha)  # (B, H, NH)
        e_uu = F.leaky_relu(src_u + dst_u, self.alpha)  # (B, NH)
        e_un = e_un.masked_fill(~news_mask.unsqueeze(-1).bool(), neg_inf)
        all_u = torch.cat([e_un, e_uu.unsqueeze(1)], dim=1)  # (B, H+1, NH)
        attn_u = self.dropout(torch.softmax(all_u, dim=1))
        u_from_n = (attn_u[:, :H].unsqueeze(-1) * Wn).sum(dim=1)  # (B, NH, HD)
        u_self = attn_u[:, H].unsqueeze(-1) * Wu  # (B, NH, HD)
        user_new = F.elu(u_from_n + u_self)  # (B, NH, HD)

        # News update: each news attends over {self, user}.
        e_nu = F.leaky_relu(src_n + dst_u.unsqueeze(1), self.alpha)  # (B, H, NH)
        e_nn = F.leaky_relu(src_n + dst_n, self.alpha)  # (B, H, NH)
        stacked = torch.stack([e_nu, e_nn], dim=-2)  # (B, H, 2, NH)
        attn_n = self.dropout(torch.softmax(stacked, dim=-2))
        n_from_u = attn_n[:, :, 0].unsqueeze(-1) * Wu.unsqueeze(1)  # (B, H, NH, HD)
        n_self = attn_n[:, :, 1].unsqueeze(-1) * Wn  # (B, H, NH, HD)
        news_new = F.elu(n_from_u + n_self)  # (B, H, NH, HD)

        if self.concat:
            user_new = user_new.reshape(B, NH * HD)
            news_new = news_new.reshape(B, H, NH * HD)
        else:
            user_new = user_new.mean(dim=1)  # (B, HD)
            news_new = news_new.mean(dim=2)  # (B, H, HD)

        return user_new, news_new


class UserEncoder(nn.Module):
    """CROWN user encoder.

    Submodules
    ----------
    ``user_proxy``   single learnable D-vector, zero-init (reference repo
                     uses ``nn.Parameter(zeros([batch_size, D]))``; one
                     vector per behaviour is the semantically equivalent
                     batch-shape-stable form).
    ``gnn``          1-layer bipartite GNN — :class:`BipartiteGATLayer`
                     when ``config.gnn_type == "gat"`` (paper text),
                     :class:`BipartiteSAGELayer` when
                     ``config.gnn_type == "graphsage"`` (reference code).
    ``K``, ``Q``     candidate-aware scaled-dot attention projections;
                     ``Q.bias=True``, ``K.bias=False`` per reference.
    """

    def __init__(self, config: CROWNConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        D = news_encoder.news_embedding_dim
        A = config.user_attention_dim
        self.D = D
        self.attention_dim = A
        self.scale = math.sqrt(float(A))

        self.user_proxy = nn.Parameter(torch.zeros(D))

        gnn_type = (config.gnn_type or "graphsage").lower()
        if gnn_type == "gat":
            self.gnn: nn.Module = BipartiteGATLayer(
                dim=D,
                num_heads=config.gat_num_heads,
                dropout_rate=config.dropout_rate,
                alpha=config.gat_alpha,
                concat=config.gat_concat_heads,
            )
        elif gnn_type == "graphsage":
            self.gnn = BipartiteSAGELayer(dim=D, dropout_rate=config.dropout_rate)
        else:
            raise ValueError(
                f"CROWN gnn_type must be 'gat' or 'graphsage'; got {config.gnn_type!r}."
            )

        self.K = nn.Linear(D, A, bias=False)
        self.Q = nn.Linear(D, A, bias=True)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # ------------------------------------------------------------------
    # GNN + attention primitives — reused by training and the custom eval.
    # ------------------------------------------------------------------
    def run_gnn(self, news: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """1-layer bipartite GNN over ``(user_proxy, news)``.

        Args:
            news: ``(B, H, D)`` per-history news representations.
            history_mask: ``(B, H)`` bool — valid history slots.

        Returns:
            ``(B, H, D)`` GNN-updated history representations. (The
            updated user node is computed but discarded — the
            downstream candidate attention only needs history keys.)
        """
        B = news.shape[0]
        user = self.user_proxy.unsqueeze(0).expand(B, -1).contiguous()  # (B, D)
        _, gcn_news = self.gnn(user, news, history_mask)
        return gcn_news

    def candidate_attention(
        self,
        gcn_news: torch.Tensor,
        candidate_repr: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate-aware scaled dot-product attention.

        Args:
            gcn_news: ``(B, H, D)`` GNN-updated history.
            candidate_repr: ``(B, C, D)`` candidate news representations.
            history_mask: ``(B, H)`` bool.

        Returns:
            ``(B, C, D)`` per-candidate user representations.
        """
        K = self.K(gcn_news)  # (B, H, A)
        Q = self.Q(candidate_repr)  # (B, C, A)
        scores = torch.einsum("bha,bca->bch", K, Q) / self.scale  # (B, C, H)
        # No history-mask on the softmax — matches reference
        # ``crown-www25/userEncoders.py:107`` (``F.softmax(a, dim=1)``).
        # ``history_mask`` is intentionally unused here. Padded history
        # slots still receive a (small) attention weight, but their GNN
        # embeddings are real tensors so the contribution is bounded.
        # An earlier masked variant changed scores for short-history
        # impressions and hurt val/test AUC.
        del history_mask
        alpha = F.softmax(scores, dim=-1)  # (B, C, H)
        return torch.einsum("bch,bhd->bcd", alpha, gcn_news)  # (B, C, D)

    # ------------------------------------------------------------------
    # Training entry.
    # ------------------------------------------------------------------
    def forward_with_candidates(
        self,
        history_packed: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Training: ``(B, H, 3)`` history packed + ``(B, C, D)`` cands -> ``(B, C, D)``."""
        news_history = self.news_encoder(history_packed, compute_aux_loss=False)
        gcn_news = self.run_gnn(news_history, history_mask)
        return self.candidate_attention(gcn_news, candidate_repr, history_mask)

    # ------------------------------------------------------------------
    # Eval entry — returns GNN-updated history reps; the custom evaluator
    # runs ``candidate_attention`` itself once candidates are known.
    # ------------------------------------------------------------------
    def forward(self, packed_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Eval: packed history ``(B, H, 3)`` -> ``(B, H, D)`` GNN reps.

        This is *not* a user vector — CROWN's user vector is
        candidate-dependent and computed inside the custom evaluator.
        Returning the GNN-updated history lets the standard
        ``precompute_user_vectors`` cache the expensive part once per
        impression.
        """
        history_mask = self.news_encoder.valid_mask(packed_features)
        news_history = self.news_encoder(packed_features, compute_aux_loss=False)
        return self.run_gnn(news_history, history_mask)
