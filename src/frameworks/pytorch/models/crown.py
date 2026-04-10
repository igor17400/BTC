"""CROWN (WWW 2025) — PyTorch implementation.

Faithful reimplementation of the official reference code at
``reference_codes/crown-www25/``, adapted to the NewsReX architecture
(spec-driven config, shared evaluation pipeline, framework adapter).

Key components:
- **News encoder**: Transformer + positional encoding → mean pool →
  category-aware k-intent disentanglement → additive attention →
  title-body cosine similarity → concatenation with category embeddings.
- **User encoder**: Category-aware heterogeneous graph (news nodes +
  category nodes with intra/inter-cluster edges) → multi-layer GraphSAGE
  → candidate-aware scaled dot-product attention.
- **Auxiliary loss**: Category prediction from title intent embeddings.

Reference: "CROWN: Contextualized Relation-aware User Interest Modeling
with Networks", WWW 2025.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import CROWNConfig

from .base import BaseModel

# ======================================================================
# Layers
# ======================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (standard transformer)."""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau-style) attention: tanh(Wx+b) @ q → weighted sum."""

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.affine = nn.Linear(feature_dim, attention_dim)
        self.query = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            features: (B, L, D) — B=batch, L=sequence length, D=feature dim.
            mask: (B, L) optional — True where positions are valid.

        Returns:
            (B, D) weighted sum over the L dimension.
        """
        scores = self.query(torch.tanh(self.affine(features))).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (features * weights).sum(dim=1)


class CandidateAwareAttention(nn.Module):
    """Scaled dot-product attention with candidate queries (paper §3.3).

    For each candidate news, computes a separate user representation by
    attending over the user's history embeddings.
    """

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.key_proj = nn.Linear(feature_dim, attention_dim, bias=True)
        self.query_proj = nn.Linear(feature_dim, attention_dim, bias=True)
        self.scale = math.sqrt(attention_dim)

    def forward(
        self,
        history: torch.Tensor,
        candidates: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            history: (B, H, D) — user history news embeddings.
            candidates: (B, C, D) — candidate news embeddings.
            history_mask: (B, H) — True where history slot is valid.

        Returns:
            (B, C, D) — per-candidate user representations.
        """
        K = self.key_proj(history)  # (B, H, A)
        Q = self.query_proj(candidates)  # (B, C, A)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, C, H)

        if history_mask is not None:
            mask = history_mask.unsqueeze(1).expand_as(scores)  # (B, C, H)
            scores = scores.masked_fill(~mask.bool(), -1e9)

        weights = torch.softmax(scores, dim=-1)  # (B, C, H)
        return torch.bmm(weights, history)  # (B, C, D)


# ======================================================================
# Graph construction (paper §3.2)
# ======================================================================


def build_category_graph(
    history_categories: torch.Tensor,
    history_mask: torch.Tensor,
    num_categories: int,
    max_history: int,
    no_self_connection: bool = False,
    normalization: str = "symmetric",
) -> torch.Tensor:
    """Build the category-aware heterogeneous graph adjacency matrix.

    Graph layout per batch:
        Nodes [0 .. max_history-1] = news nodes
        Nodes [max_history .. max_history+num_categories-1] = category nodes

    Edge types (paper):
        E_n: intra-cluster (news↔news in same category)
        E_p^1: news↔category
        E_p^2: category↔category

    Args:
        history_categories: (B, H) int — category index per history news.
        history_mask: (B, H) bool — True where history slot is valid.
        num_categories: Total number of categories.
        max_history: Maximum history length (H).
        no_self_connection: If True, omit diagonal self-loops.
        normalization: 'symmetric' for D^{-1/2}AD^{-1/2}, 'asymmetric' for D^{-1}A.

    Returns:
        (B, H+C, H+C) normalized adjacency matrix.
    """
    B = history_categories.size(0)
    graph_size = max_history + num_categories
    device = history_categories.device

    adj = torch.zeros(B, graph_size, graph_size, device=device)

    # Self connections (diagonal)
    if not no_self_connection:
        eye = torch.eye(graph_size, device=device).unsqueeze(0).expand(B, -1, -1)
        adj = adj + eye

    for b in range(B):
        cats = history_categories[b]  # (H,)
        mask = history_mask[b]  # (H,)

        # Collect valid (news_idx, category) pairs, clamping to valid range
        valid_news = []
        active_cats = set()
        for i in range(max_history):
            if mask[i]:
                c = int(cats[i].item())
                if c < 0 or c >= num_categories:
                    continue  # skip invalid category indices
                valid_news.append((i, c))
                active_cats.add(c)

        # E_n: intra-cluster edges (news↔news in same category)
        cat_to_news: dict[int, list[int]] = {}
        for news_idx, c in valid_news:
            cat_to_news.setdefault(c, []).append(news_idx)

        for news_list in cat_to_news.values():
            for i in range(len(news_list)):
                for j in range(i + 1, len(news_list)):
                    ni, nj = news_list[i], news_list[j]
                    adj[b, ni, nj] = 1.0
                    adj[b, nj, ni] = 1.0

        # E_p^1: news↔category edges
        for news_idx, c in valid_news:
            cat_node = max_history + c
            adj[b, news_idx, cat_node] = 1.0
            adj[b, cat_node, news_idx] = 1.0

        # E_p^2: category↔category edges
        active_list = list(active_cats)
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                ci = max_history + active_list[i]
                cj = max_history + active_list[j]
                adj[b, ci, cj] = 1.0
                adj[b, cj, ci] = 1.0

    # Degree normalization
    degree = adj.sum(dim=-1).clamp(min=1e-7)  # (B, N)
    if normalization == "symmetric":
        d_inv_sqrt = degree.pow(-0.5)
        # D^{-1/2} A D^{-1/2}
        adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)
    else:
        d_inv = degree.pow(-1.0)
        adj = d_inv.unsqueeze(-1) * adj

    return adj


# ======================================================================
# GraphSAGE layer (paper uses this for the category-aware graph)
# ======================================================================


class GraphSAGEConv(nn.Module):
    """Single GraphSAGE convolution layer for the heterogeneous graph.

    Operates on the full (news + category) node set.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        dropout_rate: Dropout rate.
        normalize: L2-normalize output.
        residual: Add residual connection (skip when dims differ).
        layer_norm: Apply layer norm after update.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout_rate: float = 0.2,
        normalize: bool = True,
        residual: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.do_normalize = normalize
        self.residual = residual and (in_dim == out_dim)
        self.ln = nn.LayerNorm(out_dim) if layer_norm else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) node features.
            adj: (B, N, N) normalized adjacency.

        Returns:
            (B, N, D_out) updated node features.
        """
        # Mean aggregation from neighbors (adj already normalized)
        neigh_agg = torch.bmm(adj, x)  # (B, N, D)
        out = F.relu(self.W_self(x) + self.W_neigh(neigh_agg))
        out = self.dropout(out)

        if self.residual:
            out = out + x

        if self.ln is not None:
            out = self.ln(out)

        if self.do_normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out


class GATConv(nn.Module):
    """GAT convolution layer for the heterogeneous graph.

    Multi-head attention over the adjacency structure. Same interface
    as :class:`GraphSAGEConv` so the user encoder can swap freely.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
        alpha: float = 0.2,
        residual: bool = False,
        layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.alpha = alpha
        self.residual = residual and (in_dim == out_dim)

        self.W = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(out_dim) if layer_norm else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) node features.
            adj: (B, N, N) adjacency (non-zero = edge exists).

        Returns:
            (B, N, out_dim) updated node features.
        """
        B, N, _ = x.shape
        # Project to multi-head space: (B, N, H, head_dim)
        h = self.W(x).view(B, N, self.num_heads, self.head_dim)

        # Attention scores: src_score + dst_score for each edge
        # (B, N, H) for each node
        src_scores = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        dst_scores = (h * self.a_dst).sum(dim=-1)  # (B, N, H)

        # Pairwise: (B, H, N, N) = src[i] + dst[j]
        attn = src_scores.permute(0, 2, 1).unsqueeze(-1) + dst_scores.permute(
            0, 2, 1
        ).unsqueeze(-2)
        attn = F.leaky_relu(attn, negative_slope=self.alpha)

        # Mask non-edges
        adj_mask = (adj > 0).unsqueeze(1)  # (B, 1, N, N)
        attn = attn.masked_fill(~adj_mask, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        # Aggregate: (B, H, N, N) @ (B, H, N, head_dim) → (B, H, N, head_dim)
        h_t = h.permute(0, 2, 1, 3)  # (B, H, N, head_dim)
        out = torch.matmul(attn, h_t)  # (B, H, N, head_dim)

        # Concat heads: (B, N, out_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_dim)
        out = self.dropout(F.relu(out))

        if self.residual:
            out = out + x
        if self.ln is not None:
            out = self.ln(out)

        return out


# ======================================================================
# News encoder (paper §3.1)
# ======================================================================


class CROWNNewsEncoder(nn.Module):
    """CROWN news encoder.

    Pipeline:
        1. Word embedding + dropout + positional encoding
        2. nn.TransformerEncoder (paper: 1 layer, 10 heads)
        3. Mean pooling over sequence
        4. Category-aware concatenation
        5. k-intent disentanglement (k separate FC → ReLU)
        6. Additive attention over k intents
        7. Title-body cosine similarity
        8. Concatenate [title_intent, sim * body_intent, cat_emb, subcat_emb]

    Also computes auxiliary category-prediction loss.
    """

    def __init__(
        self,
        config: CROWNConfig,
        word_embedding: nn.Embedding,
        category_embedding: nn.Embedding,
        subcategory_embedding: nn.Embedding,
    ):
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout_rate)

        # Positional encoding
        self.title_pos = PositionalEncoding(
            config.embedding_size, config.dropout_rate, config.max_title_length
        )
        self.body_pos = PositionalEncoding(
            config.embedding_size, config.dropout_rate, config.max_abstract_length
        )

        # Transformer encoder (paper: 1 layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_size,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.title_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.body_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Category affine: concat(cat, subcat) → category_embedding_dim
        cat_concat_dim = (
            config.category_embedding_dim + config.subcategory_embedding_dim
        )
        self.category_affine = nn.Linear(cat_concat_dim, config.category_embedding_dim)

        # k-intent disentanglement
        intent_input_dim = config.embedding_size + config.category_embedding_dim
        self.intent_layers = nn.ModuleList(
            [
                nn.Linear(intent_input_dim, config.intent_embedding_dim)
                for _ in range(config.intent_num)
            ]
        )

        # Intent attention
        self.title_intent_attn = AdditiveAttention(
            config.intent_embedding_dim, config.attention_dim
        )
        self.body_intent_attn = AdditiveAttention(
            config.intent_embedding_dim, config.attention_dim
        )

        # Auxiliary category predictor
        self.category_predictor = nn.Linear(
            config.intent_embedding_dim,
            1,  # placeholder, set in CROWN.__init__
        )

        # Stored auxiliary loss (accumulated during forward, reset each call)
        self.auxiliary_loss = torch.tensor(0.0)

    def _k_intent_disentangle(self, x: torch.Tensor) -> torch.Tensor:
        """(N, input_dim) → (N, k, intent_dim)"""
        intents = [F.relu(layer(x)).unsqueeze(1) for layer in self.intent_layers]
        return torch.cat(intents, dim=1)

    def forward(
        self,
        title_tokens_or_concat: torch.Tensor,
        abstract_tokens: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
        subcategory_ids: torch.Tensor | None = None,
        compute_aux_loss: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Encode news articles.

        Accepts either explicit arguments or a single concatenated tensor
        ``[title | abstract | category | subcategory]`` (used by the shared
        evaluation adapter).

        Returns:
            (N, news_embedding_dim) news representations.
        """
        cfg = self.config
        if abstract_tokens is None:
            # Concatenated input from evaluation adapter
            concat = title_tokens_or_concat
            title_tokens = concat[:, : cfg.max_title_length]
            abstract_tokens = concat[
                :, cfg.max_title_length : cfg.max_title_length + cfg.max_abstract_length
            ]
            category_ids = concat[:, cfg.max_title_length + cfg.max_abstract_length]
            subcategory_ids = concat[
                :, cfg.max_title_length + cfg.max_abstract_length + 1
            ]
        else:
            title_tokens = title_tokens_or_concat
        # 1. Word embedding + positional encoding
        title_emb = self.dropout(self.word_embedding(title_tokens))
        body_emb = self.dropout(self.word_embedding(abstract_tokens))

        title_emb = self.title_pos(title_emb)
        body_emb = self.body_pos(body_emb)

        # 2. Transformer encoder
        title_enc = self.title_transformer(title_emb)  # (N, T, E)
        body_enc = self.body_transformer(body_emb)  # (N, A, E)

        # 3. Mean pooling
        title_pool = title_enc.mean(dim=1)  # (N, E)
        body_pool = body_enc.mean(dim=1)  # (N, E)

        # 4. Category-aware representation
        cat_emb = self.category_embedding(category_ids)  # (N, cat_dim)
        subcat_emb = self.subcategory_embedding(subcategory_ids)  # (N, subcat_dim)
        cat_repr = self.category_affine(
            torch.cat([cat_emb, subcat_emb], dim=-1)
        )  # (N, cat_dim)

        cat_aware_title = torch.cat([title_pool, cat_repr], dim=-1)
        cat_aware_body = torch.cat([body_pool, cat_repr], dim=-1)

        # 5. k-intent disentanglement
        title_k = self._k_intent_disentangle(cat_aware_title)  # (N, k, intent_dim)
        body_k = self._k_intent_disentangle(cat_aware_body)

        # 6. Attention over intents
        title_intent = self.title_intent_attn(title_k)  # (N, intent_dim)
        body_intent = self.body_intent_attn(body_k)

        # 7. Auxiliary category prediction loss
        if compute_aux_loss:
            logits = self.category_predictor(title_intent)  # (N, num_categories)
            self.auxiliary_loss = F.cross_entropy(logits, category_ids)
        else:
            self.auxiliary_loss = torch.tensor(0.0, device=title_tokens.device)

        # 8. Title-body cosine similarity
        sim = F.cosine_similarity(title_intent, body_intent, dim=-1)
        sim = (sim + 1.0) / 2.0  # scale to [0, 1]

        # 9. Final news representation (full — used by user encoder GNN)
        news_repr = torch.cat(
            [
                title_intent,
                sim.unsqueeze(-1) * body_intent,
                self.dropout(cat_emb),
                self.dropout(subcat_emb),
            ],
            dim=-1,
        )

        # Store title intent for click prediction (paper §4.3:
        # "we use only the title embedding r^n_(T) for click prediction")
        self._last_title_intent = title_intent

        return news_repr  # (N, news_embedding_dim)


# ======================================================================
# User encoder (paper §3.2 + §3.3)
# ======================================================================


class CROWNUserEncoder(nn.Module):
    """CROWN user encoder.

    Pipeline:
        1. Encode history news via shared news encoder
        2. Build category-aware heterogeneous graph
        3. Multi-layer GraphSAGE over the graph
        4. Extract news node embeddings (drop category nodes)
        5. Candidate-aware scaled dot-product attention
    """

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: CROWNNewsEncoder,
        num_categories: int,
    ):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.num_categories = num_categories
        news_emb_dim = news_encoder.news_embedding_dim

        # Category node initial embeddings (learnable)
        self.category_node_emb = nn.Parameter(torch.zeros(num_categories, news_emb_dim))
        nn.init.uniform_(self.category_node_emb, -0.1, 0.1)

        # Multi-layer GNN (GAT or GraphSAGE based on config)
        gnn_cls = GATConv if config.gnn_type == "gat" else GraphSAGEConv
        gnn_kwargs = dict(
            in_dim=news_emb_dim,
            out_dim=news_emb_dim,
            dropout_rate=config.dropout_rate,
            residual=not config.no_gcn_residual,
            layer_norm=config.gcn_layer_norm,
        )
        if config.gnn_type == "gat":
            gnn_kwargs["num_heads"] = config.gat_num_heads
            gnn_kwargs["alpha"] = config.gat_alpha
        else:
            gnn_kwargs["normalize"] = config.sage_normalize
        self.gnn_layers = nn.ModuleList(
            [gnn_cls(**gnn_kwargs) for _ in range(config.graph_num_layers)]
        )

        # Candidate-aware attention (training)
        self.candidate_attention = CandidateAwareAttention(
            feature_dim=news_emb_dim,
            attention_dim=config.user_attention_dim,
        )

        # Additive attention fallback (evaluation — no candidates available)
        self.eval_attention = AdditiveAttention(
            feature_dim=news_emb_dim,
            attention_dim=config.user_attention_dim,
        )

    def _encode_history_graph(
        self,
        history_title: torch.Tensor,
        history_abstract: torch.Tensor,
        history_category: torch.Tensor,
        history_subcategory: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Shared: encode history → GNN → enhanced news embeddings.

        Returns:
            (B, H, D) GNN-enhanced history news embeddings.
        """
        B, H = history_title.shape[:2]
        cfg = self.config

        # 1. Encode history news (TimeDistributed)
        flat_title = history_title.reshape(B * H, -1)
        flat_abstract = history_abstract.reshape(B * H, -1)
        flat_cat = history_category.reshape(B * H)
        flat_subcat = history_subcategory.reshape(B * H)

        flat_news = self.news_encoder(
            flat_title, flat_abstract, flat_cat, flat_subcat, compute_aux_loss=False
        )
        history_news = flat_news.reshape(B, H, -1)  # (B, H, D)

        # 2. Build category-aware graph
        adj = build_category_graph(
            history_categories=history_category,
            history_mask=history_mask,
            num_categories=self.num_categories,
            max_history=H,
            no_self_connection=cfg.no_self_connection,
            normalization=cfg.gcn_normalization_type,
        )

        # 3. Concatenate news nodes + category nodes
        cat_nodes = self.category_node_emb.unsqueeze(0).expand(B, -1, -1)
        graph_nodes = torch.cat([history_news, cat_nodes], dim=1)

        # 4. Multi-layer GraphSAGE
        for gnn in self.gnn_layers:
            graph_nodes = gnn(graph_nodes, adj)

        # 5. Extract news node embeddings (first H nodes)
        return graph_nodes[:, :H, :]

    def forward_with_candidates(
        self,
        history_title: torch.Tensor,
        history_abstract: torch.Tensor,
        history_category: torch.Tensor,
        history_subcategory: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Training: candidate-aware user representations.

        Returns:
            (B, C, D) — per-candidate user representations.
        """
        enhanced = self._encode_history_graph(
            history_title,
            history_abstract,
            history_category,
            history_subcategory,
            history_mask,
        )
        return self.candidate_attention(enhanced, candidate_repr, history_mask)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluation: concatenated history → single user vector.

        Called by the shared evaluation adapter with a concatenated tensor
        ``[title | abstract | category | subcategory]`` per history slot.

        Args:
            inputs: (B, H, title_len + abstract_len + 2)

        Returns:
            (B, D) — single user representation.
        """
        cfg = self.config
        title_len = cfg.max_title_length
        abstract_len = cfg.max_abstract_length

        history_title = inputs[:, :, :title_len]
        history_abstract = inputs[:, :, title_len : title_len + abstract_len]
        history_category = inputs[:, :, title_len + abstract_len]
        history_subcategory = inputs[:, :, title_len + abstract_len + 1]
        history_mask = inputs.any(dim=-1)  # (B, H)

        enhanced = self._encode_history_graph(
            history_title,
            history_abstract,
            history_category,
            history_subcategory,
            history_mask,
        )
        return self.eval_attention(enhanced, mask=history_mask)


# ======================================================================
# Full CROWN model
# ======================================================================


class CROWN(BaseModel):
    """CROWN: Category-aware intent disentanglement + GNN user encoder.

    Training forward pass:
        1. Encode candidates via news encoder (with auxiliary loss).
        2. Encode user via user encoder (candidate-aware attention).
        3. Score = dot product between per-candidate user repr and candidate repr.
        4. Return raw logits (loss function handles softmax).

    Evaluation:
        Uses ``news_encoder`` and ``user_encoder`` separately via the
        shared evaluation pipeline.
    """

    def __init__(
        self,
        processed_news: dict[str, Any],
        config: CROWNConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = CROWNConfig(**kwargs)
        self.config = config
        self.process_user_id = config.process_user_id

        num_categories = int(processed_news.get("num_categories", 18))
        num_subcategories = int(processed_news.get("num_subcategories", 100))

        # Shared word embedding
        vocab_size = processed_news["vocab_size"]
        embeddings_matrix = processed_news["embeddings"]
        self.word_embedding = nn.Embedding(vocab_size, config.embedding_size)
        self.word_embedding.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # Category / subcategory embeddings (paper: uniform init, frozen in ref code)
        self.category_embedding = nn.Embedding(
            num_categories + 1, config.category_embedding_dim
        )
        self.subcategory_embedding = nn.Embedding(
            num_subcategories + 1, config.subcategory_embedding_dim
        )
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subcategory_embedding.weight, -0.1, 0.1)

        # News encoder
        self.news_encoder = CROWNNewsEncoder(
            config,
            self.word_embedding,
            self.category_embedding,
            self.subcategory_embedding,
        )

        # Set category predictor output dim now that we know num_categories
        self.news_encoder.category_predictor = nn.Linear(
            config.intent_embedding_dim, num_categories + 1
        )

        # User encoder
        self.user_encoder = CROWNUserEncoder(
            config, self.news_encoder, num_categories + 1
        )

        # Store alpha for auxiliary loss weighting
        self.alpha = config.alpha

    def _encode_candidates(
        self,
        cand_tokens: torch.Tensor,
        cand_abstract: torch.Tensor,
        cand_category: torch.Tensor,
        cand_subcategory: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode candidate news.

        Returns:
            (full_repr, title_intent) — full (B, C, 900) for user encoder,
            title_intent (B, C, intent_dim) for click prediction (paper §4.3).
        """
        B, C = cand_tokens.shape[:2]
        flat_repr = self.news_encoder(
            cand_tokens.reshape(B * C, -1),
            cand_abstract.reshape(B * C, -1),
            cand_category.reshape(B * C),
            cand_subcategory.reshape(B * C),
            compute_aux_loss=training,
        )
        title_intent = self.news_encoder._last_title_intent  # (B*C, intent_dim)
        return flat_repr.reshape(B, C, -1), title_intent.reshape(B, C, -1)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        training: bool = True,
    ) -> torch.Tensor:
        """Training forward pass. Returns raw logits (B, C).

        The shared evaluator calls ``news_encoder`` and ``user_encoder``
        directly — this method is only for training.
        """
        # Encode candidates (with auxiliary loss)
        cand_full, cand_title_intent = self._encode_candidates(
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            inputs["cand_category"],
            inputs["cand_subcategory"],
            training=training,
        )

        # History mask
        history_mask = inputs["hist_tokens"].any(dim=-1)  # (B, H)

        # Encode user (candidate-aware during training)
        # User encoder uses full news repr for GNN context
        user_repr = self.user_encoder.forward_with_candidates(
            history_title=inputs["hist_tokens"],
            history_abstract=inputs["hist_abstract_tokens"],
            history_category=inputs["hist_category"],
            history_subcategory=inputs["hist_subcategory"],
            history_mask=history_mask,
            candidate_repr=cand_full,
        )  # (B, C, news_emb_dim)

        # Click prediction uses only title intent (paper §4.3)
        # Project user repr to title intent dim for dot product
        scores = (
            user_repr[:, :, : self.config.intent_embedding_dim] * cand_title_intent
        ).sum(dim=-1)

        return scores  # raw logits (B, C)

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Return weighted auxiliary loss from the news encoder."""
        return self.alpha * self.news_encoder.auxiliary_loss
