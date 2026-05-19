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

from ..layers import PLMTokenLookup
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


class UserQueryAttention(nn.Module):
    """Additive attention with the GNN-updated user proxy node as query.

    Implements paper eq. 9:
        z_j = q^T · tanh(W_key · r^n_j + b_key)
        α_j = softmax_j(z_j)
        r^u = Σ_j α_j · r^n_j
    where q is the user proxy node's embedding after the bipartite GNN.
    Candidate-independent — one user vector per behavior, shared across
    all candidates. This makes the user encoder identical in training and
    evaluation, unlike a candidate-aware pool.
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
        """
        Args:
            news: (B, H, D) — GNN-updated history news embeddings.
            user_node: (B, D) — GNN-updated user proxy node.
            mask: (B, H) — True where a history slot is valid.

        Returns:
            (B, D) user representation.
        """
        keys = torch.tanh(self.W_key(news))  # (B, H, A)
        query = self.W_query(user_node).unsqueeze(1)  # (B, 1, A)
        scores = (keys * query).sum(dim=-1)  # (B, H)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, H, 1)
        return (news * weights).sum(dim=1)  # (B, D)


# ======================================================================
# Bipartite user-news GNN layers (paper §3.3, eq. 8)
#
# Graph structure: one learnable user proxy node + H history news nodes,
# fully-connected bipartite (user ↔ every news), no news-news edges.
# Implemented directly as tensor ops — no adjacency matrix, no Python
# loops — so the same structure ports cleanly to JAX/Keras.
# ======================================================================


class BipartiteGATLayer(nn.Module):
    """1-layer GAT on a user↔news bipartite graph with self-loops.

    Each batch element has one user node and H news nodes; user attends over
    {self, all valid news}, each news attends over {self, user}.
    """

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
        """
        Args:
            user: (B, D) — one user proxy per batch slot.
            news: (B, H, D) — history news embeddings.
            news_mask: (B, H) — True where a history slot is valid.

        Returns:
            user_new: (B, D), news_new: (B, H, D).
        """
        B, H, D = news.shape
        NH, HD = self.num_heads, self.head_dim

        Wu = self.W(user).view(B, NH, HD)  # (B, NH, HD)
        Wn = self.W(news).view(B, H, NH, HD)  # (B, H, NH, HD)

        # GAT attention score a^T [W h_i || W h_j] splits into src + dst halves
        src_u = (Wu * self.a_src).sum(dim=-1)  # (B, NH)
        dst_u = (Wu * self.a_dst).sum(dim=-1)
        src_n = (Wn * self.a_src).sum(dim=-1)  # (B, H, NH)
        dst_n = (Wn * self.a_dst).sum(dim=-1)

        neg_inf = torch.finfo(src_u.dtype).min

        # --- User update: attend over {self, all news} ---
        score_u_n = F.leaky_relu(src_u.unsqueeze(1) + dst_n, self.alpha)  # (B, H, NH)
        score_u_u = F.leaky_relu(src_u + dst_u, self.alpha)  # (B, NH)
        score_u_n = score_u_n.masked_fill(~news_mask.unsqueeze(-1).bool(), neg_inf)

        all_u = torch.cat([score_u_n, score_u_u.unsqueeze(1)], dim=1)  # (B, H+1, NH)
        attn_u = self.dropout(torch.softmax(all_u, dim=1))
        u_from_n = (attn_u[:, :H].unsqueeze(-1) * Wn).sum(dim=1)  # (B, NH, HD)
        u_self = attn_u[:, H].unsqueeze(-1) * Wu
        user_new = F.elu(u_from_n + u_self).reshape(B, D)

        # --- News update: each news attends over {self, user} ---
        score_n_u = F.leaky_relu(src_n + dst_u.unsqueeze(1), self.alpha)  # (B, H, NH)
        score_n_n = F.leaky_relu(src_n + dst_n, self.alpha)
        stacked = torch.stack([score_n_u, score_n_n], dim=-2)  # (B, H, 2, NH)
        attn_n = self.dropout(torch.softmax(stacked, dim=-2))
        n_from_u = attn_n[:, :, 0].unsqueeze(-1) * Wu.unsqueeze(1)  # (B, H, NH, HD)
        n_self = attn_n[:, :, 1].unsqueeze(-1) * Wn
        news_new = F.elu(n_from_u + n_self).reshape(B, H, D)

        return user_new, news_new


class BipartiteSAGELayer(nn.Module):
    """1-layer GraphSAGE on a user↔news bipartite graph.

    Mean aggregator; user pools masked mean of news, each news pools the
    single user neighbor. Matches the GraphSAGE variant shipped in the
    reference code (``userEncoders.py`` CROWN class).
    """

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
        m = news_mask.unsqueeze(-1).to(news.dtype)  # (B, H, 1)
        news_count = m.sum(dim=1).clamp(min=1.0)  # (B, 1)
        news_mean = (news * m).sum(dim=1) / news_count  # (B, D)

        user_new = F.relu(self.W_self(user) + self.W_neigh(news_mean))
        news_new = F.relu(
            self.W_self(news) + self.W_neigh(user.unsqueeze(1).expand_as(news))
        )
        user_new = F.normalize(user_new, p=2, dim=-1)
        news_new = F.normalize(news_new, p=2, dim=-1)
        return self.dropout(user_new), self.dropout(news_new)


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
        word_embedding: nn.Module,
        category_embedding: nn.Embedding,
        subcategory_embedding: nn.Embedding,
        *,
        encoder_type: str = "glove",
        abstract_embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.encoder_type = encoder_type
        # GloVe: a single shared ``nn.Embedding`` used for both views.
        # PLM:   ``word_embedding`` is a ``PLMTokenLookup`` over the title
        #        cache, ``abstract_embedding`` is a second lookup over the
        #        abstract cache. Both are indexed by parsed news_idx.
        self.word_embedding = word_embedding
        self.abstract_embedding = abstract_embedding
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

        GloVe mode accepts either explicit arguments
        ``(title_tokens, abstract_tokens, category_ids, subcategory_ids)``
        or a single concatenated tensor
        ``[title | abstract | category | subcategory]`` (eval adapter).

        PLM mode expects either explicit
        ``(news_idx, None, category_ids, subcategory_ids)`` or a
        concatenated ``[news_idx | category | subcategory]`` tensor —
        both title and abstract are looked up from the same news_idx
        via separate PLM caches.

        Returns:
            (N, news_embedding_dim) news representations.
        """
        cfg = self.config
        if self.encoder_type == "glove":
            if abstract_tokens is None:
                concat = title_tokens_or_concat
                title_tokens = concat[:, : cfg.max_title_length]
                abstract_tokens = concat[
                    :,
                    cfg.max_title_length : cfg.max_title_length
                    + cfg.max_abstract_length,
                ]
                category_ids = concat[:, cfg.max_title_length + cfg.max_abstract_length]
                subcategory_ids = concat[
                    :, cfg.max_title_length + cfg.max_abstract_length + 1
                ]
            else:
                title_tokens = title_tokens_or_concat

            title_emb = self.dropout(self.word_embedding(title_tokens))
            body_emb = self.dropout(self.word_embedding(abstract_tokens))
            title_mask = title_tokens != 0
            body_mask = abstract_tokens != 0
        else:
            # PLM: ``title_tokens_or_concat`` is either ``news_idx`` (B,)
            # int tensor, or a packed ``[news_idx | category | subcategory]``
            # (B, 3) tensor from the eval adapter. Either way we extract
            # news_idx as the first / only column.
            if abstract_tokens is None and title_tokens_or_concat.dim() == 2:
                concat = title_tokens_or_concat
                news_idx = concat[:, 0].long()
                category_ids = concat[:, 1].long()
                subcategory_ids = concat[:, 2].long()
            else:
                news_idx = title_tokens_or_concat.long()
            title_feats, title_mask = self.word_embedding(news_idx)
            body_feats, body_mask = self.abstract_embedding(news_idx)
            title_emb = self.dropout(title_feats)
            body_emb = self.dropout(body_feats)
            title_mask = title_mask.bool()
            body_mask = body_mask.bool()

        # 1. Positional encoding
        title_emb = self.title_pos(title_emb)
        body_emb = self.body_pos(body_emb)

        # 2. Transformer encoder
        title_enc = self.title_transformer(title_emb)  # (N, T, E)
        body_enc = self.body_transformer(body_emb)  # (N, A, E)

        # 3. Mask-weighted mean pooling (under GloVe, padding tokens
        # embed to near-zero so masking is near-equivalent to mean; under
        # PLM it matters since BERT padding embeddings are non-zero).
        title_w = title_mask.to(title_enc.dtype).unsqueeze(-1)
        body_w = body_mask.to(body_enc.dtype).unsqueeze(-1)
        title_pool = (title_enc * title_w).sum(dim=1) / title_w.sum(dim=1).clamp(
            min=1.0
        )
        body_pool = (body_enc * body_w).sum(dim=1) / body_w.sum(dim=1).clamp(min=1.0)

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
    """CROWN user encoder (paper §3.3).

    Pipeline:
        1. Encode history news via shared news encoder.
        2. Initialise a learnable user proxy node.
        3. ``graph_num_layers`` layers of bipartite GNN (GAT or GraphSAGE)
           over the user↔news graph — returns both the updated user proxy
           and the updated news nodes.
        4. Additive attention (paper eq. 9) with the updated user proxy as
           query, pooling the updated news nodes into a single user vector.
           Candidate-independent — same mechanism in training and eval.
    """

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: CROWNNewsEncoder,
    ):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        news_emb_dim = news_encoder.news_embedding_dim

        # Learnable user proxy node (paper: randomly initialised)
        self.user_node = nn.Parameter(torch.empty(news_emb_dim))
        nn.init.uniform_(self.user_node, -0.1, 0.1)

        # Bipartite GNN stack
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

        # Shared user-query attention — used for both train and eval.
        self.user_attention = UserQueryAttention(
            feature_dim=news_emb_dim,
            attention_dim=config.user_attention_dim,
        )

    def _encode_history_graph(
        self,
        history_title: torch.Tensor,
        history_abstract: torch.Tensor | None,
        history_category: torch.Tensor,
        history_subcategory: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode history → bipartite GNN → (user_node, news_nodes).

        Under PLM, ``history_title`` is the per-slot ``news_idx``
        ``(B, H)`` int tensor and ``history_abstract`` is ``None`` — the
        news encoder looks up both title and abstract features from the
        same id via separate PLM caches.

        Returns:
            user_node: (B, D) GNN-updated user proxy.
            news: (B, H, D) GNN-updated history news embeddings.
        """
        B, H = history_title.shape[:2]

        flat_title = (
            history_title.reshape(B * H, -1)
            if history_title.dim() > 2
            else history_title.reshape(B * H)
        )
        flat_abstract = (
            history_abstract.reshape(B * H, -1)
            if history_abstract is not None
            else None
        )
        flat_cat = history_category.reshape(B * H)
        flat_subcat = history_subcategory.reshape(B * H)

        flat_news = self.news_encoder(
            flat_title, flat_abstract, flat_cat, flat_subcat, compute_aux_loss=False
        )
        news = flat_news.reshape(B, H, -1)  # (B, H, D)

        user = self.user_node.unsqueeze(0).expand(B, -1).contiguous()  # (B, D)
        for gnn in self.gnn_layers:
            user, news = gnn(user, news, history_mask)
        return user, news

    def forward_with_candidates(
        self,
        history_title: torch.Tensor,
        history_abstract: torch.Tensor | None,
        history_category: torch.Tensor,
        history_subcategory: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Training: one user representation per behavior.

        ``candidate_repr`` is accepted for API compatibility but not used —
        paper eq. 9 is candidate-independent.

        Returns:
            (B, D) user representation.
        """
        user_node, news = self._encode_history_graph(
            history_title,
            history_abstract,
            history_category,
            history_subcategory,
            history_mask,
        )
        return self.user_attention(news, user_node, mask=history_mask)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluation: concatenated history → single user vector.

        GloVe: input is ``(B, H, T+A+2)`` packed
        ``[title | abstract | category | subcategory]`` per history slot.
        PLM:   input is ``(B, H, 3)`` packed
        ``[news_idx | category | subcategory]`` per history slot — title
        and abstract are looked up from ``news_idx``.
        """
        cfg = self.config
        encoder_type = self.news_encoder.encoder_type

        if encoder_type == "glove":
            title_len = cfg.max_title_length
            abstract_len = cfg.max_abstract_length
            history_title = inputs[:, :, :title_len]
            history_abstract = inputs[:, :, title_len : title_len + abstract_len]
            history_category = inputs[:, :, title_len + abstract_len]
            history_subcategory = inputs[:, :, title_len + abstract_len + 1]
        else:
            history_title = inputs[:, :, 0].long()  # news_idx (B, H)
            history_abstract = None
            history_category = inputs[:, :, 1].long()
            history_subcategory = inputs[:, :, 2].long()
        history_mask = inputs.any(dim=-1)  # (B, H)

        user_node, news = self._encode_history_graph(
            history_title,
            history_abstract,
            history_category,
            history_subcategory,
            history_mask,
        )
        return self.user_attention(news, user_node, mask=history_mask)


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
        encoder_type = getattr(getattr(config, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        # Word / token feature source
        if encoder_type == "glove":
            vocab_size = processed_news["vocab_size"]
            embeddings_matrix = processed_news["embeddings"]
            self.word_embedding: nn.Module = nn.Embedding(
                vocab_size, config.embedding_size
            )
            self.word_embedding.weight = nn.Parameter(
                torch.tensor(embeddings_matrix, dtype=torch.float32)
            )
            self.abstract_embedding: nn.Module | None = None
        else:
            # CROWN is two-view: it needs a separate PLM cache for
            # title (max_title_length tokens) and abstract
            # (max_abstract_length tokens). The runner attaches both
            # under ``plm_token_embeddings_by_id`` (title) and
            # ``plm_abstract_token_embeddings_by_id`` (abstract).
            if "plm_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"CROWN with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id'] (title cache)."
                )
            if "plm_abstract_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"CROWN with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_abstract_token_embeddings_by_id'] "
                    "(abstract cache). Set encoder.text_field_abstract=abstract."
                )
            plm_dim = int(processed_news["plm_dim"])
            self.word_embedding = PLMTokenLookup(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                plm_dim=plm_dim,
                output_dim=config.embedding_size,
            )
            self.abstract_embedding = PLMTokenLookup(
                processed_news["plm_abstract_token_embeddings_by_id"],
                processed_news["plm_abstract_attention_mask_by_id"],
                plm_dim=plm_dim,
                output_dim=config.embedding_size,
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
            encoder_type=encoder_type,
            abstract_embedding=self.abstract_embedding,
        )

        # Set category predictor output dim now that we know num_categories
        self.news_encoder.category_predictor = nn.Linear(
            config.intent_embedding_dim, num_categories + 1
        )

        # User encoder
        self.user_encoder = CROWNUserEncoder(config, self.news_encoder)

        # Store alpha for auxiliary loss weighting
        self.alpha = config.alpha

    def _encode_candidates(
        self,
        cand_tokens: torch.Tensor,
        cand_abstract: torch.Tensor | None,
        cand_category: torch.Tensor,
        cand_subcategory: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode candidate news.

        Under PLM, ``cand_tokens`` carries parsed ``news_idx`` ``(B, C)``
        and ``cand_abstract`` is ``None`` (both views looked up by id).

        Returns:
            (full_repr, title_intent) — full (B, C, 900) for user encoder,
            title_intent (B, C, intent_dim) for click prediction (paper §4.3).
        """
        B, C = cand_tokens.shape[:2]
        flat_title = (
            cand_tokens.reshape(B * C, -1)
            if cand_tokens.dim() > 2
            else cand_tokens.reshape(B * C)
        )
        flat_abstract = (
            cand_abstract.reshape(B * C, -1) if cand_abstract is not None else None
        )
        flat_repr = self.news_encoder(
            flat_title,
            flat_abstract,
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

        GloVe keys: ``cand_tokens`` ``(B, C, T)``,
        ``cand_abstract_tokens`` ``(B, C, A)``, ``cand_category`` ``(B, C)``,
        ``cand_subcategory`` ``(B, C)`` (and ``hist_*`` analogues).

        PLM keys: ``cand_features`` ``(B, C)`` parsed news_idx (title +
        abstract looked up from the cache), ``cand_category``,
        ``cand_subcategory``, ``hist_features``, ``hist_category``,
        ``hist_subcategory``.

        The shared evaluator calls ``news_encoder`` and ``user_encoder``
        directly — this method is only for training.
        """
        if self.encoder_type == "glove":
            cand_tokens = inputs["cand_tokens"]
            cand_abstract = inputs["cand_abstract_tokens"]
            hist_title = inputs["hist_tokens"]
            hist_abstract = inputs["hist_abstract_tokens"]
        else:
            cand_tokens = inputs["cand_features"]
            cand_abstract = None
            hist_title = inputs["hist_features"]
            hist_abstract = None

        # Encode candidates (with auxiliary loss)
        cand_full, _ = self._encode_candidates(
            cand_tokens,
            cand_abstract,
            inputs["cand_category"],
            inputs["cand_subcategory"],
            training=training,
        )
        # Snapshot now — history encoding below calls news_encoder with
        # compute_aux_loss=False, which overwrites self.auxiliary_loss.
        self._aux_loss_snapshot = self.news_encoder.auxiliary_loss

        # History validity mask: under GloVe a slot is valid iff any
        # token is non-zero; under PLM iff the news_idx is non-zero.
        if hist_title.dim() == 3:
            history_mask = hist_title.any(dim=-1)
        else:
            history_mask = hist_title != 0

        user_repr = self.user_encoder.forward_with_candidates(
            history_title=hist_title,
            history_abstract=hist_abstract,
            history_category=inputs["hist_category"],
            history_subcategory=inputs["hist_subcategory"],
            history_mask=history_mask,
            candidate_repr=cand_full,
        )  # (B, news_emb_dim) — candidate-independent (paper eq. 9)

        # Broadcast user rep over candidates, full-vector dot product.
        scores = (user_repr.unsqueeze(1) * cand_full).sum(dim=-1)
        return scores

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Return weighted auxiliary loss from the news encoder."""
        return self.alpha * self._aux_loss_snapshot
