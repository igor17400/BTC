"""CROWN (Category-guided intent disentanglement + GNN hybrid user repr) -- PyTorch.

Ported from the Keras implementation in src/frameworks/keras/models/crown.py.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import CROWNConfig
from ..layers import (
    AdditiveAttention,
    GraphAttentionLayer,
    GraphSAGELayer,
    MultiHeadAttentionBlock,
    PositionalEncoding,
)
from .base import BaseModel


# ------------------------------------------------------------------
# Sub-encoders
# ------------------------------------------------------------------


class CategoryPredictor(nn.Module):
    """Category prediction layer for auxiliary task."""

    def __init__(self, num_categories: int, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_categories)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


class NewsEncoder(nn.Module):
    """News encoder for CROWN.

    Pipeline:
        1. Word embedding + dropout
        2. Positional encoding
        3. Multi-head attention block (MAB)
        4. Mean pooling
        5. Category-aware k-intent disentanglement
        6. Additive attention over k intents
        7. Title-body cosine consistency
        8. Concatenate [title_intent, sim * body_intent, cat_emb, subcat_emb]
    """

    def __init__(
        self,
        config: CROWNConfig,
        embedding_layer: nn.Embedding,
        category_embedding: nn.Embedding,
        subcategory_embedding: nn.Embedding,
        category_predictor: CategoryPredictor,
    ):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding
        self.category_predictor = category_predictor

        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout_rate)

        # Positional encoding for title and body
        self.title_pos_encoder = PositionalEncoding(
            d_model=config.embedding_size,
            dropout_rate=config.dropout_rate,
            max_len=config.max_title_length,
        )
        self.body_pos_encoder = PositionalEncoding(
            d_model=config.embedding_size,
            dropout_rate=config.dropout_rate,
            max_len=config.max_abstract_length,
        )

        # MAB encoders (single block each, matching Keras)
        self.title_mab = MultiHeadAttentionBlock(
            dim_out=config.embedding_size,
            num_heads=config.num_heads,
            use_layer_norm=True,
            dropout_rate=config.dropout_rate,
        )
        self.body_mab = MultiHeadAttentionBlock(
            dim_out=config.embedding_size,
            num_heads=config.num_heads,
            use_layer_norm=True,
            dropout_rate=config.dropout_rate,
        )

        # Category affine: projects concat(cat_emb, subcat_emb) -> category_embedding_dim
        self.category_affine = nn.Linear(
            config.category_embedding_dim + config.subcategory_embedding_dim,
            config.category_embedding_dim,
        )

        # k-intent disentanglement: k separate Dense layers
        # Input to each intent layer = embedding_size (mean-pooled MAB) + category_embedding_dim
        intent_input_dim = config.embedding_size + config.category_embedding_dim
        self.intent_layers = nn.ModuleList(
            [
                nn.Linear(intent_input_dim, config.intent_embedding_dim)
                for _ in range(config.intent_num)
            ]
        )

        # Intent attention (additive) over k intents
        self.title_intent_attention = AdditiveAttention(
            input_dim=config.intent_embedding_dim,
            query_vec_dim=config.attention_dim,
        )
        self.body_intent_attention = AdditiveAttention(
            input_dim=config.intent_embedding_dim,
            query_vec_dim=config.attention_dim,
        )

    # ----- helpers --------------------------------------------------------

    def k_intent_disentangle(self, news_embedding: torch.Tensor) -> torch.Tensor:
        """Apply k-FC layers for k-intent disentanglement.

        Args:
            news_embedding: (N, embedding_size + category_embedding_dim)

        Returns:
            (N, k, intent_embedding_dim)
        """
        intents = [
            F.relu(layer(news_embedding)).unsqueeze(1) for layer in self.intent_layers
        ]
        return torch.cat(intents, dim=1)  # (N, k, intent_dim)

    @staticmethod
    def similarity_compute(
        title: torch.Tensor, body: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity scaled to [0, 1].

        Args:
            title: (N, D)
            body:  (N, D)

        Returns:
            (N,)
        """
        title_n = F.normalize(title, p=2, dim=-1)
        body_n = F.normalize(body, p=2, dim=-1)
        cosine = (title_n * body_n).sum(dim=-1)
        return (cosine + 1.0) / 2.0

    # ----- forward --------------------------------------------------------

    def forward(
        self, inputs: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode news from concatenated input.

        Args:
            inputs: (N, title_len + abstract_len + 2)  int token ids
                    Layout: [title_tokens | abstract_tokens | category_id | subcategory_id]

        Returns:
            (N, news_embedding_dim) news representations.
        """
        if not training:
            self.eval()
        else:
            self.train()

        cfg = self.config
        title_tokens = inputs[:, : cfg.max_title_length]
        abstract_tokens = inputs[
            :, cfg.max_title_length : cfg.max_title_length + cfg.max_abstract_length
        ]
        category_id = inputs[
            :, cfg.max_title_length + cfg.max_abstract_length
        ]
        subcategory_id = inputs[
            :, cfg.max_title_length + cfg.max_abstract_length + 1
        ]

        # Word embeddings + dropout
        title_w = self.dropout(self.embedding_layer(title_tokens))
        body_w = self.dropout(self.embedding_layer(abstract_tokens))

        # Positional encoding
        title_p = self.title_pos_encoder(title_w)
        body_p = self.body_pos_encoder(body_w)

        # MAB
        title_t = self.title_mab(title_p)
        body_t = self.body_mab(body_p)

        # Mean pooling
        title_embedding = title_t.mean(dim=1)  # (N, E)
        body_embedding = body_t.mean(dim=1)

        # Category embeddings
        cat_emb = self.category_embedding(category_id)       # (N, cat_dim)
        subcat_emb = self.subcategory_embedding(subcategory_id)  # (N, subcat_dim)

        # Category-aware representation
        category_concat = torch.cat([cat_emb, subcat_emb], dim=-1)
        category_repr = self.category_affine(category_concat)    # (N, cat_dim)

        category_aware_title = torch.cat([title_embedding, category_repr], dim=-1)
        category_aware_body = torch.cat([body_embedding, category_repr], dim=-1)

        # k-intent disentanglement
        title_k = self.k_intent_disentangle(category_aware_title)   # (N, k, intent_dim)
        body_k = self.k_intent_disentangle(category_aware_body)

        # Attention over intents
        title_intent = self.title_intent_attention(title_k)   # (N, intent_dim)
        body_intent = self.body_intent_attention(body_k)

        # Title-body consistency
        sim = self.similarity_compute(title_intent, body_intent)  # (N,)

        # Final news representation
        news_repr = torch.cat(
            [
                title_intent,
                sim.unsqueeze(-1) * body_intent,
                self.dropout(cat_emb),
                self.dropout(subcat_emb),
            ],
            dim=-1,
        )
        return news_repr  # (N, news_embedding_dim)


class UserEncoder(nn.Module):
    """User encoder for CROWN.

    Pipeline:
        1. TimeDistributed(NewsEncoder) over browsing history
        2. Build bipartite user-news adjacency matrix
        3. Project to graph space, apply GNN (GraphSAGE or GAT)
        4. Additive attention over GNN-enhanced news embeddings
        5. Hybrid representation = concat(gnn_user_emb, attn_user_emb) -> projection
    """

    def __init__(self, config: CROWNConfig, news_encoder: NewsEncoder):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder

        news_emb_dim = news_encoder.news_embedding_dim

        # Additive attention for user representation
        self.user_attention = AdditiveAttention(
            input_dim=config.graph_hidden_dim,
            query_vec_dim=config.attention_dim,
        )

        # GNN layer
        if config.gnn_type == "graphsage":
            self.gnn_layer = GraphSAGELayer(
                user_dim=config.graph_hidden_dim,
                news_dim=config.graph_hidden_dim,
                units=config.graph_hidden_dim,
                aggregator=config.sage_aggregator,
                dropout_rate=config.dropout_rate,
                normalize=config.sage_normalize,
            )
        elif config.gnn_type == "gat":
            self.gnn_layer = GraphAttentionLayer(
                user_dim=config.graph_hidden_dim,
                news_dim=config.graph_hidden_dim,
                units=config.graph_hidden_dim,
                num_heads=config.gat_num_heads,
                dropout_rate=config.dropout_rate,
                alpha=config.gat_alpha,
                concat_heads=config.gat_concat_heads,
            )
        else:
            raise ValueError(
                f"Unknown GNN type: {config.gnn_type}. Choose 'graphsage' or 'gat'."
            )

        # Project news embeddings into graph space
        self.user_node_init = nn.Linear(news_emb_dim, config.graph_hidden_dim)
        self.news_graph_projection = nn.Linear(news_emb_dim, config.graph_hidden_dim)

        # Final projection: concat(gnn_user, attention_user) -> news_emb_dim
        self.final_user_projection = nn.Linear(
            config.graph_hidden_dim * 2, news_emb_dim
        )

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def create_bipartite_graph(history_mask: torch.Tensor) -> torch.Tensor:
        """Build adjacency matrix for the bipartite user-news graph.

        Layout (per batch element):
            Rows/cols [0..H-1] = news nodes, [H] = user node.

        Args:
            history_mask: (B, H) float mask (1=valid, 0=pad).

        Returns:
            (B, H+1, H+1) adjacency matrix.
        """
        B, H = history_mask.shape

        # news-to-news: no connections
        hist_to_hist = torch.zeros(B, H, H, device=history_mask.device)

        # news-to-user / user-to-news: based on mask
        hist_to_user = history_mask.unsqueeze(-1)       # (B, H, 1)
        user_to_hist = history_mask.unsqueeze(1)        # (B, 1, H)
        user_to_user = torch.zeros(B, 1, 1, device=history_mask.device)

        top = torch.cat([hist_to_hist, hist_to_user], dim=2)    # (B, H, H+1)
        bottom = torch.cat([user_to_hist, user_to_user], dim=2) # (B, 1, H+1)
        return torch.cat([top, bottom], dim=1)                   # (B, H+1, H+1)

    def gnn_enhanced_user(
        self,
        news_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple:
        """GNN-enhanced hybrid user representation.

        Args:
            news_embeddings: (B, H, news_emb_dim)
            history_mask: (B, H) float mask

        Returns:
            (enhanced_user_emb, enhanced_news_embs)
        """
        # Initialise user node from masked mean of news embeddings
        mask_exp = history_mask.unsqueeze(-1)  # (B, H, 1)
        masked_news = news_embeddings * mask_exp
        count = mask_exp.sum(dim=1).clamp(min=1.0)
        user_initial = masked_news.sum(dim=1) / count  # (B, news_emb_dim)

        # Project into graph space
        user_graph = self.user_node_init(user_initial).unsqueeze(1)  # (B, 1, G)
        news_graph = self.news_graph_projection(news_embeddings)      # (B, H, G)

        # Build adjacency
        adjacency = self.create_bipartite_graph(history_mask)

        # GNN forward
        enhanced_user, enhanced_news = self.gnn_layer(
            user_graph, news_graph, adjacency
        )

        enhanced_user = enhanced_user.squeeze(1)  # (B, G)
        return enhanced_user, enhanced_news

    # ----- forward --------------------------------------------------------

    def forward(
        self, inputs: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Encode a user from their browsing history.

        Args:
            inputs: (B, H, feature_size) concatenated history features.

        Returns:
            (B, news_embedding_dim) user representations.
        """
        B, H, T = inputs.shape

        # TimeDistributed news encoder
        flat = inputs.reshape(B * H, T)
        news_vecs = self.news_encoder(flat, training=training)
        clicked_news = news_vecs.reshape(B, H, -1)  # (B, H, news_emb_dim)

        # History mask (any nonzero token in a history slot means it's valid)
        history_mask = inputs.any(dim=-1).float()  # (B, H)

        # GNN-enhanced user representation
        enhanced_user_emb, enhanced_news = self.gnn_enhanced_user(
            clicked_news, history_mask
        )

        # Attention-based user representation on GNN-enhanced news
        attn_user = self.user_attention(
            enhanced_news, mask=history_mask.bool()
        )

        # Hybrid: concat GNN user + attention user -> projection
        hybrid = torch.cat([enhanced_user_emb, attn_user], dim=-1)
        user_repr = self.final_user_projection(hybrid)

        return user_repr


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------


class CROWN(BaseModel):
    """CROWN: Category-guided intent disentanglement + GNN hybrid user representation.

    Supports three scoring modes (same interface as NRMS/LSTUR):
    - Training: softmax over K+1 candidates.
    - Single candidate: sigmoid score.
    - Multiple candidates: sigmoid scores.
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
        self.processed_news = processed_news
        self.process_user_id = config.process_user_id

        # Shared word embedding (initialised from pretrained)
        vocab_size = processed_news["vocab_size"]
        embeddings_matrix = processed_news["embeddings"]
        self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # Category / subcategory embeddings
        self.category_embedding = nn.Embedding(
            processed_news["num_categories"], config.category_embedding_dim
        )
        self.subcategory_embedding = nn.Embedding(
            processed_news["num_subcategories"], config.subcategory_embedding_dim
        )

        # Category predictor (auxiliary task)
        self.category_predictor = CategoryPredictor(
            num_categories=processed_news.get("num_categories", 18),
            input_dim=config.intent_embedding_dim,
        )

        # Build encoders
        self.news_encoder = NewsEncoder(
            config,
            self.embedding_layer,
            self.category_embedding,
            self.subcategory_embedding,
            self.category_predictor,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder)

    # ----- input concatenation helper -------------------------------------

    @staticmethod
    def _concat_inputs(
        tokens: torch.Tensor,
        abstract_tokens: torch.Tensor,
        category: torch.Tensor,
        subcategory: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate [title, abstract, category, subcategory] along the last dim.

        Handles both 2-D (single news) and 3-D (batch of news sequences) inputs.
        """
        cat = category
        subcat = subcategory

        if cat.dim() < tokens.dim():
            cat = cat.unsqueeze(-1)
        if subcat.dim() < tokens.dim():
            subcat = subcat.unsqueeze(-1)

        return torch.cat([tokens, abstract_tokens, cat, subcat], dim=-1)

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Main forward pass.

        Args:
            inputs: Dictionary with keys:
                Training / multi-candidate:
                    ``hist_tokens``, ``hist_abstract_tokens``,
                    ``hist_category``, ``hist_subcategory``,
                    ``cand_tokens``, ``cand_abstract_tokens``,
                    ``cand_category``, ``cand_subcategory``
                Single candidate:
                    ``history_tokens``, ``history_abstract_tokens``,
                    ``history_category``, ``history_subcategory``,
                    ``single_candidate_tokens``, ``single_candidate_abstract_tokens``,
                    ``single_candidate_category``, ``single_candidate_subcategory``
            training: Whether in training mode.

        Returns:
            Scores tensor.
        """
        if training:
            return self._score_training(inputs)
        elif "single_candidate_tokens" in inputs:
            return self._score_single(inputs)
        else:
            return self._score_multi(inputs)

    # ----- scoring helpers ------------------------------------------------

    def _score_training(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Softmax scores for training batch."""
        history_concat = self._concat_inputs(
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            inputs["hist_category"],
            inputs["hist_subcategory"],
        )
        candidate_concat = self._concat_inputs(
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            inputs["cand_category"],
            inputs["cand_subcategory"],
        )

        user_repr = self.user_encoder(history_concat, training=True)  # (B, D)

        # TimeDistributed over candidates
        B, C, T = candidate_concat.shape
        flat_cand = candidate_concat.reshape(B * C, T)
        cand_repr = self.news_encoder(flat_cand, training=True).reshape(B, C, -1)

        scores = (cand_repr * user_repr.unsqueeze(1)).sum(dim=-1)  # (B, C)
        return torch.softmax(scores, dim=-1)

    def _score_single(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sigmoid score for a single candidate."""
        history_concat = self._concat_inputs(
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            inputs["hist_category"],
            inputs["hist_subcategory"],
        )

        # Build single-candidate concatenated input
        cand_concat = self._concat_inputs(
            inputs["single_candidate_tokens"],
            inputs["single_candidate_abstract_tokens"],
            inputs["single_candidate_category"],
            inputs["single_candidate_subcategory"],
        )

        user_repr = self.user_encoder(history_concat, training=False)
        cand_repr = self.news_encoder(cand_concat, training=False)

        score = (cand_repr * user_repr).sum(dim=-1, keepdim=True)
        return torch.sigmoid(score)

    def _score_multi(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sigmoid scores for multiple candidates (evaluation)."""
        history_concat = self._concat_inputs(
            inputs["hist_tokens"],
            inputs["hist_abstract_tokens"],
            inputs["hist_category"],
            inputs["hist_subcategory"],
        )
        candidate_concat = self._concat_inputs(
            inputs["cand_tokens"],
            inputs["cand_abstract_tokens"],
            inputs["cand_category"],
            inputs["cand_subcategory"],
        )

        user_repr = self.user_encoder(history_concat, training=False)

        B, C, T = candidate_concat.shape
        flat_cand = candidate_concat.reshape(B * C, T)
        cand_repr = self.news_encoder(flat_cand, training=False).reshape(B, C, -1)

        scores = (cand_repr * user_repr.unsqueeze(1)).sum(dim=-1)
        return torch.sigmoid(scores)
