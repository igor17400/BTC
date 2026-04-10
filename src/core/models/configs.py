"""Shared model configuration dataclasses for all frameworks.

Each Config dataclass defines the hyperparameters for a specific model architecture.
These are framework-agnostic and used by Keras, PyTorch, and JAX implementations.
"""

from dataclasses import dataclass


@dataclass
class NRMSConfig:
    """Configuration class for NRMS model parameters."""

    embedding_size: int = 300
    multiheads: int = 16
    head_dim: int = 16
    attention_hidden_dim: int = 200
    dropout_rate: float = 0.2
    seed: int = 42
    max_title_length: int = 50
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = False


@dataclass
class NAMLConfig:
    """Configuration class for NAML model parameters."""

    max_title_length: int = 30
    max_abstract_length: int = 50
    embedding_size: int = 300
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100
    cnn_filter_num: int = 400
    cnn_kernel_size: int = 3
    word_attention_query_dim: int = 200
    view_attention_query_dim: int = 200
    user_attention_query_dim: int = 200
    dropout_rate: float = 0.2
    activation: str = "relu"
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = False
    seed: int = 42


@dataclass
class LSTURConfig:
    """Configuration class for LSTUR model parameters."""

    embedding_size: int = 300
    cnn_filter_num: int = 300
    cnn_kernel_size: int = 3
    cnn_activation: str = "relu"
    attention_hidden_dim: int = 200
    gru_unit: int = 300
    type: str = "ini"  # "ini" or "con" for different user encoder types
    dropout_rate: float = 0.2
    # Bernoulli mask on long-term user embeddings (paper §3.2). Distinct from
    # the regular dropout_rate so we can disable it during fine-tuning.
    user_embedding_dropout_rate: float = 0.5
    seed: int = 42
    max_title_length: int = 50
    max_history_length: int = 50
    max_impressions_length: int = 5
    process_user_id: bool = True  # LSTUR uses user ids as embedding layer
    use_category: bool = False
    use_subcategory: bool = False
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100


@dataclass
class CROWNConfig:
    """Configuration for CROWN (WWW 2025).

    Defaults match the official reference implementation:
    ``reference_codes/crown-www25/config.py``.
    """

    # Word / shared embeddings
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # Category / subcategory embeddings (paper: 50 each)
    category_embedding_dim: int = 50
    subcategory_embedding_dim: int = 50

    # Transformer encoder (paper: 1 layer, 10 heads, FFN 512)
    num_heads: int = 10
    head_dim: int = 30  # 300 / 10
    feedforward_dim: int = 512
    num_layers: int = 1

    # Intent disentanglement (paper: k=3, dim=400)
    intent_num: int = 3
    intent_embedding_dim: int = 400
    attention_dim: int = 400
    alpha: float = 0.3  # auxiliary category-prediction loss weight

    # GNN for user encoder (paper Appendix A.5: GAT chosen as final model)
    gnn_type: str = "gat"
    graph_num_layers: int = 5
    no_self_connection: bool = False
    gcn_normalization_type: str = "symmetric"  # 'symmetric' or 'asymmetric'
    no_gcn_residual: bool = True
    gcn_layer_norm: bool = False

    # GraphSAGE-specific
    sage_aggregator: str = "mean"
    sage_normalize: bool = True

    # GAT-specific (alternative to GraphSAGE)
    gat_num_heads: int = 4
    gat_alpha: float = 0.2
    gat_concat_heads: bool = True

    # Candidate-aware attention (paper: scaled dot-product)
    user_attention_dim: int = 400

    # Input constraints (paper: title=32, abstract=128, history=50)
    max_title_length: int = 32
    max_abstract_length: int = 128
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Training
    process_user_id: bool = False
    gradient_clip_norm: float = 4.0


@dataclass
class PPRecConfig:
    """Configuration class for PP-Rec model parameters.

    PP-Rec: News Recommendation with Personalized User Interest and
    Time-aware News Popularity (ACL 2021).
    """

    # Embedding dimensions
    embedding_size: int = 300
    news_dim: int = 400
    entity_embedding_dim: int = 100
    category_embedding_dim: int = 200

    # Multi-head self-attention (title MHSA, entity MHSA)
    num_heads: int = 20
    head_dim: int = 20
    # Cross-attention between title words and entities (paper "co1" encoder).
    # Paper uses 5 heads x 40 dim = 200 total.
    co_num_heads: int = 5
    co_head_dim: int = 40
    attention_hidden_dim: int = 200

    # Popularity embeddings
    popularity_embedding_bins: int = 200
    popularity_embedding_dim: int = 400
    recency_embedding_bins: int = 1500
    recency_embedding_dim: int = 100
    ctr_scaler_init: float = 19.0

    # PopularityPredictor MLP widths (paper PP-Rec official code).
    # content scorer: news_dim -> dims[0] -> dims[1] -> dims[2] -> 1
    pop_content_dims: tuple[int, int, int] = (256, 256, 128)
    # recency scorer: recency_emb_dim -> dims[0] -> dims[1] -> 1
    pop_recency_dims: tuple[int, int] = (64, 64)
    # gate MLP: concat[news, recency_emb] -> dims[0] -> dims[1] -> 1 (sigmoid)
    pop_gate_dims: tuple[int, int] = (128, 64)
    # ActivityGater hidden width: news_dim -> hidden -> 1 (sigmoid)
    activity_gate_hidden_dim: int = 64

    # Feature flags
    use_entity: bool = True
    use_recency: bool = True
    use_ctr: bool = True
    use_activity_gate: bool = True

    # Training
    dropout_rate: float = 0.2
    seed: int = 42

    # Input constraints
    max_title_length: int = 32
    max_history_length: int = 50
    max_impressions_length: int = 5
    max_entities: int = 5

    process_user_id: bool = False
