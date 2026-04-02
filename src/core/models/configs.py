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
    """Configuration class for CROWN model parameters."""

    # Common parameters
    embedding_size: int = 300
    dropout_rate: float = 0.2
    seed: int = 42

    # Model dimensions
    intent_embedding_dim: int = 200
    category_embedding_dim: int = 100
    subcategory_embedding_dim: int = 100
    attention_dim: int = 200

    # Intent disentanglement
    intent_num: int = 3  # Number of intents (k)
    alpha: float = 0.3  # Weight for auxiliary loss

    # MAB parameters
    num_heads: int = 12  # 300 / 12 = 25 (evenly divisible)
    head_dim: int = 25  # 12 * 25 = 300 = embedding_size
    feedforward_dim: int = 512
    num_layers: int = 2

    # GNN parameters
    gnn_type: str = "graphsage"  # 'graphsage' or 'gat'
    graph_hidden_dim: int = 300
    graph_num_layers: int = 1

    # GAT-specific parameters
    gat_num_heads: int = 4
    gat_alpha: float = 0.2
    gat_concat_heads: bool = True

    # GraphSAGE-specific parameters
    sage_aggregator: str = "mean"  # 'mean', 'max', 'sum', 'attention'
    sage_normalize: bool = True

    # Input parameters
    max_title_length: int = 50
    max_abstract_length: int = 100
    max_history_length: int = 50
    max_impressions_length: int = 5

    # Training parameters
    process_user_id: bool = False
