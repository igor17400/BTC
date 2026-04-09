"""Keras custom layers for news recommendation models.

Re-exports from sub-modules so callers can use
``from src.frameworks.keras.layers import AdditiveAttention`` as before.
"""

from .attention_layers import (
    AdditiveAttention,
    AttentivePoolingQKY,
    ComputeMasking,
    GlorotUniformMHA,
    OverwriteMasking,
)
from .graph_layers import (
    GraphAttentionLayer,
    GraphSAGELayer,
    MultiHeadAttentionBlock,
)

__all__ = [
    "AdditiveAttention",
    "AttentivePoolingQKY",
    "ComputeMasking",
    "GlorotUniformMHA",
    "OverwriteMasking",
    "GraphAttentionLayer",
    "GraphSAGELayer",
    "MultiHeadAttentionBlock",
]
