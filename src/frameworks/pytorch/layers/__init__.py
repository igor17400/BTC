"""PyTorch custom layers for news recommendation models.

Re-exports from sub-modules so callers can use
``from ..layers import AdditiveAttention`` as before.
"""

from .attention_layers import (
    AdditiveAttention,
    AttentivePoolingQKY,
    ComputeMasking,
    OverwriteMasking,
    compute_mask,
    overwrite_mask,
)
from .layer_utils import get_activation

__all__ = [
    "AdditiveAttention",
    "AttentivePoolingQKY",
    "ComputeMasking",
    "OverwriteMasking",
    "compute_mask",
    "overwrite_mask",
    "get_activation",
]
