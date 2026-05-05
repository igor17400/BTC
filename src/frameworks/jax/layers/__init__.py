"""Flax NNX custom layers for news recommendation models.

Re-exports from sub-modules so callers can use
``from ..layers import AdditiveAttention`` as before.
"""

from .attention_layers import (
    AdditiveAttention,
    AttentivePoolingQKY,
    CrossAttention,
    compute_mask,
    overwrite_mask,
)
from .layer_utils import apply_activation

__all__ = [
    "AdditiveAttention",
    "AttentivePoolingQKY",
    "CrossAttention",
    "compute_mask",
    "overwrite_mask",
    "apply_activation",
]
