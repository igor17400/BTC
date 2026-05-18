"""PyTorch custom layers for news recommendation models.

Re-exports from sub-modules so callers can use
``from ..layers import AdditiveAttention`` as before.
"""

from .attention_layers import (
    AdditiveAttention,
    AttentivePoolingQKY,
    ComputeMasking,
    CrossAttention,
    MultiHeadSelfAttention,
    OverwriteMasking,
    compute_mask,
    overwrite_mask,
)
from .layer_utils import get_activation
from .plm import PLMNewsEncoder
from .plm_pooler import Pooler
from .plm_token import PLMTokenNewsEncoder

__all__ = [
    "AdditiveAttention",
    "AttentivePoolingQKY",
    "ComputeMasking",
    "CrossAttention",
    "MultiHeadSelfAttention",
    "OverwriteMasking",
    "PLMNewsEncoder",
    "PLMTokenNewsEncoder",
    "Pooler",
    "compute_mask",
    "overwrite_mask",
    "get_activation",
]
