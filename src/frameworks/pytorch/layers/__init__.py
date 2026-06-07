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
from .plm_pooler import Pooler
from .plm_token import PLMTokenCNNEncoder, PLMTokenLookup, PLMTokenNewsEncoder
from .text_encoder import GloveTextEncoder, PLMTextEncoder, TextEncoder

__all__ = [
    "AdditiveAttention",
    "AttentivePoolingQKY",
    "ComputeMasking",
    "CrossAttention",
    "GloveTextEncoder",
    "MultiHeadSelfAttention",
    "OverwriteMasking",
    "PLMTextEncoder",
    "PLMTokenCNNEncoder",
    "PLMTokenLookup",
    "PLMTokenNewsEncoder",
    "Pooler",
    "TextEncoder",
    "compute_mask",
    "overwrite_mask",
    "get_activation",
]
