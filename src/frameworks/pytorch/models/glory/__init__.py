"""GLORY (PyTorch) — folder-style model.

Layout:
    news_encoder.py   — local text encoder + GLORY attention primitives
                        (ScaledDotProductAttention, MultiHeadAttention,
                        AttentionPooling)
    user_encoder.py   — UserEncoder (history MHA + pool) + ClickEncoder
                        (per-clicked-news view fusion)
    global_encoder.py — GatedGraphConv (pure-PyTorch GGNN) + EntityEncoder
                        and GlobalEntityEncoder (used when use_entity=True)
    model.py          — top-level GLORY + CandidateEncoder
"""

from src.core.models.configs import GLORYConfig

from .global_encoder import EntityEncoder, GatedGraphConv, GlobalEntityEncoder
from .model import GLORY, CandidateEncoder
from .news_encoder import NewsEncoder
from .user_encoder import ClickEncoder, UserEncoder

__all__ = [
    "CandidateEncoder",
    "ClickEncoder",
    "EntityEncoder",
    "GatedGraphConv",
    "GLORY",
    "GLORYConfig",
    "GlobalEntityEncoder",
    "NewsEncoder",
    "UserEncoder",
]
