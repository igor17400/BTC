"""NAML (PyTorch) — folder-style model.

Layout:
    news_encoder.py  — 4-view per-news encoder + view attention
    user_encoder.py  — additive-attention history pool
    model.py         — top-level NAML, composes the two encoders
"""

from src.core.models.configs import NAMLConfig

from .model import NAML
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder

__all__ = ["NAML", "NAMLConfig", "NewsEncoder", "UserEncoder"]
