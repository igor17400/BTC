"""LSTUR (PyTorch) — folder-style model.

Layout:
    news_encoder.py  — CNN + additive pool over TextEncoder output,
                       optional concat with category / subcategory embeddings
    user_encoder.py  — GRU over history news vectors + long-term user embedding
                       (``ini`` initial-state or ``con`` concat variant)
    model.py         — top-level LSTUR, composes the two encoders
"""

from src.core.models.configs import LSTURConfig

from .model import LSTUR
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder

__all__ = ["LSTUR", "LSTURConfig", "NewsEncoder", "UserEncoder"]
