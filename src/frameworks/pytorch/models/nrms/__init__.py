"""NRMS (PyTorch) — folder-style model.

Layout:
    news_encoder.py  — per-news MHSA + additive pool over TextEncoder output
    user_encoder.py  — history MHSA + additive pool
    model.py         — top-level NRMS, composes the two encoders
"""

from src.core.models.configs import NRMSConfig

from .model import NRMS
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder

__all__ = ["NRMS", "NRMSConfig", "NewsEncoder", "UserEncoder"]
