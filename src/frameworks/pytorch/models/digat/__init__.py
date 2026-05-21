"""DIGAT (PyTorch) — folder-style model.

Layout:
    news_encoder.py   — MSA-based per-news encoder
    graph_encoder.py  — dual interactive graph attention encoder
                        (news SAG + user-topic graph) + scatter helpers
    model.py          — top-level DIGAT, composes the two encoders
"""

from src.core.models.configs import DIGATConfig

from .graph_encoder import GraphEncoder
from .model import DIGAT
from .news_encoder import NewsEncoder

__all__ = ["DIGAT", "DIGATConfig", "GraphEncoder", "NewsEncoder"]
