"""CROWN (Flax NNX) — folder-style model.

Layout:
    news_encoder.py  — transformer + positional encoding + k-intent
                       disentanglement + title-body fusion
    user_encoder.py  — bipartite user↔news GNN (GAT or GraphSAGE) +
                       paper eq. 9 user-query attention
    model.py         — top-level CROWN, composes the two encoders
"""

from src.core.models.configs import CROWNConfig

from .model import CROWN
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder

__all__ = ["CROWN", "CROWNConfig", "NewsEncoder", "UserEncoder"]
