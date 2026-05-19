"""CAUM (PyTorch) — folder-style model.

Layout:
    news_encoder.py  — title MHSA + AddAttn, optional entity / category branches
    user_encoder.py  — fallback mean-pool over history (BaseModel contract)
    inter_model.py   — candidate-aware scoring
                       (Candi-CNN + Candi-SelfAtt + Candi-Att)
    model.py         — top-level CAUM, composes the encoders + inter_model
"""

from src.core.models.configs import CAUMConfig

from .inter_model import DenseAttentionScorer, InterModel
from .model import CAUM
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder

__all__ = [
    "CAUM",
    "CAUMConfig",
    "DenseAttentionScorer",
    "InterModel",
    "NewsEncoder",
    "UserEncoder",
]
