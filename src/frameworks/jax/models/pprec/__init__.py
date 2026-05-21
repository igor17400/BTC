"""PP-Rec (Flax NNX) — folder-style model.

Layout:
    news_encoder.py  — title MHSA + bidirectional MHCA + entity/category fusion
    user_encoder.py  — MHSA + Content-Popularity Joint Attention (CPJA)
    popularity.py    — content + recency + CTR popularity predictor + activity gate
    model.py         — top-level PPRec, composes the components
"""

from src.core.models.configs import PPRecConfig

from .model import PPRec
from .news_encoder import NewsEncoder
from .popularity import ActivityGater, PopularityPredictor
from .user_encoder import UserEncoder

# Backward-compatibility aliases — TCCM and any external code keeps
# importing PPRecNewsEncoder / PPRecUserEncoder.
PPRecNewsEncoder = NewsEncoder
PPRecUserEncoder = UserEncoder

__all__ = [
    "ActivityGater",
    "NewsEncoder",
    "PPRec",
    "PPRecConfig",
    "PPRecNewsEncoder",
    "PPRecUserEncoder",
    "PopularityPredictor",
    "UserEncoder",
]
