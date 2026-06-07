"""TCCM (Flax NNX) — folder-style model.

Layout:
    popularity.py — TCCM-specific popularity encoder + activity gater
    model.py      — top-level TCCM, composes PP-Rec encoders +
                    TCCM popularity branch

Relevance branch (news / user encoder) is reused from PP-Rec; see
:mod:`src.frameworks.jax.models.pprec`.
"""

from .model import TCCM
from .popularity import TCCMActivityGater, TCCMPopularityEncoder

__all__ = [
    "TCCM",
    "TCCMActivityGater",
    "TCCMPopularityEncoder",
]
