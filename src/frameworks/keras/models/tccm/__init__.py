"""TCCM (Time and Content-Aware Causal Model) Keras implementation.

CIKM 2023. Reuses PP-Rec's news/user encoders for the user-content
matching score and replaces the bias news encoder with a content-aware
popularity encoder driven by per-token bucketed CTR plus a
reciprocal-power timeliness module.
"""

from .layers import TCCMActivityGater, TCCMNewsEncoder, TCCMPopularityEncoder
from .model import TCCM

__all__ = [
    "TCCM",
    "TCCMActivityGater",
    "TCCMNewsEncoder",
    "TCCMPopularityEncoder",
]
