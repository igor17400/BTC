"""TCCM (Time and Content-Aware Causal Model) — PyTorch port.

Same architecture as the Keras TCCM: PP-Rec ``co1`` news/user encoders
for the user-content matching score, plus a content-aware popularity
encoder driven by per-token bucketed CTR and a reciprocal-power
timeliness module.
"""

from .layers import TCCMActivityGater, TCCMNewsEncoder, TCCMPopularityEncoder
from .model import TCCM

__all__ = [
    "TCCM",
    "TCCMActivityGater",
    "TCCMNewsEncoder",
    "TCCMPopularityEncoder",
]
