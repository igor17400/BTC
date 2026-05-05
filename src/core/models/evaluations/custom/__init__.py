"""Model-specific evaluation pipelines.

Custom evaluators for models that require scoring logic beyond the
standard dot-product pipeline (e.g. graph interaction, multi-interest
matching, popularity-aware scoring).

- :mod:`.caum` — CAUM candidate-aware scoring.
- :mod:`.digat` — DIGAT dual graph interaction scoring.
- :mod:`.glory` — GLORY global graph + entity scoring.
- :mod:`.miner` — MINER multi-interest matching.
- :mod:`.pp_rec` — PP-Rec popularity-aware scoring.
"""

from .caum import caum_fast_evaluate
from .digat import digat_evaluate
from .glory import glory_evaluate
from .miner import miner_fast_evaluate
from .pp_rec import pprec_fast_evaluate

__all__ = [
    "caum_fast_evaluate",
    "digat_evaluate",
    "glory_evaluate",
    "miner_fast_evaluate",
    "pprec_fast_evaluate",
]
