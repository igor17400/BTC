"""Backward-compatible re-exports from the new evaluations package.

The evaluation logic now lives in :mod:`src.core.models.evaluations`:

- :mod:`~.evaluations.default` — shared dot-product evaluation.
- :mod:`~.evaluations.pp_rec` — PP-Rec popularity-aware evaluation.
"""

from src.core.models.evaluations.default import fast_evaluate
from src.core.models.evaluations.pp_rec import pprec_fast_evaluate

__all__ = ["fast_evaluate", "pprec_fast_evaluate"]
