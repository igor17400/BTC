"""Model evaluation pipelines.

- :mod:`.default` — shared fast evaluation (dot-product scoring).
- :mod:`.pp_rec` — PP-Rec evaluation with full popularity-aware scoring.
- :mod:`.digat` — DIGAT evaluation with dual graph interaction scoring.
- :mod:`.utils` — shared precomputation and metric helpers.

Use :func:`get_evaluator` to resolve a registry-based evaluator by name.
The spec YAML declares which evaluator a model uses via
``evaluation.evaluator`` (defaults to ``"default"``).

Note: DIGAT uses :func:`.digat.digat_evaluate` directly rather than the
registry, because its calling convention differs (it takes raw
``news_encoder``, ``graph_encoder``, and dataset-level arguments instead
of dataloaders).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from .default import fast_evaluate
from .digat import digat_evaluate
from .pp_rec import pprec_fast_evaluate

if TYPE_CHECKING:
    from src.core.models.adapter import FrameworkAdapter

EVALUATOR_REGISTRY: dict[str, Callable] = {
    "default": fast_evaluate,
    "pp_rec": pprec_fast_evaluate,
}


def get_evaluator(evaluator_name: str, adapter: FrameworkAdapter) -> Callable:
    """Resolve an evaluator by name and pre-bind the framework adapter.

    Args:
        evaluator_name: Key in :data:`EVALUATOR_REGISTRY` (e.g. ``"default"``,
            ``"pp_rec"``).
        adapter: Framework adapter to pre-bind.

    Returns:
        A callable with the same signature as :func:`fast_evaluate` /
        :func:`pprec_fast_evaluate`, but with ``adapter`` already filled in.
    """
    if evaluator_name not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator '{evaluator_name}'. "
            f"Available: {sorted(EVALUATOR_REGISTRY)}"
        )
    return partial(EVALUATOR_REGISTRY[evaluator_name], adapter=adapter)


__all__ = [
    "EVALUATOR_REGISTRY",
    "digat_evaluate",
    "fast_evaluate",
    "get_evaluator",
    "pprec_fast_evaluate",
]
