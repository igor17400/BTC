"""Model evaluation pipelines.

- :mod:`.default` — shared fast evaluation (dot-product scoring).
- :mod:`.utils` — shared precomputation and metric helpers.
- :mod:`.custom` — model-specific evaluators (DIGAT, GLORY, MINER, PP-Rec).

Use :func:`get_evaluator` to resolve a registry-based evaluator by name.
The spec YAML declares which evaluator a model uses via
``evaluation.evaluator`` (defaults to ``"default"``).

Note: DIGAT and GLORY use their evaluators directly rather than the
registry, because their calling conventions differ (they take raw
encoder modules and dataset-level arguments instead of dataloaders).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from .custom import (
    digat_evaluate,
    glory_evaluate,
    miner_fast_evaluate,
    pprec_fast_evaluate,
)
from .default import fast_evaluate

if TYPE_CHECKING:
    from src.core.models.adapter import FrameworkAdapter

EVALUATOR_REGISTRY: dict[str, Callable] = {
    "default": fast_evaluate,
    "pp_rec": pprec_fast_evaluate,
    "miner": miner_fast_evaluate,
}


def get_evaluator(evaluator_name: str, adapter: FrameworkAdapter) -> Callable:
    """Resolve an evaluator by name and pre-bind the framework adapter.

    Args:
        evaluator_name: Key in :data:`EVALUATOR_REGISTRY` (e.g. ``"default"``,
            ``"pp_rec"``, ``"miner"``).
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
    "glory_evaluate",
    "miner_fast_evaluate",
    "pprec_fast_evaluate",
]
