"""JAX-specific binding of the evaluation pipeline.

Exports :func:`get_evaluator` which resolves the correct evaluator from
the spec YAML and pre-binds the :class:`JAXAdapter`.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.models.evaluations import get_evaluator as _get_evaluator
from src.frameworks.jax.models.adapter import JAXAdapter


def get_evaluator(spec) -> Callable:
    """Resolve the evaluator declared in the spec and bind the JAX adapter.

    Args:
        spec: The Hydra spec config (``cfg.spec``). The evaluator name is
            read from ``spec.evaluation.evaluator``, defaulting to ``"default"``.

    Returns:
        A callable evaluator with the JAX adapter pre-bound.
    """
    name = spec.evaluation.get("evaluator", "default")
    return _get_evaluator(name, JAXAdapter())


__all__ = ["get_evaluator"]
