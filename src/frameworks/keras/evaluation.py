"""Keras-specific binding of the evaluation pipeline.

Exports :func:`get_evaluator` which resolves the correct evaluator from
the spec YAML and pre-binds the :class:`KerasAdapter`.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.models.evaluations import get_evaluator as _get_evaluator
from src.frameworks.keras.models.adapter import KerasAdapter


def get_evaluator(spec) -> Callable:
    """Resolve the evaluator declared in the spec and bind the Keras adapter.

    Args:
        spec: The Hydra spec config (``cfg.spec``). The evaluator name is
            read from ``spec.evaluation.evaluator``, defaulting to ``"default"``.

    Returns:
        A callable evaluator with the Keras adapter pre-bound.
    """
    name = spec.evaluation.get("evaluator", "default")
    return _get_evaluator(name, KerasAdapter())


__all__ = ["get_evaluator"]
