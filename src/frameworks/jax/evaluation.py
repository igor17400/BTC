"""JAX-specific binding of the shared fast evaluation pipeline.

The actual evaluation algorithm lives in :mod:`src.core.models.evaluation`
and is shared by Keras, PyTorch, and JAX. This module simply pre-binds
the :class:`JAXAdapter` so the runner can call ``fast_evaluate(...)``
without thinking about adapters.
"""

from __future__ import annotations

from functools import partial

from src.core.models.evaluation import fast_evaluate as _shared_fast_evaluate
from src.frameworks.jax.models.adapter import JAXAdapter

#: Pre-bound :func:`src.core.models.evaluation.fast_evaluate` with the
#: JAX adapter. The runner imports this name and calls it like before;
#: the adapter parameter is invisible to callers.
fast_evaluate = partial(_shared_fast_evaluate, adapter=JAXAdapter())

__all__ = ["fast_evaluate"]
