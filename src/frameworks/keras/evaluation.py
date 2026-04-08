"""Keras-specific binding of the shared fast evaluation pipeline.

The actual evaluation algorithm lives in :mod:`src.core.models.evaluation`
and is shared by Keras, PyTorch, and JAX. This module simply pre-binds
the :class:`KerasAdapter` so the runner can call ``fast_evaluate(...)``
without thinking about adapters.
"""

from __future__ import annotations

from functools import partial

from src.core.models.evaluation import fast_evaluate as _shared_fast_evaluate
from src.frameworks.keras.models.adapter import KerasAdapter

#: Pre-bound :func:`src.core.models.evaluation.fast_evaluate` with the
#: Keras adapter. The runner imports this name and calls it like the
#: JAX/PyTorch runners; the adapter parameter is invisible to callers.
fast_evaluate = partial(_shared_fast_evaluate, adapter=KerasAdapter())

__all__ = ["fast_evaluate"]
