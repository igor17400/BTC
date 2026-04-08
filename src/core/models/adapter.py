"""Framework adapter Protocol for the shared evaluation pipeline.

The shared :mod:`src.core.models.evaluation` module needs to invoke
framework-specific encoder modules and convert framework-native tensors
to numpy. To stay framework-agnostic it never imports torch / keras /
jax directly. Instead, each framework provides a small adapter that
implements the :class:`FrameworkAdapter` protocol below, and the shared
evaluator only depends on this protocol.

This is the **only** seam between the shared evaluation algorithm and
the per-framework implementations. Adding a new framework means writing
a new ~30-line adapter and nothing more.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class FrameworkAdapter(Protocol):
    """Glue between the shared evaluator and a specific ML framework.

    Implementations live in ``src.frameworks.{framework}.models.adapter``.
    The adapter only handles operations that fundamentally require
    framework knowledge: invoking framework-native encoder modules and
    converting framework-native tensors to numpy. Everything else
    (impression iteration, dot-product scoring, NaN handling, loss,
    metric aggregation) is pure numpy in the shared evaluator.
    """

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert any framework-native tensor (or numpy array) to numpy.

        Must be a no-op when ``value`` is already a numpy array.
        """
        ...

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run a news encoder on a feature batch and return numpy vectors.

        Args:
            encoder: Framework-native module that maps news features to
                vectors. The shared evaluator never inspects this object;
                it just passes it through.
            features: Batched news features as produced by the
                framework's ``NewsBatchDataloader``.

        Returns:
            ``(batch_size, news_dim)`` numpy array of news vectors.
        """
        ...

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run a user encoder on a history batch and return numpy vectors.

        Args:
            encoder: Framework-native user encoder module.
            features: Batched history features.
            user_ids: Optional user-id tensor (only used by LSTUR).
            process_user_id: ``True`` if the encoder requires explicit
                user IDs alongside history (LSTUR pattern).

        Returns:
            ``(batch_size, user_dim)`` numpy array of user vectors.
        """
        ...
