"""JAX/Flax NNX implementation of the framework adapter Protocol.

The shared evaluator at :mod:`src.core.models.evaluation` invokes JAX
encoder modules through this adapter and converts JAX arrays to numpy.
This is the only place in the JAX path that needs to know about the
shared evaluation pipeline; everything downstream is pure numpy.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


class JAXAdapter:
    """Framework adapter for Flax NNX news recommendation models."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a JAX array (or anything array-like) to numpy."""
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run the news encoder on a feature batch and return numpy vectors."""
        if not isinstance(features, jnp.ndarray):
            features = jnp.asarray(features)
        return np.asarray(encoder(features, training=False))

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run the user encoder on a history batch and return numpy vectors.

        For LSTUR-style encoders (``process_user_id=True``) the encoder is
        called as ``encoder(features, user_ids, training=False)``. For all
        other models it's ``encoder(features, training=False)``.
        """
        if not isinstance(features, jnp.ndarray):
            features = jnp.asarray(features)
        if process_user_id:
            if not isinstance(user_ids, jnp.ndarray):
                user_ids = jnp.asarray(user_ids)
            vec = encoder(features, user_ids, training=False)
        else:
            vec = encoder(features, training=False)
        return np.asarray(vec)
