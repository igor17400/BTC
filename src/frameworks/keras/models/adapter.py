"""Keras implementation of the framework adapter Protocol.

The shared evaluator at :mod:`src.core.models.evaluation` invokes Keras
encoder modules through this adapter and converts Keras tensors to
numpy via the backend-agnostic ``keras.ops`` API. This is the only
place in the Keras path that needs to know about the shared evaluation
pipeline; everything downstream is pure numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from keras import ops


class KerasAdapter:
    """Framework adapter for Keras news recommendation models."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert any Keras tensor (or numpy array) to numpy."""
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "numpy") or hasattr(value, "__array__"):
            return ops.convert_to_numpy(value)
        return np.asarray(value)

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run the news encoder on a feature batch and return numpy vectors."""
        return ops.convert_to_numpy(encoder(features, training=False))

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run the user encoder on a history batch and return numpy vectors.

        For LSTUR-style encoders (``process_user_id=True``) the encoder is
        called as ``encoder([features, user_ids], training=False)``. For
        all other models it's ``encoder(features, training=False)``.
        """
        if process_user_id:
            vec = encoder([features, user_ids], training=False)
        else:
            vec = encoder(features, training=False)
        return ops.convert_to_numpy(vec)

    def run_activity_gater(self, gater: Any, user_vecs: Any) -> np.ndarray:
        """Run the ActivityGater on a batch of user vectors."""
        return ops.convert_to_numpy(gater(user_vecs, training=False))

    def run_popularity_predictor(
        self, predictor: Any, bias_vecs: Any, recency: Any | None, ctr: Any | None,
    ) -> np.ndarray:
        """Run the PopularityPredictor with full inputs."""
        return ops.convert_to_numpy(
            predictor(bias_vecs, recency_indices=recency, ctr_values=ctr, training=False)
        )
