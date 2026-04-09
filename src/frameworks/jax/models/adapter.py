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

    def run_activity_gater(
        self, gater: Any, user_vecs: Any
    ) -> np.ndarray:
        """Run the ActivityGater on a batch of user vectors.

        Args:
            gater: Framework-native ActivityGater module.
            user_vecs: ``(B, news_dim)`` user vectors.

        Returns:
            ``(B,)`` numpy array of gate values (eta) in [0, 1].
        """
        user_jnp = jnp.asarray(user_vecs)
        return np.asarray(gater(user_jnp))

    def run_popularity_predictor(
        self,
        predictor: Any,
        bias_vecs: Any,
        recency: Any | None,
        ctr: Any | None,
    ) -> np.ndarray:
        """Run the PopularityPredictor with full inputs (bias + recency + CTR).

        Args:
            predictor: Framework-native PopularityPredictor module.
            bias_vecs: ``(C, news_dim)`` bias news vectors.
            recency: Optional ``(C,)`` int recency bucket indices.
            ctr: Optional ``(C,)`` float CTR values.

        Returns:
            ``(C,)`` numpy array of popularity scores.
        """
        bias_jnp = jnp.asarray(bias_vecs)
        recency_jnp = jnp.asarray(recency).astype(jnp.int32) if recency is not None else None
        ctr_jnp = jnp.asarray(ctr).astype(jnp.float32) if ctr is not None else None
        pop_scores = predictor(
            bias_jnp, recency_indices=recency_jnp, ctr_values=ctr_jnp
        )
        return np.asarray(pop_scores)
