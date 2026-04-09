"""PyTorch implementation of the framework adapter Protocol.

The shared evaluator at :mod:`src.core.models.evaluation` invokes PyTorch
encoder modules through this adapter and converts PyTorch tensors to
numpy. This is the only place in the PyTorch path that needs to know
about the shared evaluation pipeline; everything downstream is pure
numpy.

Device placement, ``model.eval()``, and ``torch.no_grad()`` are all
handled inside the adapter so the shared evaluator stays
framework-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class PyTorchAdapter:
    """Framework adapter for PyTorch news recommendation models."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert any PyTorch tensor (or numpy array) to numpy."""
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def encode_news(self, encoder: Any, features: Any) -> np.ndarray:
        """Run the news encoder on a feature batch and return numpy vectors.

        Handles device placement, eval-mode, and no-grad internally.
        """
        encoder.eval()
        device = next(encoder.parameters()).device
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features)
        features = features.to(device)
        with torch.no_grad():
            vec = encoder(features, training=False)
        return vec.detach().cpu().numpy()

    def encode_user(
        self,
        encoder: Any,
        features: Any,
        user_ids: Any | None,
        process_user_id: bool,
    ) -> np.ndarray:
        """Run the user encoder on a history batch and return numpy vectors.

        For LSTUR-style encoders (``process_user_id=True``) the encoder is
        called as ``encoder([features, user_ids], training=False)``. For all
        other models it's ``encoder(features, training=False)``.
        """
        encoder.eval()
        device = next(encoder.parameters()).device
        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features)
        features = features.to(device)
        with torch.no_grad():
            if process_user_id:
                if isinstance(user_ids, np.ndarray):
                    user_ids = torch.as_tensor(user_ids)
                user_ids = user_ids.to(device)
                vec = encoder([features, user_ids], training=False)
            else:
                vec = encoder(features, training=False)
        return vec.detach().cpu().numpy()

    def run_activity_gater(self, gater: Any, user_vecs: Any) -> np.ndarray:
        """Run the ActivityGater on a batch of user vectors."""
        gater.eval()
        device = next(gater.parameters()).device
        user_t = torch.as_tensor(user_vecs, dtype=torch.float32).to(device)
        with torch.no_grad():
            return gater(user_t).detach().cpu().numpy()

    def run_popularity_predictor(
        self,
        predictor: Any,
        bias_vecs: Any,
        recency: Any | None,
        ctr: Any | None,
    ) -> np.ndarray:
        """Run the PopularityPredictor with full inputs."""
        predictor.eval()
        device = next(predictor.parameters()).device
        bias_t = torch.as_tensor(bias_vecs, dtype=torch.float32).to(device)
        recency_t = (
            torch.as_tensor(recency).long().to(device) if recency is not None else None
        )
        ctr_t = (
            torch.as_tensor(ctr, dtype=torch.float32).to(device)
            if ctr is not None
            else None
        )
        with torch.no_grad():
            return (
                predictor(bias_t, recency_indices=recency_t, ctr_values=ctr_t)
                .detach()
                .cpu()
                .numpy()
            )
