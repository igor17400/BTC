"""PyTorch framework implementation for NewsReX news recommendation models."""

from .device import setup_device
from .layers import (
    AdditiveAttention,
    ComputeMasking,
    OverwriteMasking,
    compute_mask,
    overwrite_mask,
)
from .losses import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss, get_loss

__all__ = [
    "setup_device",
    "CategoricalCrossEntropyLoss",
    "BinaryCrossEntropyLoss",
    "AdditiveAttention",
    "ComputeMasking",
    "OverwriteMasking",
    "compute_mask",
    "overwrite_mask",
    "get_loss",
]
