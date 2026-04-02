"""PyTorch framework implementation for NewsReX news recommendation models."""

from .device import setup_device
from .losses import CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss, get_loss
from .layers import AdditiveAttention, ComputeMasking, OverwriteMasking, compute_mask, overwrite_mask

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
