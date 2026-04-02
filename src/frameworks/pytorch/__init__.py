"""PyTorch framework implementation for NewsReX news recommendation models."""

from .device import setup_device
from .losses import CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss
from .layers import AdditiveAttention, ComputeMasking, OverwriteMasking

__all__ = [
    "setup_device",
    "CategoricalCrossEntropyLoss",
    "BinaryCrossEntropyLoss",
    "AdditiveAttention",
    "ComputeMasking",
    "OverwriteMasking",
]
