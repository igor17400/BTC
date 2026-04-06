"""Loss functions for PyTorch news recommendation models."""

from typing import Any

import torch
import torch.nn as nn


class CategoricalCrossEntropyLoss(nn.Module):
    """Categorical cross-entropy loss over candidate news articles.

    Expects model output to be raw logits of shape (batch, num_candidates).
    The target is a one-hot label vector of the same shape indicating the
    positive candidate. Uses nn.CrossEntropyLoss which applies log-softmax
    internally via the numerically stable log-sum-exp trick.

    Note: ``from_logits`` is accepted for interface consistency with Keras/JAX
    but is always True — PyTorch CrossEntropyLoss always expects logits.
    """

    def __init__(self, label_smoothing: float = 0.0, from_logits: bool = True, **kwargs: Any):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw scores (batch, num_candidates).
            targets: One-hot labels (batch, num_candidates) **or**
                     class indices (batch,).

        Returns:
            Scalar loss.
        """
        if targets.dim() == logits.dim() and targets.shape == logits.shape:
            targets = targets.argmax(dim=-1)
        return self.ce(logits, targets)


class BinaryCrossEntropyLoss(nn.Module):
    """Binary cross-entropy loss with logits.

    Wraps nn.BCEWithLogitsLoss for convenience.  Accepts raw logits.
    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw scores (any shape).
            targets: Float targets of the same shape.

        Returns:
            Scalar loss.
        """
        return self.bce(logits, targets.float())


def get_loss(loss_name: str, **kwargs: Any) -> nn.Module:
    """Get a loss function by name.

    Args:
        loss_name: Name of the loss function.
        **kwargs: Additional arguments (label_smoothing, etc.).

    Returns:
        Loss function instance.

    Raises:
        ValueError: If the loss name is not recognized.
    """
    losses = {
        "categorical_crossentropy": CategoricalCrossEntropyLoss,
        "binary_crossentropy": BinaryCrossEntropyLoss,
    }

    if loss_name not in losses:
        raise ValueError(
            f"Unknown loss: {loss_name}. Available losses: {list(losses.keys())}"
        )

    return losses[loss_name](**kwargs)
