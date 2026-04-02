"""Loss functions for PyTorch news recommendation models."""

import torch
import torch.nn as nn


class CategoricalCrossEntropyLoss(nn.Module):
    """Categorical cross-entropy loss over candidate news articles.

    Expects model output to be raw logits of shape (batch, num_candidates).
    The target is a one-hot label vector of the same shape indicating the
    positive candidate.  We convert to class indices and delegate to
    nn.CrossEntropyLoss which applies log-softmax internally.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

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
            # One-hot encoded targets -> convert to class indices
            targets = targets.argmax(dim=-1)
        return self.ce(logits, targets)


class BinaryCrossEntropyLoss(nn.Module):
    """Binary cross-entropy loss with logits.

    Wraps nn.BCEWithLogitsLoss for convenience.  Accepts raw logits.
    """

    def __init__(self):
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
