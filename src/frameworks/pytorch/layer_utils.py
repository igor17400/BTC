"""Generic layer utilities for PyTorch models.

Framework-specific helpers that don't belong to any particular layer
category (attention, graph, etc.).
"""

import torch.nn.functional as F


def get_activation(name: str):
    """Resolve a string activation name to a ``torch.nn.functional`` callable.

    Args:
        name: Activation name (e.g. ``"relu"``, ``"gelu"``, ``"tanh"``).

    Returns:
        The corresponding function from ``torch.nn.functional``.

    Raises:
        ValueError: If the name doesn't match any function in ``F``.
    """
    fn = getattr(F, name, None)
    if fn is None:
        raise ValueError(f"Unknown activation '{name}' for torch.nn.functional")
    return fn
