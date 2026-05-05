"""Unified loss function registry for all frameworks.

Dispatches to framework-specific loss implementations based on the loss name
and target framework. This is the single entry point that runners should use
to obtain loss functions from the config.

Usage::

    from src.core.losses import get_loss

    # PyTorch — returns an nn.Module
    loss_fn = get_loss("categorical_crossentropy", framework="pytorch",
                       from_logits=True, label_smoothing=0.0)

    # Keras — returns a keras.losses.Loss
    loss_fn = get_loss("categorical_crossentropy", framework="keras",
                       from_logits=True, label_smoothing=0.0)

    # JAX — returns a callable (y_true, y_pred) -> scalar
    loss_fn = get_loss("categorical_crossentropy", framework="jax",
                       from_logits=True, label_smoothing=0.0)
"""

from typing import Any


def get_loss(loss_name: str, framework: str, **kwargs: Any):
    """Get a loss function by name and framework.

    Args:
        loss_name: Name of the loss function (e.g. ``"categorical_crossentropy"``).
        framework: Target framework (``"pytorch"``, ``"keras"``, or ``"jax"``).
        **kwargs: Framework-specific keyword arguments forwarded to the
            underlying loss constructor. Common keys:

            - ``from_logits`` (bool): Whether predictions are raw logits.
            - ``label_smoothing`` (float): Label smoothing factor.
            - ``reduction`` (str): Reduction mode (Keras only).

    Returns:
        A loss function appropriate for *framework*.

    Raises:
        ValueError: If *framework* is unknown.
    """
    framework = framework.lower()

    if framework == "pytorch":
        from src.frameworks.pytorch.losses import get_loss as _get_loss
    elif framework == "keras":
        from src.frameworks.keras.losses import get_loss as _get_loss
    elif framework == "jax":
        from src.frameworks.jax.losses import get_loss as _get_loss
    else:
        raise ValueError(
            f"Unknown framework '{framework}'. Supported: 'pytorch', 'keras', 'jax'"
        )

    return _get_loss(loss_name, **kwargs)
