"""Pure JAX loss functions for news recommendation training."""

import jax.numpy as jnp


def categorical_cross_entropy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    label_smoothing: float = 0.0,
    epsilon: float = 1e-7,
) -> jnp.ndarray:
    """Categorical cross-entropy loss (pure JAX).

    Used during training with softmax-activated predictions over multiple
    candidates.

    Args:
        y_true: One-hot encoded ground truth ``(batch, num_classes)``.
        y_pred: Predicted probabilities after softmax ``(batch, num_classes)``.
        label_smoothing: Smoothing factor in ``[0, 1]``.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar mean loss over the batch.
    """
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)

    if label_smoothing > 0.0:
        num_classes = y_true.shape[-1]
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    loss = -jnp.sum(y_true * jnp.log(y_pred), axis=-1)
    return jnp.mean(loss)


def binary_cross_entropy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    label_smoothing: float = 0.0,
    epsilon: float = 1e-7,
) -> jnp.ndarray:
    """Binary cross-entropy loss (pure JAX).

    Used during validation/testing with sigmoid-activated predictions for
    single-candidate scoring.

    Args:
        y_true: Ground truth labels ``0`` or ``1``.
        y_pred: Predicted probabilities after sigmoid.
        label_smoothing: Smoothing factor in ``[0, 1]``.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar mean loss.
    """
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)

    if label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = -(y_true * jnp.log(y_pred) + (1.0 - y_true) * jnp.log(1.0 - y_pred))
    return jnp.mean(loss)
