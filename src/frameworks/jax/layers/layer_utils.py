"""Generic layer utilities for JAX/Flax NNX models.

Framework-specific helpers that don't belong to any particular layer
category (attention, graph, etc.).
"""

import jax
import jax.numpy as jnp


def apply_activation(x: jax.Array, activation: str) -> jax.Array:
    """Apply an activation function by name.

    Args:
        x: Input tensor.
        activation: One of ``"relu"``, ``"tanh"``, ``"gelu"``.

    Returns:
        Activated tensor with the same shape.
    """
    if activation == "relu":
        return jax.nn.relu(x)
    elif activation == "tanh":
        return jnp.tanh(x)
    elif activation == "gelu":
        return jax.nn.gelu(x)
    return x
