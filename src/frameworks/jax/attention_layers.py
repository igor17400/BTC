"""Attention and masking layers for Flax NNX news recommendation models.

Core layers used by NRMS, NAML, LSTUR, and PP-Rec:
- :class:`AdditiveAttention` — soft-alignment attention (all models).
- :class:`AttentivePoolingQKY` — key/value-split attention (PP-Rec CPJA).
- :func:`compute_mask` / :func:`overwrite_mask` — pure JAX masking utils.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class AdditiveAttention(nnx.Module):
    """Soft-alignment-based attention layer (Flax NNX).

    Computes a weighted sum of the input sequence where the weights are
    learned during training. This is the standard additive attention mechanism
    used across NRMS, NAML, and LSTUR models.

    Args:
        input_dim: The feature dimension of the input (last axis size).
        query_vec_dim: The hidden dimension of the attention mechanism.
        rngs: NNX random number generator state.
    """

    def __init__(self, input_dim: int, query_vec_dim: int = 200, *, rngs: nnx.Rngs):
        glorot_init = nnx.initializers.glorot_uniform()
        key_w = rngs.params()
        key_q = rngs.params()

        self.W = nnx.Param(glorot_init(key_w, (input_dim, query_vec_dim)))
        self.b = nnx.Param(jnp.zeros((query_vec_dim,)))
        self.q = nnx.Param(glorot_init(key_q, (query_vec_dim, 1)))

    def __call__(self, inputs: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Forward pass.

        Args:
            inputs: 3-D tensor of shape ``(batch, time_steps, features)``.
            mask: Optional boolean mask ``(batch, time_steps)``; ``True``
                for positions to attend to.

        Returns:
            2-D tensor ``(batch, features)`` -- weighted sum of the input
            sequence.
        """
        attention_hidden = jnp.tanh(jnp.matmul(inputs, self.W.value) + self.b.value)
        attention_scores = jnp.squeeze(
            jnp.matmul(attention_hidden, self.q.value), axis=-1
        )

        if mask is None:
            attention = jnp.exp(attention_scores)
        else:
            attention = jnp.exp(attention_scores) * mask.astype(inputs.dtype)

        attention_weights = attention / (
            jnp.sum(attention, axis=-1, keepdims=True) + 1e-7
        )

        attention_weights_expanded = jnp.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights_expanded

        return jnp.sum(weighted_input, axis=1)


class AttentivePoolingQKY(nnx.Module):
    """Attentive pooling where attention keys differ from values (Flax NNX).

    Computes attention weights from ``key_input`` (e.g. concatenated
    content + popularity embeddings) but returns a weighted sum of
    ``value_input`` (e.g. content embeddings only). Used in PP-Rec's
    Content-Popularity Joint Attention (CPJA).

    Args:
        key_dim: Last dimension of ``key_input``.
        query_vec_dim: Hidden dimension of the attention MLP.
        rngs: NNX random number generator state.
    """

    def __init__(self, key_dim: int, query_vec_dim: int = 200, *, rngs: nnx.Rngs):
        glorot_init = nnx.initializers.glorot_uniform()
        key_w = rngs.params()
        key_q = rngs.params()
        self.W = nnx.Param(glorot_init(key_w, (key_dim, query_vec_dim)))
        self.b = nnx.Param(jnp.zeros((query_vec_dim,)))
        self.q = nnx.Param(glorot_init(key_q, (query_vec_dim, 1)))

    def __call__(
        self,
        key_input: jax.Array,
        value_input: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass.

        Args:
            key_input:   ``(batch, seq_len, key_dim)`` used for attention weights.
            value_input: ``(batch, seq_len, val_dim)`` weighted by attention.
            mask:        ``(batch, seq_len)`` bool, ``True`` keeps.

        Returns:
            ``(batch, val_dim)`` weighted sum.
        """
        attention_hidden = jnp.tanh(
            jnp.matmul(key_input, self.W.value) + self.b.value
        )
        attention_scores = jnp.squeeze(
            jnp.matmul(attention_hidden, self.q.value), axis=-1
        )

        if mask is None:
            attention = jnp.exp(attention_scores)
        else:
            attention = jnp.exp(attention_scores) * mask.astype(value_input.dtype)

        attention_weights = attention / (
            jnp.sum(attention, axis=-1, keepdims=True) + 1e-7
        )
        attention_weights_expanded = jnp.expand_dims(attention_weights, axis=-1)
        weighted_input = value_input * attention_weights_expanded
        return jnp.sum(weighted_input, axis=1)


# ---------------------------------------------------------------------------
# Pure JAX masking utilities
# ---------------------------------------------------------------------------


def compute_mask(inputs: jax.Array) -> jax.Array:
    """Compute a boolean mask where non-zero positions are ``True``."""
    return jnp.not_equal(inputs, 0)


def overwrite_mask(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Zero out positions indicated by a mask."""
    return values * jnp.expand_dims(mask, axis=-1)
