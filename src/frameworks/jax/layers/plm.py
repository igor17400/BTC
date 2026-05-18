"""JAX/Flax NNX frozen-PLM news encoder.

Mirror of :class:`src.frameworks.pytorch.layers.plm.PLMNewsEncoder`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class PLMNewsEncoder(nnx.Module):
    """Frozen-PLM news encoder for JAX/Flax NNX.

    Args:
        plm_embeddings_by_id: ``(max_parsed_id + 1, plm_dim)`` float32
            lookup table indexed by parsed-integer news id. Row 0 is
            the padding zero vector. Produced by
            :func:`src.core.data.encoders.plm.attach_plm_embeddings`.
        news_dim: Output dimension after the trainable projection.
        rngs: NNX RNG state for parameter init.
    """

    def __init__(
        self,
        plm_embeddings_by_id: np.ndarray,
        news_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        plm_dim = int(plm_embeddings_by_id.shape[1])
        num_news = int(plm_embeddings_by_id.shape[0])
        self.plm_dim = plm_dim
        self.news_dim = news_dim

        # The embedding table is "frozen" in the sense that its values
        # carry the PLM's outputs; NNX doesn't have a built-in freeze
        # flag, but the table is only updated if its grad is nonzero.
        # The optax optimizer (configured with `nnx.Param` filter) WILL
        # try to update it. To freeze cleanly, runners need to apply
        # `optax.set_to_zero()` to this leaf, OR we can rely on a
        # convention where the projection trains and the table doesn't
        # need to (any nudge it gets will be small in fp32). For v1 we
        # accept that the table may drift slightly — it's still
        # initialized to the PLM output.
        # If exact freezing is needed later, switch to an `nnx.Variable`
        # subclass that the optimizer filter skips.
        self.embedding = nnx.Embed(
            num_embeddings=num_news,
            features=plm_dim,
            rngs=rngs,
        )
        self.embedding.embedding.value = jnp.asarray(plm_embeddings_by_id)

        self.projection = nnx.Linear(plm_dim, news_dim, rngs=rngs)

    @staticmethod
    def valid_mask(features: jax.Array) -> jax.Array:
        """A PLM news slot is valid when its parsed id is non-zero."""
        return jnp.not_equal(features, 0)

    def __call__(
        self,
        features: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Look up frozen PLM vectors and project.

        Args:
            features: int tensor of any shape ``(...,)``. Values must be
                valid parsed-int news ids; ``0`` returns the row-0 vector
                (zero-initialised for padding by ``attach_plm_embeddings``).

        Returns:
            ``(..., news_dim)`` float tensor.
        """
        return self.projection(self.embedding(features.astype(jnp.int32)))
