"""PP-Rec popularity predictor + activity gater (Flax NNX).

Mirror of the PyTorch :mod:`popularity`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import PPRecConfig


class PopularityPredictor(nnx.Module):
    """Time-aware news popularity predictor."""

    def __init__(self, config: PPRecConfig, *, rngs: nnx.Rngs):
        self.config = config

        c1, c2, c3 = config.pop_content_dims
        self.content_dense1 = nnx.Linear(config.news_dim, c1, rngs=rngs)
        self.content_dense2 = nnx.Linear(c1, c2, rngs=rngs)
        self.content_dense3 = nnx.Linear(c2, c3, rngs=rngs)
        self.content_out = nnx.Linear(c3, 1, use_bias=False, rngs=rngs)

        if config.use_recency:
            r1, r2 = config.pop_recency_dims
            g1, g2 = config.pop_gate_dims
            self.recency_embedding = nnx.Embed(
                num_embeddings=config.recency_embedding_bins,
                features=config.recency_embedding_dim,
                rngs=rngs,
            )
            self.recency_dense1 = nnx.Linear(
                config.recency_embedding_dim, r1, rngs=rngs
            )
            self.recency_dense2 = nnx.Linear(r1, r2, rngs=rngs)
            self.recency_out = nnx.Linear(r2, 1, use_bias=False, rngs=rngs)

            gate_in = config.news_dim + config.recency_embedding_dim
            self.gate_dense1 = nnx.Linear(gate_in, g1, rngs=rngs)
            self.gate_dense2 = nnx.Linear(g1, g2, rngs=rngs)
            self.gate_out = nnx.Linear(g2, 1, rngs=rngs)

        if config.use_ctr:
            self.ctr_scaler = nnx.Linear(1, 1, use_bias=False, rngs=rngs)
            self.ctr_scaler.kernel.value = jnp.full(
                (1, 1), config.ctr_scaler_init, dtype=jnp.float32
            )

    def __call__(
        self,
        bias_news_vec: jax.Array,
        recency_indices: jax.Array | None = None,
        ctr_values: jax.Array | None = None,
    ) -> jax.Array:
        x = jnp.tanh(self.content_dense1(bias_news_vec))
        x = jnp.tanh(self.content_dense2(x))
        x = self.content_dense3(x)
        content_score = jnp.squeeze(self.content_out(x), axis=-1)

        pop_score = content_score

        if self.config.use_recency and recency_indices is not None:
            recency_emb = self.recency_embedding(recency_indices.astype(jnp.int32))
            r = jnp.tanh(self.recency_dense1(recency_emb))
            r = jnp.tanh(self.recency_dense2(r))
            recency_score = jnp.squeeze(self.recency_out(r), axis=-1)

            gate_input = jnp.concatenate([bias_news_vec, recency_emb], axis=-1)
            g = jnp.tanh(self.gate_dense1(gate_input))
            g = jnp.tanh(self.gate_dense2(g))
            gate = jnp.squeeze(jax.nn.sigmoid(self.gate_out(g)), axis=-1)

            pop_score = (1.0 - gate) * content_score + gate * recency_score

        if self.config.use_ctr and ctr_values is not None:
            ctr_input = jnp.expand_dims(ctr_values.astype(jnp.float32), axis=-1)
            ctr_score = jnp.squeeze(self.ctr_scaler(ctr_input), axis=-1)
            pop_score = pop_score + ctr_score

        return pop_score


class ActivityGater(nnx.Module):
    """Per-user gate balancing relevance vs popularity."""

    def __init__(self, config: PPRecConfig, *, rngs: nnx.Rngs):
        h = config.activity_gate_hidden_dim
        self.dense1 = nnx.Linear(config.news_dim, h, rngs=rngs)
        self.dense2 = nnx.Linear(h, 1, rngs=rngs)

    def __call__(self, user_vec: jax.Array) -> jax.Array:
        x = jnp.tanh(self.dense1(user_vec))
        return jnp.squeeze(jax.nn.sigmoid(self.dense2(x)), axis=-1)
