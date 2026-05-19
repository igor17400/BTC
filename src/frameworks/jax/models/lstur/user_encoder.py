"""LSTUR user encoder (Flax NNX) — GRU + long-term user embedding.

Mirror of the PyTorch user encoder. Two variants:
    * ``"ini"`` — user embedding initialises the GRU hidden state.
    * ``"con"`` — GRU output is concatenated with the user embedding and
      projected through a linear layer.

Pipeline:
    news_encoder(history) -> (B, H, news_dim)
    user_embedding(user_idx) -> (B, gru_unit)        # long-term repr
        -> Bernoulli mask during training (paper §3.2)
    "ini": RNN(GRUCell(news_dim -> gru_unit)) with long-term repr as h0
           -> last state                              # short-term repr
    "con": RNN(GRUCell(news_dim -> gru_unit)) with zero h0
           -> last state
           -> Linear(2*gru_unit -> gru_unit) over concat(short, long)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import LSTURConfig

from .news_encoder import NewsEncoder


class UserEncoder(nnx.Module):
    """LSTUR user encoder."""

    def __init__(
        self,
        config: LSTURConfig,
        news_encoder: NewsEncoder,
        num_users: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder

        self.user_embedding = nnx.Embed(
            num_embeddings=num_users,
            features=config.gru_unit,
            rngs=rngs,
        )
        # Zero-init the long-term user table (matches the reference Keras model).
        self.user_embedding.embedding.value = jnp.zeros_like(
            self.user_embedding.embedding.value
        )

        self.user_embedding_dropout = nnx.Dropout(
            rate=config.user_embedding_dropout_rate, rngs=rngs
        )

        self.gru = nnx.RNN(
            nnx.GRUCell(
                in_features=news_encoder.output_dim,
                hidden_features=config.gru_unit,
                rngs=rngs,
            ),
        )

        if config.type == "con":
            self.concat_dense = nnx.Linear(
                in_features=config.gru_unit * 2,
                out_features=config.gru_unit,
                rngs=rngs,
            )

    def __call__(
        self,
        history_features: jax.Array,
        user_indices: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Encode a user from browsing history + user id.

        Args:
            history_features: ``(B, H)`` plain news_idx, or ``(B, H, k)``
                packed ``[news_idx | category | subcategory]``.
            user_indices: ``(B,)`` or ``(B, 1)`` int.
            training: Controls dropout.

        Returns:
            ``(B, gru_unit)`` user representations.
        """
        if user_indices.ndim == 1:
            user_indices = jnp.expand_dims(user_indices, axis=-1)
        long_u_emb = self.user_embedding(user_indices)  # (B, 1, gru_unit)
        long_u_emb = jnp.squeeze(long_u_emb, axis=1)
        long_u_emb = self.user_embedding_dropout(long_u_emb, deterministic=not training)

        history_repr = self.news_encoder(
            history_features, training=training
        )  # (B, H, news_dim)

        if self.config.type == "ini":
            all_states = self.gru(history_repr, initial_carry=long_u_emb)
            user_present = all_states[:, -1, :]
        elif self.config.type == "con":
            all_states = self.gru(history_repr)
            short_emb = all_states[:, -1, :]
            concat_emb = jnp.concatenate([short_emb, long_u_emb], axis=-1)
            user_present = self.concat_dense(concat_emb)
        else:
            raise ValueError(f"Invalid user encoder type: {self.config.type}")

        return user_present
