"""Base model class for Flax NNX news recommendation models.

The actual evaluation logic lives in :mod:`src.core.models.evaluation`
and is shared by Keras, PyTorch, and JAX. This base class only declares
the contract that concrete models (NRMS / NAML / LSTUR / PP-Rec) must
satisfy so the shared evaluator can drive them.
"""

from __future__ import annotations

from flax import nnx


class BaseModel(nnx.Module):
    """Base class for Flax NNX news recommendation models.

    Subclasses must set the following attributes during ``__init__``:

    * ``news_encoder`` -- an ``nnx.Module`` that maps news features to
      vectors. The shared evaluator calls it via the JAX adapter as
      ``encoder(features, training=False)``.
    * ``user_encoder`` -- an ``nnx.Module`` that maps user history to
      vectors. For LSTUR-style models the encoder is called as
      ``encoder(features, user_ids, training=False)`` and the model
      sets ``process_user_id = True``.
    * ``process_user_id`` -- whether the user encoder requires explicit
      user IDs alongside history.
    """

    # These will be assigned by concrete subclasses.
    news_encoder: nnx.Module | None
    user_encoder: nnx.Module | None
    process_user_id: bool
