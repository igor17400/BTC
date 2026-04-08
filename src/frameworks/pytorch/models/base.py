"""Base model class for PyTorch news recommendation models.

The actual evaluation logic lives in :mod:`src.core.models.evaluation`
and is shared by Keras, PyTorch, and JAX. This base class only declares
the contract that concrete models (NRMS / NAML / LSTUR / PP-Rec) must
satisfy so the shared evaluator can drive them.
"""

from __future__ import annotations

import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for PyTorch news recommendation models.

    Subclasses must set the following attributes during ``__init__``:

    * ``news_encoder`` -- an ``nn.Module`` that maps news features to
      vectors. The shared evaluator calls it via the PyTorch adapter as
      ``encoder(features, training=False)``.
    * ``user_encoder`` -- an ``nn.Module`` that maps user history to
      vectors. For LSTUR-style models the encoder is called as
      ``encoder([features, user_ids], training=False)`` and the model
      sets ``process_user_id = True``.
    * ``process_user_id`` -- whether the user encoder requires explicit
      user IDs alongside history.
    """

    def __init__(self):
        super().__init__()
        # Subclasses must set these in their __init__
        self.news_encoder: nn.Module
        self.user_encoder: nn.Module
        self.process_user_id: bool = False
