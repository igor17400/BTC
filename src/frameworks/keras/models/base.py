"""Base model class for Keras news recommendation models.

The actual evaluation logic lives in :mod:`src.core.models.evaluation`
and is shared by Keras, PyTorch, and JAX. This base class only declares
the contract that concrete models (NRMS / NAML / LSTUR / PP-Rec) must
satisfy so the shared evaluator can drive them.
"""

from __future__ import annotations

import keras
import numpy as np


class BaseModel(keras.Model):
    """Base class for Keras news recommendation models.

    Subclasses must set the following attributes during ``__init__`` /
    ``build``:

    * ``news_encoder`` -- a ``keras.Model`` that maps news features to
      vectors. The shared evaluator calls it via the Keras adapter as
      ``encoder(features, training=False)``.
    * ``user_encoder`` -- a ``keras.Model`` that maps user history to
      vectors. For LSTUR-style models the encoder is called as
      ``encoder([features, user_ids], training=False)`` and the model
      sets ``process_user_id = True``.
    * ``process_user_id`` -- whether the user encoder requires explicit
      user IDs alongside history.

    Subclasses must also set ``self.config`` (a model-specific Config
    dataclass) and ``self.processed_news`` (the dataset's processed
    news dict). ``_validate_processed_news`` is provided as a helper.
    """

    def __init__(self, name: str = "base_news_recommender"):
        super().__init__(name=name)

        # Attributes to be set by subclasses
        self.news_encoder: keras.Model | None = None
        self.user_encoder: keras.Model | None = None
        self.process_user_id: bool = False
        self.float_dtype: str = "float32"  # Will be set by training config

    def _validate_processed_news(self) -> None:
        """Validate processed news data integrity."""
        required_keys = [
            "vocab_size",
            "embeddings",
            "num_categories",
            "num_subcategories",
        ]
        for key in required_keys:
            if key not in self.processed_news:
                raise ValueError(f"Missing required key '{key}' in processed_news")

        embeddings_matrix = self.processed_news["embeddings"]
        if np.isnan(embeddings_matrix).any():
            raise ValueError("Embeddings matrix contains NaN values")

        if embeddings_matrix.shape[1] != self.config.embedding_size:
            raise ValueError(
                f"Embeddings dimension {embeddings_matrix.shape[1]} doesn't match "
                f"configured embedding_size {self.config.embedding_size}"
            )
