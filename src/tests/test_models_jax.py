"""Tests for JAX/Flax NNX model implementations."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
flax = pytest.importorskip("flax")
from flax import nnx


def _make_processed_news(vocab_size=100, embedding_size=300, num_categories=10, num_subcategories=20):
    """Create dummy processed_news dict for testing."""
    return {
        "vocab_size": vocab_size,
        "embeddings": np.random.randn(vocab_size, embedding_size).astype(np.float32),
        "num_categories": num_categories,
        "num_subcategories": num_subcategories,
    }


class TestNRMS:
    def test_forward(self):
        from src.frameworks.jax.models.nrms import NRMS

        processed_news = _make_processed_news()
        rngs = nnx.Rngs(0)
        model = NRMS(
            processed_news=processed_news,
            embedding_size=300,
            multiheads=4,
            head_dim=16,
            max_title_length=30,
            max_history_length=10,
            max_impressions_length=5,
            rngs=rngs,
        )

        hist_tokens = jnp.array(np.random.randint(0, 100, (2, 10, 30)))
        cand_tokens = jnp.array(np.random.randint(0, 100, (2, 5, 30)))
        output = model(hist_tokens, cand_tokens, training=True)
        assert output.shape == (2, 5)

    def test_news_encoder(self):
        from src.frameworks.jax.models.nrms import NewsEncoder, NRMSConfig

        config = NRMSConfig(embedding_size=300, multiheads=4, head_dim=16, max_title_length=30)
        rngs = nnx.Rngs(0)
        embedding_layer = nnx.Embed(100, 300, rngs=rngs)
        encoder = NewsEncoder(config, embedding_layer, rngs=rngs)

        tokens = jnp.array(np.random.randint(0, 100, (4, 30)))
        output = encoder(tokens, training=False)
        assert output.shape == (4, 300)


class TestNAML:
    def test_forward(self):
        from src.frameworks.jax.models.naml import NAML

        processed_news = _make_processed_news()
        rngs = nnx.Rngs(0)
        model = NAML(
            processed_news=processed_news,
            max_title_length=30,
            max_abstract_length=50,
            embedding_size=300,
            cnn_filter_num=200,
            max_history_length=10,
            max_impressions_length=5,
            rngs=rngs,
        )

        hist = jnp.array(np.random.randint(0, 100, (2, 10, 82)))
        cand = jnp.array(np.random.randint(0, 100, (2, 5, 82)))
        output = model(hist, cand, training=True)
        assert output.shape == (2, 5)


class TestLSTUR:
    def test_forward(self):
        from src.frameworks.jax.models.lstur import LSTUR

        processed_news = _make_processed_news()
        rngs = nnx.Rngs(0)
        model = LSTUR(
            processed_news=processed_news,
            num_users=100,
            embedding_size=300,
            cnn_filter_num=200,
            gru_unit=200,
            max_title_length=30,
            max_history_length=10,
            max_impressions_length=5,
            rngs=rngs,
        )

        hist_tokens = jnp.array(np.random.randint(0, 100, (2, 10, 30)))
        cand_tokens = jnp.array(np.random.randint(0, 100, (2, 5, 30)))
        user_ids = jnp.array(np.random.randint(0, 100, (2,)))
        output = model(hist_tokens, cand_tokens, user_ids, training=True)
        assert output.shape == (2, 5)
