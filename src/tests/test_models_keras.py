"""Tests for Keras model implementations."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras


def _make_processed_news(vocab_size=100, embedding_size=300, num_categories=10, num_subcategories=20):
    """Create dummy processed_news dict for testing."""
    return {
        "vocab_size": vocab_size,
        "embeddings": np.random.randn(vocab_size, embedding_size).astype(np.float32),
        "num_categories": num_categories,
        "num_subcategories": num_subcategories,
    }


class TestNRMS:
    def test_forward_training(self):
        from src.frameworks.keras.models.nrms import NRMS

        processed_news = _make_processed_news()
        model = NRMS(
            processed_news=processed_news,
            embedding_size=300,
            multiheads=4,
            head_dim=16,
            max_title_length=30,
            max_history_length=10,
            max_impressions_length=5,
        )

        inputs = {
            "hist_tokens": np.random.randint(0, 100, (2, 10, 30)),
            "cand_tokens": np.random.randint(0, 100, (2, 5, 30)),
        }
        output = model(inputs, training=True)
        assert output.shape == (2, 5)

    def test_forward_inference(self):
        from src.frameworks.keras.models.nrms import NRMS

        processed_news = _make_processed_news()
        model = NRMS(
            processed_news=processed_news,
            embedding_size=300,
            multiheads=4,
            head_dim=16,
            max_title_length=30,
            max_history_length=10,
            max_impressions_length=5,
        )

        inputs = {
            "hist_tokens": np.random.randint(0, 100, (2, 10, 30)),
            "cand_tokens": np.random.randint(0, 100, (2, 5, 30)),
        }
        output = model(inputs, training=False)
        assert output.shape == (2, 5)

    def test_news_encoder(self):
        from src.frameworks.keras.models.nrms import NewsEncoder, NRMSConfig

        config = NRMSConfig(embedding_size=300, multiheads=4, head_dim=16, max_title_length=30)
        embedding_layer = keras.layers.Embedding(100, 300)
        encoder = NewsEncoder(config, embedding_layer)

        tokens = np.random.randint(0, 100, (4, 30))
        output = encoder(tokens, training=False)
        assert output.shape == (4, 300)


class TestNAML:
    def test_forward_training(self):
        from src.frameworks.keras.models.naml import NAML

        processed_news = _make_processed_news()
        model = NAML(
            processed_news=processed_news,
            max_title_length=30,
            max_abstract_length=50,
            embedding_size=300,
            cnn_filter_num=200,
            max_history_length=10,
            max_impressions_length=5,
        )

        inputs = {
            "hist_tokens": np.random.randint(0, 100, (2, 10, 30)),
            "cand_tokens": np.random.randint(0, 100, (2, 5, 30)),
            "hist_abstract_tokens": np.random.randint(0, 100, (2, 10, 50)),
            "cand_abstract_tokens": np.random.randint(0, 100, (2, 5, 50)),
            "hist_category": np.random.randint(0, 10, (2, 10)),
            "cand_category": np.random.randint(0, 10, (2, 5)),
            "hist_subcategory": np.random.randint(0, 20, (2, 10)),
            "cand_subcategory": np.random.randint(0, 20, (2, 5)),
        }
        output = model(inputs, training=True)
        assert output.shape == (2, 5)


class TestLSTUR:
    def test_forward_training(self):
        from src.frameworks.keras.models.lstur import LSTUR

        processed_news = _make_processed_news()
        model = LSTUR(
            processed_news=processed_news,
            num_users=100,
            embedding_size=300,
            cnn_filter_num=200,
            gru_unit=200,
            max_title_length=30,
            max_history_length=10,
            max_impressions_length=5,
        )

        inputs = {
            "hist_tokens": np.random.randint(0, 100, (2, 10, 30)),
            "cand_tokens": np.random.randint(0, 100, (2, 5, 30)),
            "user_ids": np.random.randint(0, 100, (2,)),
        }
        output = model(inputs, training=True)
        assert output.shape == (2, 5)
