"""Tests for PyTorch model implementations."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


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
        from src.frameworks.pytorch.models.nrms import NRMS

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
        model.eval()

        hist_tokens = torch.randint(0, 100, (2, 10, 30))
        cand_tokens = torch.randint(0, 100, (2, 5, 30))
        output = model(hist_tokens, cand_tokens, training=True)
        assert output.shape == (2, 5)

    def test_news_encoder(self):
        from src.frameworks.pytorch.models.nrms import NewsEncoder, NRMSConfig

        config = NRMSConfig(embedding_size=300, multiheads=4, head_dim=16, max_title_length=30)
        embedding_layer = torch.nn.Embedding(100, 300)
        encoder = NewsEncoder(config, embedding_layer)

        tokens = torch.randint(0, 100, (4, 30))
        output = encoder(tokens)
        assert output.shape == (4, 300)


class TestNAML:
    def test_forward_training(self):
        from src.frameworks.pytorch.models.naml import NAML

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
        model.eval()

        hist = torch.randint(0, 100, (2, 10, 82))  # 30+50+1+1
        cand = torch.randint(0, 100, (2, 5, 82))
        output = model.score_training(hist, cand)
        assert output.shape == (2, 5)


class TestLSTUR:
    def test_forward_training(self):
        from src.frameworks.pytorch.models.lstur import LSTUR

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
        model.eval()

        hist_tokens = torch.randint(0, 100, (2, 10, 30))
        cand_tokens = torch.randint(0, 100, (2, 5, 30))
        user_ids = torch.randint(0, 100, (2,))
        output = model(hist_tokens, cand_tokens, user_ids, training=True)
        assert output.shape == (2, 5)
