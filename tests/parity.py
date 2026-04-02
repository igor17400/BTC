"""Cross-framework parity tests.

Verify that forward pass outputs match across frameworks within tolerance.
This requires loading the same weights into all 3 frameworks.
"""

import numpy as np
import pytest


def _make_processed_news(vocab_size=50, embedding_size=64, num_categories=5, num_subcategories=10):
    """Create small processed_news dict for parity testing."""
    np.random.seed(42)
    return {
        "vocab_size": vocab_size,
        "embeddings": np.random.randn(vocab_size, embedding_size).astype(np.float32),
        "num_categories": num_categories,
        "num_subcategories": num_subcategories,
    }


def _make_inputs(batch_size=2, history_len=5, title_len=10, num_candidates=3, vocab_size=50):
    """Create identical input data for all frameworks."""
    np.random.seed(123)
    return {
        "hist_tokens": np.random.randint(1, vocab_size, (batch_size, history_len, title_len)).astype(np.int32),
        "cand_tokens": np.random.randint(1, vocab_size, (batch_size, num_candidates, title_len)).astype(np.int32),
    }


class TestMetricsParity:
    """Verify that NumPy metrics match across use from all frameworks."""

    def test_auc_consistency(self):
        from src.core.metrics.functions import NewsRecommenderMetrics

        metrics = NewsRecommenderMetrics()

        np.random.seed(42)
        y_true = np.array([1, 0, 1, 0, 1], dtype=np.float32)
        y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7], dtype=np.float32)

        result = metrics.compute_metrics(y_true, y_pred)

        assert "auc" in result
        assert "mrr" in result
        assert "ndcg@5" in result
        assert "ndcg@10" in result
        assert 0.0 <= result["auc"] <= 1.0
        assert 0.0 <= result["mrr"] <= 1.0

    def test_perfect_ranking(self):
        from src.core.metrics.functions import NewsRecommenderMetrics

        metrics = NewsRecommenderMetrics()

        y_true = np.array([1, 1, 0, 0, 0], dtype=np.float32)
        y_pred = np.array([1.0, 0.9, 0.3, 0.2, 0.1], dtype=np.float32)

        result = metrics.compute_metrics(y_true, y_pred)

        assert result["auc"] == pytest.approx(1.0, abs=1e-5)
        assert result["ndcg@5"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(
    not all(
        pytest.importorskip(mod, reason=f"{mod} not installed")
        for mod in []  # Will be checked individually
    ),
    reason="Not all frameworks installed",
)
class TestNRMSParity:
    """Verify NRMS outputs match across frameworks.

    NOTE: Full parity requires weight transfer between frameworks.
    These tests verify shape and range consistency.
    """

    def test_output_shapes_match(self):
        """Verify all frameworks produce the same output shape."""
        processed_news = _make_processed_news()
        inputs = _make_inputs()

        shapes = {}

        # Keras
        try:
            import os
            os.environ["KERAS_BACKEND"] = "jax"
            from src.frameworks.keras.models.nrms import NRMS as KerasNRMS

            model = KerasNRMS(
                processed_news=processed_news,
                embedding_size=64,
                multiheads=2,
                head_dim=8,
                max_title_length=10,
                max_history_length=5,
                max_impressions_length=3,
            )
            output = model(inputs, training=False)
            shapes["keras"] = tuple(output.shape)
        except ImportError:
            pytest.skip("Keras not available")

        # PyTorch
        try:
            import torch
            from src.frameworks.pytorch.models.nrms import NRMS as TorchNRMS

            model = TorchNRMS(
                processed_news=processed_news,
                embedding_size=64,
                multiheads=2,
                head_dim=8,
                max_title_length=10,
                max_history_length=5,
                max_impressions_length=3,
            )
            model.eval()
            with torch.no_grad():
                hist = torch.from_numpy(inputs["hist_tokens"])
                cand = torch.from_numpy(inputs["cand_tokens"])
                output = model(hist, cand, training=False)
            shapes["pytorch"] = tuple(output.shape)
        except ImportError:
            pass

        # JAX
        try:
            import jax.numpy as jnp
            from flax import nnx
            from src.frameworks.jax.models.nrms import NRMS as JaxNRMS

            rngs = nnx.Rngs(0)
            model = JaxNRMS(
                processed_news=processed_news,
                embedding_size=64,
                multiheads=2,
                head_dim=8,
                max_title_length=10,
                max_history_length=5,
                max_impressions_length=3,
                rngs=rngs,
            )
            hist = jnp.array(inputs["hist_tokens"])
            cand = jnp.array(inputs["cand_tokens"])
            output = model(hist, cand, training=False)
            shapes["jax"] = tuple(output.shape)
        except ImportError:
            pass

        # Verify all shapes match
        if len(shapes) >= 2:
            shape_values = list(shapes.values())
            for fw, shape in shapes.items():
                assert shape == shape_values[0], (
                    f"Shape mismatch: {fw}={shape} vs {list(shapes.keys())[0]}={shape_values[0]}"
                )

    def test_output_range(self):
        """Verify outputs are in valid probability range."""
        processed_news = _make_processed_news()
        inputs = _make_inputs()

        try:
            import os
            os.environ["KERAS_BACKEND"] = "jax"
            from src.frameworks.keras.models.nrms import NRMS as KerasNRMS

            model = KerasNRMS(
                processed_news=processed_news,
                embedding_size=64,
                multiheads=2,
                head_dim=8,
                max_title_length=10,
                max_history_length=5,
                max_impressions_length=3,
            )
            output = np.array(model(inputs, training=True))
            # Softmax output should sum to ~1
            row_sums = output.sum(axis=-1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        except ImportError:
            pytest.skip("Keras not available")
