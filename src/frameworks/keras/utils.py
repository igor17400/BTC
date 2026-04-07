"""Keras-specific utilities: JIT warmup, metric wrappers, test step."""

import logging

import keras

from src.core.metrics.functions import NewsRecommenderMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# JIT Warmup
# =============================================================================


def warmup_jit_compilation(model, sample_batch):
    """Run a forward pass to trigger JIT compilation (avoids slow first epoch)."""
    try:
        if isinstance(sample_batch, tuple):
            inputs = sample_batch[0]
        else:
            inputs = sample_batch

        _ = model(inputs, training=False)
        _ = model(inputs, training=True)

        if hasattr(model, "training_model") and model.training_model is not None:
            hist_key, cand_key = "hist_tokens", "cand_tokens"
            if hist_key in inputs and cand_key in inputs:
                history_parts = [inputs[hist_key]]
                candidate_parts = [inputs[cand_key]]

                for hk, ck in [
                    ("hist_abstract_tokens", "cand_abstract_tokens"),
                    ("hist_category", "cand_category"),
                    ("hist_subcategory", "cand_subcategory"),
                ]:
                    if hk in inputs and ck in inputs:
                        h = inputs[hk]
                        c = inputs[ck]
                        if h.ndim == 1:
                            h = keras.ops.expand_dims(h, axis=-1)
                            c = keras.ops.expand_dims(c, axis=-1)
                        history_parts.append(h)
                        candidate_parts.append(c)

                hist = keras.ops.concatenate(history_parts, axis=-1)
                cand = keras.ops.concatenate(candidate_parts, axis=-1)
                _ = model.training_model([hist, cand], training=False)
                _ = model.training_model([hist, cand], training=True)

        # Block until complete (JAX backend only)
        if keras.backend.backend() == "jax":
            import jax

            jax.block_until_ready(jax.device_put(0))

    except Exception as e:
        logger.warning(f"JIT warmup error (non-critical): {e}")


# =============================================================================
# Keras Metric Wrappers
# =============================================================================


class LightweightNewsMetrics:
    """Lightweight training metrics that avoid expensive per-batch computation."""

    @staticmethod
    def should_use_lightweight_metrics(cfg) -> bool:
        return cfg.eval.fast_evaluation

    @staticmethod
    def create_training_metrics() -> list:
        return [keras.metrics.CategoricalAccuracy(name="accuracy")]


def create_news_metrics(engine: NewsRecommenderMetrics) -> list:
    """Create full Keras metric wrappers for news recommendation."""
    return [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
    ]


# =============================================================================
# Test Step
# =============================================================================


def test_step_fn(
    model: keras.Model,
    data: tuple[dict[str, keras.KerasTensor], keras.KerasTensor],
) -> dict[str, keras.KerasTensor]:
    """Custom test step logic."""
    features, labels = data
    return model.test_on_batch(features, labels, return_dict=True)
