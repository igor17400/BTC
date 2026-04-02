"""Keras framework utilities.

Consolidates JIT warmup, model/dataset initialization, Keras metric wrappers,
and test step functions for the Keras framework.
"""

import inspect
import logging
from typing import Any

import hydra
import keras
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.frameworks.keras.losses import get_loss
from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.io.logging import console

logger = logging.getLogger(__name__)


# =============================================================================
# JIT Warmup
# =============================================================================

def warmup_jit_compilation(model, sample_batch):
    """Warm up JIT compilation by running a forward pass with sample data.

    Args:
        model: The Keras model to warm up
        sample_batch: A sample batch of data matching the model's expected input format
                     Can be either a dict of inputs or a tuple of (inputs, labels)
    """
    logger.info("Warming up JIT compilation...")

    try:
        # Handle different input formats
        if isinstance(sample_batch, tuple):
            # If it's a tuple, assume it's (inputs, labels) from dataloader
            inputs = sample_batch[0]
            labels = sample_batch[1] if len(sample_batch) > 1 else None
        else:
            # If it's already a dict, use it directly
            inputs = sample_batch
            labels = None

        # Run a forward pass to trigger compilation
        _ = model(inputs, training=False)
        _ = model(inputs, training=True)

        # If the model has internal models, warm them up too
        if hasattr(model, 'training_model') and model.training_model is not None:
            # For training model, we need to prepare the proper input format

            # Our base keys
            base_hist_key = "hist_tokens"
            base_cand_key = "cand_tokens"

            # A list of optional key pairs to check for.
            optional_key_pairs = [
                ("hist_abstract_tokens", "cand_abstract_tokens"),
                ("hist_category", "cand_category"),
                ("hist_subcategory", "cand_subcategory"),
            ]

            # Check if the base tokens exist. If not, we can't proceed.
            if base_hist_key not in inputs or base_cand_key not in inputs:
                print("Warning: Base tokens are missing. Cannot perform concatenation.")
                return None, None

            # Start with the base tokens as the initial arrays to concatenate.
            history_to_concat = [inputs[base_hist_key]]
            candidate_to_concat = [inputs[base_cand_key]]

            # Iterate through the optional keys and add them if they exist.
            for hist_key, cand_key in optional_key_pairs:
                if hist_key in inputs and cand_key in inputs:
                    # Check if we need to expand dimensions for scalar values.
                    if inputs[hist_key].ndim == 1:
                        history_to_concat.append(keras.ops.expand_dims(inputs[hist_key], axis=-1))
                        candidate_to_concat.append(keras.ops.expand_dims(inputs[cand_key], axis=-1))
                    else:
                        history_to_concat.append(inputs[hist_key])
                        candidate_to_concat.append(inputs[cand_key])

            # Perform the concatenation.
            history_concat = keras.ops.concatenate(history_to_concat, axis=-1)
            candidate_concat = keras.ops.concatenate(candidate_to_concat, axis=-1)

            # Warm up training model with concatenated inputs
            _ = model.training_model([history_concat, candidate_concat], training=False)
            _ = model.training_model([history_concat, candidate_concat], training=True)

        if hasattr(model, 'scorer_model') and model.scorer_model is not None:
            # Scorer model warmup - it has different input shapes
            # We'll skip this for now as it's less critical
            pass

        # Block until computations are complete (backend-specific)
        if keras.backend.backend() == "jax":
            import jax
            jax.block_until_ready(jax.device_put(0))

        logger.info("JIT compilation warmup completed")

    except Exception as e:
        logger.warning(f"JIT warmup encountered an error (non-critical): {e}")
        # Continue anyway - warmup is optimization, not required


# =============================================================================
# Model and Dataset Initialization
# =============================================================================

def initialize_model_and_dataset(cfg: DictConfig, training_metrics: list = None) -> tuple[keras.Model, Any]:
    """Instantiate dataset and model based on Hydra configuration."""
    console.log("Initializing dataset provider...")
    dataset_provider: Any = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news_data = dataset_provider.processed_news

    console.log(f"Initializing model: {cfg.model._target_}...")

    # --- Generic Model Instantiation ---
    model_class = hydra.utils.get_class(cfg.model._target_)
    model_signature = inspect.signature(model_class.__init__)

    # Start with parameters from the model's config section
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    # Remove hydra's target, it's not a model parameter
    model_params.pop("_target_", None)

    # Add parameters that are required by the model but are not in its specific config section
    if "processed_news" in model_signature.parameters:
        model_params["processed_news"] = processed_news_data
    if "seed" in model_signature.parameters:
        model_params["seed"] = cfg.seed
    if "max_title_length" in model_signature.parameters:
        model_params["max_title_length"] = cfg.dataset.max_title_length
    if "max_abstract_length" in model_signature.parameters:
        model_params["max_abstract_length"] = cfg.dataset.max_abstract_length
    if "max_history_length" in model_signature.parameters:
        model_params["max_history_length"] = cfg.dataset.max_history_length
    if "max_impressions_length" in model_signature.parameters:
        model_params["max_impressions_length"] = cfg.dataset.max_impressions_length

    # Filter out any parameters that are not in the model's __init__ signature
    valid_params = {k: v for k, v in model_params.items() if k in model_signature.parameters}

    model: keras.Model = model_class(**valid_params)
    # --- End Generic Model Instantiation ---

    console.log(f"Successfully instantiated {model.name} model.")

    optimizer = keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    if (keras.mixed_precision.global_policy().name == "mixed_float16"):
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_function = get_loss(
        loss_name=cfg.model.loss.name,
        from_logits=cfg.model.loss.from_logits,
        reduction=cfg.model.loss.reduction,
        label_smoothing=cfg.model.loss.label_smoothing
    )

    # Compile with metrics if provided
    compile_kwargs = {"optimizer": optimizer, "loss": loss_function}
    if training_metrics is not None:
        compile_kwargs["metrics"] = training_metrics

    model.compile(**compile_kwargs)

    metrics_info = f", Metrics: {len(training_metrics)} metrics" if training_metrics else ""
    console.log(
        f"Model compiled. Optimizer: {type(optimizer).__name__}, Loss: {loss_function.name}{metrics_info}"
    )

    console.log(f"Mixed precision global policy: {keras.mixed_precision.global_policy().name}")
    console.log(f"Optimizer type: {type(optimizer)}")

    return model, dataset_provider


def initialize_model_from_spec(cfg: DictConfig, training_metrics: list = None) -> tuple[keras.Model, Any]:
    """Instantiate dataset and model from a YAML DSL spec.

    This is the spec-aware alternative to initialize_model_and_dataset().
    It reads from cfg.spec instead of cfg.model._target_.
    """
    from src.core.models.spec import build_model_from_spec

    console.log("Initializing dataset provider...")
    dataset_provider: Any = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news_data = dataset_provider.processed_news

    spec = cfg.spec
    framework = getattr(cfg, "framework", "keras")
    console.log(f"Initializing model from spec: {spec.model.name} (framework: {framework})...")

    model = build_model_from_spec(spec, framework, processed_news_data)
    console.log(f"Successfully instantiated {model.name} model from spec.")

    optimizer = keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    if keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Get loss config from spec or model config
    loss_cfg = spec.training.loss
    loss_function = get_loss(
        loss_name=loss_cfg.name,
        from_logits=loss_cfg.get("from_logits", False),
        reduction=loss_cfg.get("reduction", "sum_over_batch_size"),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )

    compile_kwargs = {"optimizer": optimizer, "loss": loss_function}
    if training_metrics is not None:
        compile_kwargs["metrics"] = training_metrics

    model.compile(**compile_kwargs)

    metrics_info = f", Metrics: {len(training_metrics)} metrics" if training_metrics else ""
    console.log(
        f"Model compiled. Optimizer: {type(optimizer).__name__}, Loss: {loss_function.name}{metrics_info}"
    )

    return model, dataset_provider


# =============================================================================
# Keras Metric Wrappers
# =============================================================================

class NewsRecommenderKerasMetric(keras.metrics.Metric):
    """Keras 3 compatible wrapper for news recommendation metrics."""

    def __init__(self, metric_name: str, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(name=metric_name, **kwargs)
        self.metric_name = metric_name
        self.custom_metrics_engine = custom_metrics_engine

        # State variables to accumulate predictions and labels
        self.total_predictions = self.add_weight(name="total_predictions", initializer="zeros", shape=())
        self.total_labels = self.add_weight(name="total_labels", initializer="zeros", shape=())

        # Lists to store batch data (note: this is not ideal for very large datasets)
        self.predictions_list = []
        self.labels_list = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with batch predictions and labels."""
        # Convert to numpy for processing
        y_true_np = keras.ops.convert_to_numpy(y_true)
        y_pred_np = keras.ops.convert_to_numpy(y_pred)

        # Store batch data
        self.predictions_list.append(y_pred_np)
        self.labels_list.append(y_true_np)

        # Update counters
        batch_size = keras.ops.shape(y_true)[0]
        self.total_predictions.assign_add(keras.ops.cast(batch_size, self.dtype))

    def result(self):
        """Compute the final metric value."""
        if not self.predictions_list or not self.labels_list:
            return 0.0

        # Concatenate all batch data
        all_predictions = np.concatenate(self.predictions_list, axis=0)
        all_labels = np.concatenate(self.labels_list, axis=0)

        # Compute custom metrics
        try:
            metrics_dict = self.custom_metrics_engine.compute_metrics(
                y_true=all_labels,
                y_pred_logits=all_predictions,
                progress=None  # No progress bar in metric computation
            )

            # Return the specific metric
            return float(metrics_dict.get(self.metric_name, 0.0))

        except Exception as e:
            print(f"Error computing {self.metric_name}: {e}")
            return 0.0

    def reset_state(self):
        """Reset the metric state."""
        self.total_predictions.assign(0.0)
        self.total_labels.assign(0.0)
        self.predictions_list.clear()
        self.labels_list.clear()


class AUCNewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible AUC metric for news recommendation."""

    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="auc", custom_metrics_engine=custom_metrics_engine, **kwargs)


class MRRNewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible MRR metric for news recommendation."""

    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="mrr", custom_metrics_engine=custom_metrics_engine, **kwargs)


class NDCG5NewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible nDCG@5 metric for news recommendation."""

    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="ndcg@5", custom_metrics_engine=custom_metrics_engine, **kwargs)


class NDCG10NewsMetric(NewsRecommenderKerasMetric):
    """Keras 3 compatible nDCG@10 metric for news recommendation."""

    def __init__(self, custom_metrics_engine: NewsRecommenderMetrics, **kwargs):
        super().__init__(metric_name="ndcg@10", custom_metrics_engine=custom_metrics_engine, **kwargs)


def create_news_metrics(custom_metrics_engine: NewsRecommenderMetrics) -> list:
    """Create a list of Keras 3 compatible news recommendation metrics.

    Args:
        custom_metrics_engine: The custom metrics calculator

    Returns:
        List of Keras metrics that can be used in model.compile()
    """
    return [
        AUCNewsMetric(custom_metrics_engine),
        MRRNewsMetric(custom_metrics_engine),
        NDCG5NewsMetric(custom_metrics_engine),
        NDCG10NewsMetric(custom_metrics_engine),
    ]


class LightweightNewsMetrics:
    """Lightweight version that only computes metrics on validation, not during training."""

    @staticmethod
    def create_training_metrics() -> list:
        """Create lightweight metrics for training monitoring.

        These are standard Keras metrics that provide quick feedback during training
        without the computational overhead of custom news recommendation metrics.
        """
        return [
            'accuracy',  # Standard accuracy for quick monitoring
            keras.metrics.AUC(name='keras_auc'),  # Standard AUC
        ]

    @staticmethod
    def should_use_lightweight_metrics(cfg) -> bool:
        """Determine if lightweight metrics should be used during training.

        Args:
            cfg: Configuration object

        Returns:
            True if lightweight metrics should be used, False otherwise
        """
        # Use lightweight metrics if fast evaluation is enabled
        # (custom metrics will be computed in callbacks)
        return cfg.eval.fast_evaluation


# =============================================================================
# Test Step Function
# =============================================================================

def test_step_fn(
        model: keras.Model, data: tuple[dict[str, keras.KerasTensor], keras.KerasTensor]
) -> dict[str, keras.KerasTensor]:
    """Custom test step logic."""
    features, labels = data

    # Use Keras 3's backend-agnostic test step
    batch_metrics = model.test_on_batch(features, labels, return_dict=True)
    return batch_metrics


