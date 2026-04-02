"""Keras framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for Keras training,
keeping train.py as a thin dispatcher.
"""

import os

from omegaconf import DictConfig
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.io.saving import get_output_run_dir
from src.core.io.logging import console, setup_wandb_session


def _setup(cfg: DictConfig):
    """Setup Keras backend and precision."""
    os.environ["KERAS_BACKEND"] = "jax"
    import keras

    # Precision setup
    precision = getattr(cfg.device, "precision", "float32")
    precision_map = {
        "float32": "float32",
        "float16": "mixed_float16",
        "bfloat16": "mixed_bfloat16",
    }
    policy_name = precision_map.get(precision, "float32")
    if precision not in precision_map:
        console.log(f"[yellow]Warning: Invalid precision '{precision}'. Using 'float32'.[/yellow]")
        policy_name = "float32"

    console.log(f"Setting Keras precision policy to '{policy_name}' (precision: {precision}).")
    policy = keras.mixed_precision.Policy(policy_name)
    keras.mixed_precision.set_global_policy(policy)
    console.log(f"  Compute dtype: {policy.compute_dtype}")
    console.log(f"  Variable dtype: {policy.variable_dtype}")

    if precision in ["float16", "bfloat16"]:
        console.log(f"  [green]Mixed precision enabled: Computations in {precision}, variables in float32[/green]")

    # Set random seeds
    keras.utils.set_random_seed(cfg.seed)

    # Device setup
    from src.frameworks.keras.device import setup_device
    setup_device(
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else [],
        memory_limit=cfg.device.memory_limit if hasattr(cfg.device, "memory_limit") else 0.9,
    )


def run(cfg: DictConfig):
    """Run training with Keras framework."""
    _setup(cfg)

    from src.frameworks.keras.utils import (
        initialize_model_and_dataset,
        initialize_model_from_spec,
        create_news_metrics,
        LightweightNewsMetrics,
    )
    from src.frameworks.keras.training import training_loop_orchestrator

    # Initialize WandB
    setup_wandb_session(cfg)

    # Prepare training metrics
    if LightweightNewsMetrics.should_use_lightweight_metrics(cfg):
        training_metrics = LightweightNewsMetrics.create_training_metrics()
        console.log("Using lightweight metrics during training with custom metrics in callbacks")
    else:
        training_metrics = create_news_metrics(
            NewsRecommenderMetrics(**cfg.metrics.params if hasattr(cfg.metrics, "params") else {})
        )
        console.log("Using full custom metrics during training")

    # Model and Dataset Initialization
    has_spec = hasattr(cfg, "spec") and cfg.spec is not None
    if has_spec:
        console.log("Using YAML DSL spec for model initialization.")
        model, dataset_provider = initialize_model_from_spec(cfg, training_metrics)
    else:
        model, dataset_provider = initialize_model_and_dataset(cfg, training_metrics)

    # Metrics Calculator
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Setup output directory
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"All outputs for this run will be saved in: {output_run_dir.resolve()}")

    # Rich Progress Bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("({task.completed} of {task.total} batches)"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as global_progress_bar:
        training_loop_orchestrator(
            model,
            dataset_provider,
            cfg,
            metrics_engine,
            global_progress_bar,
            output_run_dir,
        )

    console.log(f"--- {cfg.model_name} Training Run Finished ---")
