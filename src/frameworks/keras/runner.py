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

from src.core.io.logging import console, setup_wandb_session
from src.core.io.saving import get_output_run_dir
from src.core.metrics.functions import NewsRecommenderMetrics


SUPPORTED_BACKENDS = ("jax", "torch")


def _setup(cfg: DictConfig):
    """Setup Keras backend and precision."""
    backend = getattr(cfg.device, "keras_backend", "jax")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported Keras backend: '{backend}'. Use one of: {SUPPORTED_BACKENDS}"
        )

    # KERAS_BACKEND must be set before first import. If keras is already
    # loaded with a different backend, we cannot switch in-process.
    current = os.environ.get("KERAS_BACKEND")
    if current and current != backend:
        raise RuntimeError(
            f"Cannot switch Keras backend from '{current}' to '{backend}' "
            f"in the same process. Run in a separate process."
        )
    os.environ["KERAS_BACKEND"] = backend
    import keras

    console.log(f"Keras backend: {keras.backend.backend()}")

    # Precision setup
    precision = getattr(cfg.device, "precision", "float32")
    precision_map = {
        "float32": "float32",
        "float16": "mixed_float16",
        "bfloat16": "mixed_bfloat16",
    }
    policy_name = precision_map.get(precision, "float32")
    if precision not in precision_map:
        console.log(
            f"[yellow]Warning: Invalid precision '{precision}'. Using 'float32'.[/yellow]"
        )
        policy_name = "float32"

    console.log(
        f"Setting Keras precision policy to '{policy_name}' (precision: {precision})."
    )
    policy = keras.mixed_precision.Policy(policy_name)
    keras.mixed_precision.set_global_policy(policy)
    console.log(f"  Compute dtype: {policy.compute_dtype}")
    console.log(f"  Variable dtype: {policy.variable_dtype}")

    if precision in ["float16", "bfloat16"]:
        console.log(
            f"  [green]Mixed precision enabled: Computations in {precision}, variables in float32[/green]"
        )

    # Set random seeds
    keras.utils.set_random_seed(cfg.seed)

    # Device setup
    from src.frameworks.keras.device import setup_device

    setup_device(
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else [],
        memory_limit=cfg.device.memory_limit
        if hasattr(cfg.device, "memory_limit")
        else 0.9,
    )


def _ensure_keras_dataloaders(dataset_provider):
    """Add Keras dataloader methods if the dataset doesn't have them."""
    if hasattr(dataset_provider, "train_dataloader"):
        return dataset_provider

    from src.frameworks.keras.dataloaders import (
        ImpressionIterator,
        NewsBatchDataloader,
        NewsDataLoader,
        UserHistoryBatchDataloader,
    )

    import numpy as np

    def train_dataloader(batch_size, model_name="nrms"):
        data = dataset_provider.train_behaviors_data
        return NewsDataLoader.create_train_dataset(
            history_news_tokens=data["history_news_tokens"],
            history_news_abstract_tokens=data["history_news_abstract_tokens"],
            history_news_category=data["history_news_categories"],
            history_news_subcategory=data["history_news_subcategories"],
            candidate_news_tokens=data["candidate_news_tokens"],
            candidate_news_abstract_tokens=data["candidate_news_abstract_tokens"],
            candidate_news_category=data["candidate_news_categories"],
            candidate_news_subcategory=data["candidate_news_subcategories"],
            labels=data["labels"],
            user_ids=data["user_ids"],
            batch_size=batch_size,
            process_title=dataset_provider.process_title,
            process_abstract=dataset_provider.process_abstract,
            process_category=dataset_provider.process_category,
            process_subcategory=dataset_provider.process_subcategory,
            process_user_id=dataset_provider.process_user_id,
            model_name=model_name,
        )

    def user_history_dataloader(mode, batch_size=32):
        data = dataset_provider.val_behaviors_data if mode == "val" else dataset_provider.test_behaviors_data
        return UserHistoryBatchDataloader(
            history_tokens=data["history_news_tokens"],
            history_abstract_tokens=data["history_news_abstract_tokens"],
            history_category=data["history_news_categories"],
            history_subcategory=data["history_news_subcategories"],
            impression_ids=data["impression_ids"],
            user_ids=data["user_ids"],
            batch_size=batch_size,
            process_title=dataset_provider.process_title,
            process_abstract=dataset_provider.process_abstract,
            process_category=dataset_provider.process_category,
            process_subcategory=dataset_provider.process_subcategory,
        )

    def impression_dataloader(mode):
        data = dataset_provider.val_behaviors_data if mode == "val" else dataset_provider.test_behaviors_data
        return ImpressionIterator(
            impression_tokens=data["candidate_news_tokens"],
            impression_abstract_tokens=data["candidate_news_abstract_tokens"],
            impression_category=data["candidate_news_categories"],
            impression_subcategory=data["candidate_news_subcategories"],
            labels=data["labels"],
            impression_ids=data["impression_ids"],
            candidate_ids=data["candidate_news_ids"],
            process_title=dataset_provider.process_title,
            process_abstract=dataset_provider.process_abstract,
            process_category=dataset_provider.process_category,
            process_subcategory=dataset_provider.process_subcategory,
        )

    def news_dataloader(batch_size=64):
        pn = dataset_provider.processed_news
        return NewsBatchDataloader(
            news_ids=np.array(pn["news_ids_original_strings"]),
            news_tokens=pn["tokens"],
            news_abstract_tokens=pn["abstract_tokens"],
            news_category_indices=pn["category_indices"],
            news_subcategory_indices=pn["subcategory_indices"],
            batch_size=batch_size,
            process_title=dataset_provider.process_title,
            process_abstract=dataset_provider.process_abstract,
            process_category=dataset_provider.process_category,
            process_subcategory=dataset_provider.process_subcategory,
        )

    dataset_provider.train_dataloader = train_dataloader
    dataset_provider.user_history_dataloader = user_history_dataloader
    dataset_provider.impression_dataloader = impression_dataloader
    dataset_provider.news_dataloader = news_dataloader
    return dataset_provider


def run(cfg: DictConfig):
    """Run training with Keras framework."""
    _setup(cfg)

    from src.frameworks.keras.training import training_loop_orchestrator
    from src.frameworks.keras.utils import (
        LightweightNewsMetrics,
        create_news_metrics,
        initialize_model_and_dataset,
        initialize_model_from_spec,
    )

    # Initialize WandB
    setup_wandb_session(cfg)

    # Prepare training metrics
    if LightweightNewsMetrics.should_use_lightweight_metrics(cfg):
        training_metrics = LightweightNewsMetrics.create_training_metrics()
        console.log(
            "Using lightweight metrics during training with custom metrics in callbacks"
        )
    else:
        training_metrics = create_news_metrics(
            NewsRecommenderMetrics(
                **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
            )
        )
        console.log("Using full custom metrics during training")

    # Model and Dataset Initialization
    has_spec = hasattr(cfg, "spec") and cfg.spec is not None
    if has_spec:
        console.log("Using YAML DSL spec for model initialization.")
        model, dataset_provider = initialize_model_from_spec(cfg, training_metrics)
    else:
        model, dataset_provider = initialize_model_and_dataset(cfg, training_metrics)

    # Ensure Keras dataloaders are available (for SyntheticDataset or other minimal datasets)
    dataset_provider = _ensure_keras_dataloaders(dataset_provider)

    # Metrics Calculator
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Setup output directory
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    console.log(
        f"All outputs for this run will be saved in: {output_run_dir.resolve()}"
    )

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
        best_epoch_metrics, test_metrics = training_loop_orchestrator(
            model,
            dataset_provider,
            cfg,
            metrics_engine,
            global_progress_bar,
            output_run_dir,
        )

    return test_metrics or best_epoch_metrics or {}
