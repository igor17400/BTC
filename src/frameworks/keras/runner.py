"""Keras framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for Keras training.
Same pattern as JAX/PyTorch: dataset → model → build dataloaders → train.
"""

import os

import numpy as np
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
from src.core.models.spec import build_model_from_spec

SUPPORTED_BACKENDS = ("jax", "torch")


def _build_train_features(dataset_provider) -> tuple:
    """Extract raw numpy features and labels from the dataset provider."""
    import keras

    data = dataset_provider.train_behaviors_data
    features = {}

    if dataset_provider.process_title:
        features["hist_tokens"] = keras.ops.convert_to_numpy(data["history_news_tokens"])
        features["cand_tokens"] = keras.ops.convert_to_numpy(data["candidate_news_tokens"])
    if dataset_provider.process_abstract:
        features["hist_abstract_tokens"] = keras.ops.convert_to_numpy(
            data["history_news_abstract_tokens"]
        )
        features["cand_abstract_tokens"] = keras.ops.convert_to_numpy(
            data["candidate_news_abstract_tokens"]
        )
    if dataset_provider.process_category:
        features["hist_category"] = keras.ops.convert_to_numpy(data["history_news_categories"])
        features["cand_category"] = keras.ops.convert_to_numpy(data["candidate_news_categories"])
    if dataset_provider.process_subcategory:
        features["hist_subcategory"] = keras.ops.convert_to_numpy(data["history_news_subcategories"])
        features["cand_subcategory"] = keras.ops.convert_to_numpy(data["candidate_news_subcategories"])
    if dataset_provider.process_user_id:
        features["user_ids"] = keras.ops.convert_to_numpy(data["user_ids"])

    # PP-Rec: concatenate entity and category into hist_tokens/cand_tokens
    if "history_news_entities" in data and dataset_provider.process_entities:
        hist_ent = keras.ops.convert_to_numpy(data["history_news_entities"])
        cand_ent = keras.ops.convert_to_numpy(data["candidate_news_entities"])
        # Append entity indices to token features: (B, H, title_len + max_entities)
        if "hist_tokens" in features:
            features["hist_tokens"] = np.concatenate(
                [features["hist_tokens"], hist_ent], axis=-1
            )
            features["cand_tokens"] = np.concatenate(
                [features["cand_tokens"], cand_ent], axis=-1
            )
    if dataset_provider.process_category and dataset_provider.process_entities:
        # Append category index as last feature
        hist_cat = keras.ops.convert_to_numpy(data["history_news_categories"])
        cand_cat = keras.ops.convert_to_numpy(data["candidate_news_categories"])
        if "hist_tokens" in features:
            features["hist_tokens"] = np.concatenate(
                [features["hist_tokens"], np.expand_dims(hist_cat, axis=-1)], axis=-1
            )
            features["cand_tokens"] = np.concatenate(
                [features["cand_tokens"], np.expand_dims(cand_cat, axis=-1)], axis=-1
            )

    # PP-Rec CTR features
    if "history_news_ctr" in data:
        ctr = keras.ops.convert_to_numpy(data["history_news_ctr"])
        # Discretize history CTR for embedding lookup: ceil(ctr * 200), capped at 199
        features["hist_ctr"] = np.minimum(np.ceil(ctr * 200).astype(np.int32), 199)
        features["cand_ctr"] = keras.ops.convert_to_numpy(data["candidate_news_ctr"])
    if "candidate_news_recency" in data:
        features["cand_recency"] = keras.ops.convert_to_numpy(
            data["candidate_news_recency"]
        )

    labels = keras.ops.convert_to_numpy(data["labels"])
    return features, labels


def _setup(cfg: DictConfig):
    """Setup Keras backend and precision."""
    backend = getattr(cfg.device, "keras_backend", "jax")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported Keras backend: '{backend}'. Use one of: {SUPPORTED_BACKENDS}"
        )

    current = os.environ.get("KERAS_BACKEND")
    if current and current != backend:
        raise RuntimeError(
            f"Cannot switch Keras backend from '{current}' to '{backend}' in the same process."
        )
    os.environ["KERAS_BACKEND"] = backend
    import keras

    console.log(f"Keras backend: {keras.backend.backend()}")

    # Precision
    precision = getattr(cfg.device, "precision", "float32")
    precision_map = {"float32": "float32", "float16": "mixed_float16", "bfloat16": "mixed_bfloat16"}
    policy_name = precision_map.get(precision, "float32")
    policy = keras.mixed_precision.Policy(policy_name)
    keras.mixed_precision.set_global_policy(policy)

    # Seeds
    keras.utils.set_random_seed(cfg.seed)

    # Device
    from src.frameworks.keras.device import setup_device

    setup_device(
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else [],
        memory_limit=cfg.device.memory_limit if hasattr(cfg.device, "memory_limit") else 0.9,
    )


def _build_eval_dataloaders(dataset_provider, cfg, mode="val"):
    """Build Keras-native dataloaders for evaluation.

    Isomorphic with ``_build_eval_dataloaders`` in PyTorch and JAX runners.
    """
    from src.frameworks.keras.dataloaders import (
        ImpressionIterator,
        NewsBatchDataloader,
        UserHistoryBatchDataloader,
    )

    pn = dataset_provider.processed_news
    data = (
        dataset_provider.val_behaviors_data
        if mode == "val"
        else dataset_provider.test_behaviors_data
    )
    batch_size = cfg.eval.batch_size

    news_dl = NewsBatchDataloader(
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
        news_entity_indices=pn.get("entity_indices"),
    )

    user_dl = UserHistoryBatchDataloader(
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
        history_entity_indices=data.get("history_news_entities"),
    )

    imp_iter = ImpressionIterator(
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

    return news_dl, user_dl, imp_iter


def run(cfg: DictConfig):
    """Run training with Keras framework."""
    _setup(cfg)

    import hydra as _hydra
    import keras

    from src.frameworks.keras.dataloaders import create_train_dataloader
    from src.frameworks.keras.evaluation import fast_evaluate
    from src.frameworks.keras.training import training_loop

    from src.core.losses import get_loss
    from src.frameworks.keras.utils import LightweightNewsMetrics, create_news_metrics

    setup_wandb_session(cfg)

    # Dataset (same as JAX/PyTorch)
    dataset_provider = _hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news = dataset_provider.processed_news

    # Model from spec (same as JAX/PyTorch)
    spec = cfg.spec
    # LSTUR needs num_users for user ID embeddings (auto-computed by dataset)
    extra_kwargs = {}
    if spec.model.name.lower() == "lstur":
        extra_kwargs["num_users"] = processed_news["num_users"]
        console.log(f"Auto-detected num_users: {processed_news['num_users']}")
    model = build_model_from_spec(spec, "keras", processed_news, **extra_kwargs)
    console.log(f"Model {spec.model.name} instantiated for Keras.")

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    loss_fn = get_loss(
        loss_name=spec.training.loss.name,
        framework="keras",
        from_logits=spec.training.loss.get("from_logits", True),
        reduction=spec.training.loss.get("reduction", "sum_over_batch_size"),
        label_smoothing=spec.training.loss.get("label_smoothing", 0.0),
    )
    training_metrics = (
        LightweightNewsMetrics.create_training_metrics()
        if LightweightNewsMetrics.should_use_lightweight_metrics(cfg)
        else create_news_metrics(
            NewsRecommenderMetrics(**cfg.metrics.params if hasattr(cfg.metrics, "params") else {})
        )
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=training_metrics)

    # Build train dataloader (isomorphic with PyTorch/JAX)
    features, labels = _build_train_features(dataset_provider)
    train_dataloader = create_train_dataloader(
        features=features,
        labels=labels,
        batch_size=cfg.train.batch_size,
        model_name=spec.model.name.lower(),
    )

    # Metrics engine for evaluation
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Output directory
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation function (isomorphic with JAX/PyTorch)
    def eval_fn(model, mode="val"):
        news_dl, user_dl, imp_iter = _build_eval_dataloaders(
            dataset_provider, cfg, mode=mode
        )
        with Progress(transient=True) as progress:
            return fast_evaluate(
                news_encoder=model.news_encoder,
                user_encoder=model.user_encoder,
                user_hist_dataloader=user_dl,
                news_dataloader=news_dl,
                impression_iterator=imp_iter,
                metrics_calculator=metrics_engine,
                progress=progress,
                process_user_id=getattr(model, "process_user_id", False),
                int_to_news_id_map=dataset_provider.get_int_to_news_id_map(),
            )

    # Test function
    def test_fn(model, best_model_path):
        if best_model_path.exists():
            model.load_weights(best_model_path)
        else:
            console.log("[yellow]Best weights not found, using current weights.[/yellow]")
        # Load test data (not loaded during mode="train" init)
        if not dataset_provider.test_behaviors_data:
            dataset_provider._load_data("test")
        return eval_fn(model, mode="test")

    # Train
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        best_epoch_metrics, test_metrics = training_loop(
            model=model,
            train_dataset=train_dataloader,
            eval_fn=eval_fn,
            test_fn=test_fn,
            cfg=cfg,
            metrics_engine=metrics_engine,
            progress=progress,
            output_directory=output_run_dir,
        )

    return test_metrics or best_epoch_metrics or {}
