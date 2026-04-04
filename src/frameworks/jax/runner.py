"""JAX/Flax NNX framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for JAX training,
keeping train.py as a thin dispatcher.
"""

import random
import time

import hydra
import numpy as np
import wandb
from flax import nnx
from omegaconf import DictConfig
from rich.progress import Progress

from src.core.io.logging import console, log_test_results, log_training_complete, setup_wandb_session
from src.core.io.saving import get_output_run_dir
from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.models.spec import build_model_from_spec


def _build_train_features(dataset_provider) -> tuple:
    """Extract raw numpy features and labels from the dataset provider."""
    data = dataset_provider.train_behaviors_data
    features = {}

    if dataset_provider.process_title:
        features["hist_tokens"] = np.asarray(data["history_news_tokens"])
        features["cand_tokens"] = np.asarray(data["candidate_news_tokens"])
    if dataset_provider.process_abstract:
        features["hist_abstract_tokens"] = np.asarray(
            data["history_news_abstract_tokens"]
        )
        features["cand_abstract_tokens"] = np.asarray(
            data["candidate_news_abstract_tokens"]
        )
    if dataset_provider.process_category:
        features["hist_category"] = np.asarray(data["history_news_categories"])
        features["cand_category"] = np.asarray(data["candidate_news_categories"])
    if dataset_provider.process_subcategory:
        features["hist_subcategory"] = np.asarray(data["history_news_subcategories"])
        features["cand_subcategory"] = np.asarray(data["candidate_news_subcategories"])
    if dataset_provider.process_user_id:
        features["user_ids"] = np.asarray(data["user_ids"])

    labels = np.asarray(data["labels"])
    return features, labels


def _build_eval_dataloaders(dataset_provider, cfg, mode="val"):
    """Build JAX-native dataloaders for evaluation."""
    from src.frameworks.jax.dataloaders import (
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

    news_dl = NewsBatchDataloader(
        news_ids=np.array(pn["news_ids_original_strings"]),
        news_tokens=pn["tokens"],
        news_abstract_tokens=pn.get("abstract_tokens"),
        news_category_indices=pn.get("category_indices"),
        news_subcategory_indices=pn.get("subcategory_indices"),
        batch_size=cfg.eval.batch_size,
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    user_dl = UserHistoryBatchDataloader(
        history_tokens=data["history_news_tokens"],
        impression_ids=data["impression_ids"],
        history_abstract_tokens=data.get("history_news_abstract_tokens"),
        history_category=data.get("history_news_categories"),
        history_subcategory=data.get("history_news_subcategories"),
        user_ids=data.get("user_ids"),
        batch_size=cfg.eval.batch_size,
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    imp_iter = ImpressionIterator(
        impression_tokens=data["candidate_news_tokens"],
        labels=data["labels"],
        impression_ids=data["impression_ids"],
        candidate_ids=data["candidate_news_ids"],
        impression_abstract_tokens=data.get("candidate_news_abstract_tokens"),
        impression_category=data.get("candidate_news_categories"),
        impression_subcategory=data.get("candidate_news_subcategories"),
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    return news_dl, user_dl, imp_iter


def run(cfg: DictConfig):
    """Run training with JAX/Flax NNX framework."""
    from src.frameworks.jax.dataloaders import create_train_dataloader
    from src.frameworks.jax.evaluation import fast_evaluate
    from src.frameworks.jax.training import training_loop

    start_time = time.time()
    console.log("[bold]Initializing JAX/Flax NNX training...[/bold]")
    setup_wandb_session(cfg)

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Dataset
    dataset_provider = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news = dataset_provider.processed_news

    # Model
    spec = cfg.spec
    # LSTUR needs num_users for user ID embeddings
    extra_kwargs = {"rngs": nnx.Rngs(cfg.seed)}
    if spec.model.name.lower() == "lstur":
        extra_kwargs["num_users"] = spec.model.get("num_users", int(np.max(dataset_provider.train_behaviors_data["user_ids"])) + 1)
    model = build_model_from_spec(spec, "jax", processed_news, **extra_kwargs)
    console.log(f"Model {spec.model.name} instantiated for JAX.")

    # Train dataloader (isomorphic with Keras/PyTorch)
    features, labels = _build_train_features(dataset_provider)
    train_dataloader = create_train_dataloader(
        features=features,
        labels=labels,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )

    # Metrics
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Output
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation function (called at end of each epoch)
    def eval_fn(model, **kwargs):
        news_dl, user_dl, imp_iter = _build_eval_dataloaders(
            dataset_provider, cfg, mode="val"
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

    # Train
    best_metrics = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        early_stopping_patience=cfg.train.early_stopping.patience,
        eval_fn=eval_fn if cfg.eval.fast_evaluation else None,
        enable_wandb=cfg.logging.enable_wandb,
        save_dir=str(output_run_dir / "models"),
    )

    # Test evaluation
    test_metrics = None
    if cfg.eval.run_test_after_training:
        console.log("[bold]Running test evaluation...[/bold]")
        # Load test data (not loaded during mode="train" init)
        if not dataset_provider.test_behaviors_data:
            dataset_provider._load_data("test")
        news_dl, user_dl, imp_iter = _build_eval_dataloaders(
            dataset_provider, cfg, mode="test"
        )
        with Progress(transient=True) as progress:
            test_metrics = fast_evaluate(
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
        log_test_results(test_metrics)

    log_training_complete(cfg.model_name, "jax", time.time() - start_time)

    if wandb.run:
        wandb.finish()

    return test_metrics or best_metrics
