"""PyTorch framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for PyTorch training,
keeping train.py as a thin dispatcher.
"""

import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.core.io.logging import console, log_training_complete
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


def _build_eval_provider(dataset_provider, cfg, mode="val"):
    """Build a dict of PyTorch-native eval dataloaders."""
    from src.frameworks.pytorch.dataloaders import (
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
        news_abstract_tokens=pn.get(
            "abstract_tokens", np.zeros((len(pn["tokens"]), 1))
        ),
        news_category_indices=pn.get("category_indices", np.zeros(len(pn["tokens"]))),
        news_subcategory_indices=pn.get(
            "subcategory_indices", np.zeros(len(pn["tokens"]))
        ),
        batch_size=batch_size,
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    user_dl = UserHistoryBatchDataloader(
        history_tokens=data["history_news_tokens"],
        history_abstract_tokens=data.get(
            "history_news_abstract_tokens", np.zeros((len(data["labels"]), 1))
        ),
        history_category=data.get(
            "history_news_categories", np.zeros((len(data["labels"]), 1))
        ),
        history_subcategory=data.get(
            "history_news_subcategories", np.zeros((len(data["labels"]), 1))
        ),
        impression_ids=data["impression_ids"],
        user_ids=data.get("user_ids"),
        batch_size=batch_size,
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    imp_iter = ImpressionIterator(
        impression_tokens=data["candidate_news_tokens"],
        impression_abstract_tokens=data.get(
            "candidate_news_abstract_tokens", data["candidate_news_tokens"]
        ),
        impression_category=data.get(
            "candidate_news_categories", np.zeros((len(data["labels"]), 1))
        ),
        impression_subcategory=data.get(
            "candidate_news_subcategories", np.zeros((len(data["labels"]), 1))
        ),
        labels=data["labels"],
        impression_ids=data["impression_ids"],
        candidate_ids=data["candidate_news_ids"],
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    return {
        "user_hist_dataloader": user_dl,
        "news_dataloader": news_dl,
        "impression_iterator": imp_iter,
    }


def run(cfg: DictConfig):
    """Run training with PyTorch framework."""
    from src.frameworks.pytorch.dataloaders import create_train_dataloader
    from src.frameworks.pytorch.training import training_loop

    start_time = time.time()
    console.log("[bold]Initializing PyTorch training...[/bold]")

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Dataset
    dataset_provider = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news = dataset_provider.processed_news

    # Model
    spec = cfg.spec
    model = build_model_from_spec(spec, "pytorch", processed_news)
    console.log(f"Model {spec.model.name} instantiated for PyTorch.")

    # Train dataloader
    features, labels = _build_train_features(dataset_provider)
    train_dataloader = create_train_dataloader(
        features=features,
        labels=labels,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )

    # Metrics
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Output
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Build eval provider with PyTorch-native dataloaders (no Keras dependency)
    val_provider = _build_eval_provider(dataset_provider, cfg, mode="val")

    # Train
    result = training_loop(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        val_dataset_provider=val_provider,
        metrics_engine=metrics_engine,
        epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        patience=cfg.train.early_stopping.patience,
        checkpoint_dir=str(output_run_dir / "models"),
        use_wandb=cfg.logging.enable_wandb,
        gpu_ids=",".join(str(g) for g in cfg.device.gpu_ids)
        if hasattr(cfg.device, "gpu_ids")
        else "",
        int_to_news_id_map=dataset_provider.get_int_to_news_id_map()
        if hasattr(dataset_provider, "get_int_to_news_id_map")
        else None,
    )

    log_training_complete(cfg.model_name, "pytorch", time.time() - start_time)
    return result.get("best_metrics", result) if isinstance(result, dict) else {}
