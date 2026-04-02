"""PyTorch framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for PyTorch training,
keeping train.py as a thin dispatcher.
"""

import hydra
import numpy as np
from omegaconf import DictConfig

from src.core.io.logging import console
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
        features["hist_abstract_tokens"] = np.asarray(data["history_news_abstract_tokens"])
        features["cand_abstract_tokens"] = np.asarray(data["candidate_news_abstract_tokens"])
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


def run(cfg: DictConfig):
    """Run training with PyTorch framework."""
    from src.frameworks.pytorch.dataloaders import create_train_dataloader
    from src.frameworks.pytorch.training import training_loop

    console.log("[bold]Initializing PyTorch training...[/bold]")

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

    # Train
    training_loop(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        val_dataset_provider=dataset_provider,
        metrics_engine=metrics_engine,
        epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        patience=cfg.train.early_stopping.patience,
        checkpoint_dir=str(output_run_dir / "models"),
        use_wandb=cfg.logging.enable_wandb,
        gpu_ids=",".join(str(g) for g in cfg.device.gpu_ids) if hasattr(cfg.device, "gpu_ids") else "",
        int_to_news_id_map=dataset_provider.get_int_to_news_id_map()
        if hasattr(dataset_provider, "get_int_to_news_id_map") else None,
    )

    console.log(f"--- {cfg.model_name} PyTorch Training Run Finished ---")
