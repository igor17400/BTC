"""JAX/Flax NNX framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for JAX training,
keeping train.py as a thin dispatcher.
"""

import hydra
import numpy as np
from flax import nnx
from omegaconf import DictConfig

from src.core.io.logging import console
from src.core.io.saving import get_output_run_dir
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


def run(cfg: DictConfig):
    """Run training with JAX/Flax NNX framework."""
    from src.frameworks.jax.dataloaders import TrainingBatchIterator
    from src.frameworks.jax.training import training_loop

    console.log("[bold]Initializing JAX/Flax NNX training...[/bold]")

    # Dataset
    dataset_provider = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news = dataset_provider.processed_news

    # Model
    spec = cfg.spec
    model = build_model_from_spec(spec, "jax", processed_news, rngs=nnx.Rngs(cfg.seed))
    console.log(f"Model {spec.model.name} instantiated for JAX.")

    # Train iterator
    features, labels = _build_train_features(dataset_provider)
    train_iterator = TrainingBatchIterator(
        features=features,
        labels=labels,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )

    # Output
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Train
    training_loop(
        model=model,
        train_iterator=train_iterator,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        early_stopping_patience=cfg.train.early_stopping.patience,
        enable_wandb=cfg.logging.enable_wandb,
        save_dir=str(output_run_dir / "models"),
    )

    console.log(f"--- {cfg.model_name} JAX Training Run Finished ---")
