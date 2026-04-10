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
from rich.progress import Progress

import wandb
from src.core.io.logging import (
    console,
    log_test_results,
    log_training_complete,
    setup_wandb_session,
)
from src.core.io.saving import get_output_run_dir
from src.core.losses import get_loss
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

    # PP-Rec: concatenate entity and category into hist_tokens/cand_tokens
    if "history_news_entities" in data and dataset_provider.process_entities:
        hist_ent = np.asarray(data["history_news_entities"])
        cand_ent = np.asarray(data["candidate_news_entities"])
        if "hist_tokens" in features:
            features["hist_tokens"] = np.concatenate(
                [features["hist_tokens"], hist_ent], axis=-1
            )
            features["cand_tokens"] = np.concatenate(
                [features["cand_tokens"], cand_ent], axis=-1
            )
    if dataset_provider.process_category and getattr(
        dataset_provider, "process_entities", False
    ):
        hist_cat = np.asarray(data["history_news_categories"])
        cand_cat = np.asarray(data["candidate_news_categories"])
        if "hist_tokens" in features:
            features["hist_tokens"] = np.concatenate(
                [features["hist_tokens"], np.expand_dims(hist_cat, axis=-1)],
                axis=-1,
            )
            features["cand_tokens"] = np.concatenate(
                [features["cand_tokens"], np.expand_dims(cand_cat, axis=-1)],
                axis=-1,
            )

    # PP-Rec CTR features
    if "history_news_ctr" in data:
        ctr = np.asarray(data["history_news_ctr"])
        # Discretize history CTR for embedding lookup: ceil(ctr * 200), capped at 199
        features["hist_ctr"] = np.minimum(np.ceil(ctr * 200).astype(np.int32), 199)
        features["cand_ctr"] = np.asarray(data["candidate_news_ctr"])
    if "candidate_news_recency" in data:
        features["cand_recency"] = np.asarray(data["candidate_news_recency"])

    labels = np.asarray(data["labels"])
    return features, labels


def _build_eval_dataloaders(dataset_provider, cfg, mode="val"):
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
        news_entity_indices=pn.get("entity_indices"),
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
        history_entity_indices=data.get("history_news_entities"),
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
    setup_wandb_session(cfg)

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
    # LSTUR needs num_users for user ID embeddings (auto-computed by dataset)
    extra_kwargs = {}
    if spec.model.name.lower() == "lstur":
        extra_kwargs["num_users"] = processed_news["num_users"]
        console.log(f"Auto-detected num_users: {processed_news['num_users']}")
    model = build_model_from_spec(spec, "pytorch", processed_news, **extra_kwargs)
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

    # Evaluation function (isomorphic with Keras/JAX)
    from src.frameworks.pytorch.evaluation import get_evaluator

    evaluate = get_evaluator(spec)

    int_to_news_id_map = (
        dataset_provider.get_int_to_news_id_map()
        if hasattr(dataset_provider, "get_int_to_news_id_map")
        else None
    )

    def eval_fn(model, mode="val"):
        provider = _build_eval_dataloaders(dataset_provider, cfg, mode=mode)
        behaviors_data = (
            dataset_provider.val_behaviors_data
            if mode == "val"
            else dataset_provider.test_behaviors_data
        )
        with Progress(transient=True) as progress:
            return evaluate(
                model=model,
                news_dataloader=provider["news_dataloader"],
                user_hist_dataloader=provider["user_hist_dataloader"],
                impression_iterator=provider["impression_iterator"],
                behaviors_data=behaviors_data,
                metrics_calculator=metrics_engine,
                progress=progress,
                int_to_news_id_map=int_to_news_id_map,
                mode=mode,
            )

    # Loss function from config
    loss_fn = get_loss(
        loss_name=spec.training.loss.name,
        framework="pytorch",
        from_logits=spec.training.loss.get("from_logits", True),
        label_smoothing=spec.training.loss.get("label_smoothing", 0.0),
    )

    # Train
    best_metrics = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        eval_fn=eval_fn if cfg.eval.fast_evaluation else None,
        cfg=cfg,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        early_stopping_patience=cfg.train.early_stopping.patience,
        enable_wandb=cfg.logging.enable_wandb,
        save_dir=str(output_run_dir / "models"),
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else None,
        loss_fn=loss_fn,
    )

    # Test evaluation
    test_metrics = None
    if cfg.eval.run_test_after_training:
        # Load best checkpoint if available
        ckpt_path = output_run_dir / "models" / "best_model.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Load test data (not loaded during mode="train" init)
        if not dataset_provider.test_behaviors_data:
            dataset_provider._load_data("test")

        test_metrics = eval_fn(model, mode="test")
        if test_metrics:
            log_test_results(test_metrics)

    log_training_complete(cfg.model_name, "pytorch", time.time() - start_time)

    if wandb.run:
        wandb.finish()

    return test_metrics or best_metrics
