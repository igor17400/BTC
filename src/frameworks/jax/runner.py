"""JAX/Flax NNX framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for JAX training,
keeping train.py as a thin dispatcher.

Some models require some especial logic, for instance DIGAT. Thus,
the logic should be adapted to work with them.
"""

import json
import random
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import nnx
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from safetensors.numpy import load_file

from src.core.io.logging import (
    console,
    log_test_results,
    log_training_complete,
    setup_wandb_session,
)
from src.core.io.progress import create_progress
from src.core.io.saving import get_output_run_dir, save_run_summary_fn
from src.core.losses import get_loss
from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.models.spec import build_model_from_spec
from src.core.setup import setup_model
from src.frameworks.jax.dataloaders import (
    ImpressionIterator,
    NewsBatchDataloader,
    UserHistoryBatchDataloader,
    create_glory_jax_dataloader,
    create_train_dataloader,
)
from src.frameworks.jax.evaluation import get_evaluator
from src.frameworks.jax.models.adapter import JAXAdapter
from src.frameworks.jax.training import training_loop


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
        features["hist_ctr"] = np.minimum(np.ceil(ctr * 200).astype(np.int32), 199)
        features["cand_ctr"] = np.asarray(data["candidate_news_ctr"])
    if "candidate_news_recency" in data:
        features["cand_recency"] = np.asarray(data["candidate_news_recency"])

    labels = np.asarray(data["labels"])
    return features, labels


def _build_eval_dataloaders(dataset_provider, cfg, mode="val"):
    """Build JAX-native dataloaders for evaluation."""
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
        news_entity_indices=pn.get("entity_indices"),
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
        history_entity_indices=data.get("history_news_entities"),
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

    start_time = time.time()
    console.log("[bold]Initializing JAX/Flax NNX training...[/bold]")

    # Create output directory early so wandb saves inside it.
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    setup_wandb_session(cfg, output_dir=output_run_dir)

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Dataset
    dataset_provider = hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news = dataset_provider.processed_news

    # Model
    spec = cfg.spec
    # LSTUR needs num_users for user ID embeddings (auto-computed by dataset)
    extra_kwargs = {"rngs": nnx.Rngs(cfg.seed)}
    if spec.model.name.lower() == "lstur":
        extra_kwargs["num_users"] = processed_news["num_users"]
        console.log(f"Auto-detected num_users: {processed_news['num_users']}")
    model = build_model_from_spec(spec, "jax", processed_news, **extra_kwargs)
    console.log(f"Model {spec.model.name} instantiated for JAX.")

    # Load pre-trained weights if provided (eval-only mode).
    _weights_path = cfg.get("_weights_path", None) or cfg.get("weights", None)
    if _weights_path:
        _load_safetensors(model, str(_weights_path))

    # Model-specific setup (DIGAT, GLORY) or standard pipeline
    model_setup = setup_model(spec, dataset_provider, processed_news)

    if model_setup is not None:
        features = model_setup.features
        labels = model_setup.labels

        # Build dataloader — GLORY uses its own DataLoader
        if model_setup.train_dataset is not None:
            train_dataloader = create_glory_jax_dataloader(
                dataset=model_setup.train_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=0,
            )
        else:
            train_dataloader = create_train_dataloader(
                features=features,
                labels=labels,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                seed=cfg.seed,
            )

        # Build eval_fn from setup hook
        metrics_engine = NewsRecommenderMetrics(
            **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
        )
        _adapter = JAXAdapter()
        eval_fn = model_setup.make_eval_fn(
            model,
            _adapter,
            metrics_engine,
            dataset_provider,
            processed_news,
            cfg.eval.batch_size,
            output_run_dir,
            build_eval_dataloaders=_build_eval_dataloaders,
        )
    else:
        # Standard pipeline (NRMS, NAML, LSTUR, MINER, PP-REC, CROWN)
        features, labels = _build_train_features(dataset_provider)
        train_dataloader = create_train_dataloader(
            features=features,
            labels=labels,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            seed=cfg.seed,
        )
        metrics_engine = NewsRecommenderMetrics(
            **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
        )
        evaluate = get_evaluator(spec)

        def eval_fn(model, mode="val", epoch=None, **kwargs):
            news_dl, user_dl, imp_iter = _build_eval_dataloaders(
                dataset_provider, cfg, mode=mode
            )
            behaviors_data = (
                dataset_provider.val_behaviors_data
                if mode == "val"
                else dataset_provider.test_behaviors_data
            )
            with create_progress(transient=True) as progress:
                return evaluate(
                    model=model,
                    news_dataloader=news_dl,
                    user_hist_dataloader=user_dl,
                    impression_iterator=imp_iter,
                    behaviors_data=behaviors_data,
                    metrics_calculator=metrics_engine,
                    progress=progress,
                    int_to_news_id_map=dataset_provider.get_int_to_news_id_map(),
                    save_predictions_path=str(output_run_dir / "predictions"),
                    epoch=epoch,
                    mode=mode,
                )

    # Loss function
    loss_fn = get_loss(
        loss_name=spec.training.loss.name,
        framework="jax",
        from_logits=spec.training.loss.get("from_logits", True),
        label_smoothing=spec.training.loss.get("label_smoothing", 0.0),
    )

    get_aux_loss = (
        (lambda m: m.get_auxiliary_loss())
        if hasattr(model, "get_auxiliary_loss")
        else None
    )

    # Train
    best_metrics = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        gradient_clip_norm=cfg.train.get("gradient_clip_val", 0.0),
        early_stopping_patience=cfg.train.early_stopping.patience,
        early_stopping_min_improvement=cfg.train.early_stopping.get(
            "min_improvement", 0.01
        ),
        warmup_ratio=cfg.train.get("warmup_ratio", 0.0),
        loss_fn=loss_fn,
        get_aux_loss=get_aux_loss,
        use_jit=True,
        eval_fn=eval_fn if cfg.eval.fast_evaluation else None,
        enable_wandb=cfg.logging.enable_wandb,
        save_dir=str(output_run_dir / "models"),
    )

    # Test evaluation
    test_metrics = None
    if cfg.eval.run_test_after_training:
        console.log("[bold]Running test evaluation...[/bold]")
        if not dataset_provider.test_behaviors_data:
            dataset_provider._load_data("test")

        # Rebuild ID remap for models that need it (DIGAT, GLORY)
        if model_setup is not None and model_setup.rebuild_test_remap is not None:
            model_setup.rebuild_test_remap(dataset_provider, processed_news)

        test_metrics = eval_fn(model, mode="test")
        log_test_results(test_metrics)

    log_training_complete(cfg.model_name, "jax", time.time() - start_time)

    # Save eval results (always — both train and eval-only modes).
    if test_metrics:
        eval_path = output_run_dir / "test_results.json"
        with open(eval_path, "w") as f:
            json.dump(
                {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in test_metrics.items()
                },
                f,
                indent=2,
            )
        console.log(f"Saved eval results to {eval_path}")

    # Save run summary (training mode only).
    if not cfg.get("_eval_only", False):
        save_run_summary_fn(
            summary_output_dir=output_run_dir,
            hydra_cfg=cfg,
            initial_metrics_dict={},
            best_metrics_summary_dict=best_metrics,
            test_metrics_dict=test_metrics,
        )

    if wandb.run:
        wandb.finish()

    return test_metrics or best_metrics


def _load_safetensors(model, weights_path: str):
    """Load safetensors weights into a JAX/Flax NNX model.

    Supports local paths and HuggingFace Hub paths (hf://org/repo/file).
    """
    if weights_path.startswith("hf://"):
        # Parse hf://org/repo/filename
        parts = weights_path[5:].split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2] if len(parts) > 2 else "model.safetensors"
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        console.log(f"Downloaded weights from HuggingFace Hub: {repo_id}/{filename}")

    weights = load_file(weights_path)
    state = nnx.state(model)
    new_leaves = []
    restored = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        name = ".".join(str(k) for k in path)
        if name in weights:
            new_leaves.append(jnp.asarray(weights[name]))
            restored += 1
        else:
            new_leaves.append(leaf)
    new_state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(state),
        new_leaves,
    )
    nnx.update(model, new_state)
    console.log(f"Loaded {restored}/{len(weights)} weights from {weights_path}")


def evaluate(cfg: DictConfig, *, weights_path: str, mode: str = "test") -> dict:
    """Load weights and run evaluation without training.

    Reuses ``run()`` for setup, skips training, loads weights, runs eval.
    """

    cfg_copy = OmegaConf.to_container(cfg, resolve=True)
    cfg_copy["train"]["num_epochs"] = 0
    cfg_copy["eval"]["run_test_after_training"] = True
    cfg_copy["eval"]["fast_evaluation"] = False
    cfg_copy["_weights_path"] = weights_path
    cfg_copy["_eval_mode"] = mode

    return run(OmegaConf.create(cfg_copy))
