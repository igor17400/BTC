"""PyTorch framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for PyTorch training,
keeping train.py as a thin dispatcher.
"""

import json
import random
import time

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from safetensors.torch import load_file as load_safetensors

from src.core.io.logging import (
    console,
    log_test_results,
    log_training_complete,
    setup_wandb_session,
)
from src.core.io.progress import create_progress
from src.core.io.saving import get_output_run_dir, save_run_summary_fn
from src.core.io.timing import (
    PhaseStats,
    RunTiming,
    count_params,
    dump_run_timing,
)
from src.core.losses import get_loss
from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.models.spec import build_model_from_spec
from src.core.setup import setup_model
from src.frameworks.pytorch.dataloaders import (
    ImpressionIterator,
    NewsBatchDataloader,
    UserHistoryBatchDataloader,
    create_glory_train_dataloader,
    create_train_dataloader,
)
from src.frameworks.pytorch.evaluation import get_evaluator
from src.frameworks.pytorch.models.adapter import PyTorchAdapter
from src.frameworks.pytorch.training import training_loop


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
    start_time = time.time()
    console.log("[bold]Initializing PyTorch training...[/bold]")

    # Create output directory early so wandb saves inside it.
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    setup_wandb_session(cfg, output_dir=output_run_dir)

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

    # Model-specific setup (DIGAT, GLORY) or standard pipeline
    model_setup = setup_model(spec, dataset_provider, processed_news)

    if model_setup is not None:
        features = model_setup.features
        labels = model_setup.labels

        # Build dataloader — GLORY uses its own DataLoader
        if model_setup.train_dataset is not None:
            train_dataloader = create_glory_train_dataloader(
                dataset=model_setup.train_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        else:
            train_dataloader = create_train_dataloader(
                features=features,
                labels=labels,
                batch_size=cfg.train.batch_size,
                shuffle=True,
            )

        # Build eval_fn from setup hook
        metrics_engine = NewsRecommenderMetrics(
            **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
        )
        _adapter = PyTorchAdapter()
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
        )

        # Metrics
        metrics_engine = NewsRecommenderMetrics(
            **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
        )

        # Evaluation function
        evaluate = get_evaluator(spec)

        int_to_news_id_map = (
            dataset_provider.get_int_to_news_id_map()
            if hasattr(dataset_provider, "get_int_to_news_id_map")
            else None
        )

        def eval_fn(model, mode="val", epoch=None):
            provider = _build_eval_dataloaders(dataset_provider, cfg, mode=mode)
            behaviors_data = (
                dataset_provider.val_behaviors_data
                if mode == "val"
                else dataset_provider.test_behaviors_data
            )
            with create_progress(transient=True) as progress:
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
                    epoch=epoch,
                    save_predictions_path=str(output_run_dir / "predictions"),
                )

    # Loss function from config
    loss_fn = get_loss(
        loss_name=spec.training.loss.name,
        framework="pytorch",
        from_logits=spec.training.loss.get("from_logits", True),
        label_smoothing=spec.training.loss.get("label_smoothing", 0.0),
    )

    # Train
    get_aux_loss = (
        (lambda m: m.get_auxiliary_loss())
        if hasattr(model, "get_auxiliary_loss")
        else None
    )

    best_metrics = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        eval_fn=eval_fn if cfg.eval.fast_evaluation else None,
        cfg=cfg,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        early_stopping_patience=cfg.train.early_stopping.patience,
        early_stopping_min_improvement=cfg.train.early_stopping.get(
            "min_improvement", 0.01
        ),
        enable_wandb=cfg.logging.enable_wandb,
        save_dir=str(output_run_dir / "models"),
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else None,
        loss_fn=loss_fn,
        get_aux_loss=get_aux_loss,
    )

    # Test evaluation
    test_metrics = None
    if cfg.eval.run_test_after_training:
        # Load best checkpoint (safetensors — HF canonical, matches JAX).
        ckpt_path = output_run_dir / "models" / "model.safetensors"
        if ckpt_path.exists():
            model.load_state_dict(load_safetensors(str(ckpt_path)))

        # Load test data (not loaded during mode="train" init)
        if not dataset_provider.test_behaviors_data:
            dataset_provider._load_data("test")

        # Rebuild ID remap for models that need it (DIGAT, GLORY)
        if model_setup is not None and model_setup.rebuild_test_remap is not None:
            model_setup.rebuild_test_remap(dataset_provider, processed_news)

        _test_start = time.time()
        test_metrics = eval_fn(model, mode="test")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _test_wall = time.time() - _test_start
        if test_metrics:
            log_test_results(test_metrics)
        _n_test_imps = (
            int(test_metrics.get("_num_impressions", 0)) if test_metrics else 0
        )
        _test_phase = PhaseStats(
            wall_seconds=_test_wall,
            n_samples=_n_test_imps or None,
            throughput_samples_per_s=(
                _n_test_imps / _test_wall if (_n_test_imps and _test_wall > 0) else None
            ),
        )
        # Stash on best_metrics["timing"] alongside train/val phase stats.
        if "timing" in best_metrics and isinstance(best_metrics["timing"], dict):
            best_metrics["timing"]["test"] = _test_phase.to_dict()

    log_training_complete(cfg.model_name, "pytorch", time.time() - start_time)

    # ---- Dump timing.json + push to W&B ----
    _total_seconds = time.time() - start_time
    _timing = best_metrics.get("timing", {}) if isinstance(best_metrics, dict) else {}
    _dataset_name = (
        cfg.dataset.name
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "name")
        else "unknown"
    )
    _params = count_params(model, "pytorch")
    _run_timing = RunTiming(
        framework="pytorch",
        model_name=cfg.model_name,
        dataset_name=_dataset_name,
        seed=int(cfg.get("seed", 0)),
        n_params_total=int(_params.get("total", 0)),
        n_params_trainable=int(_params.get("trainable", 0)),
        time_to_first_step_seconds=_timing.get("time_to_first_step_seconds"),
        total_seconds=_total_seconds,
    )
    for k in ("train_epochs", "val_epochs"):
        for p in _timing.get(k, []):
            getattr(_run_timing, k).append(PhaseStats(**p))
    if _timing.get("test"):
        _run_timing.test = PhaseStats(**_timing["test"])
    try:
        dump_run_timing(_run_timing, output_run_dir / "timing.json")
        console.log(f"Saved timing to {output_run_dir / 'timing.json'}")
    except Exception as _e:
        console.log(f"[yellow]timing.json dump failed: {_e}[/yellow]")

    if wandb.run is not None:
        _wb: dict[str, float | int] = {
            "timing/n_params_total": _run_timing.n_params_total,
            "timing/n_params_trainable": _run_timing.n_params_trainable,
            "timing/total_seconds": _total_seconds,
        }
        if _timing.get("time_to_first_step_seconds") is not None:
            _wb["timing/time_to_first_step_seconds"] = _timing[
                "time_to_first_step_seconds"
            ]
        if _run_timing.test is not None:
            _wb["timing/test_wall_seconds"] = _run_timing.test.wall_seconds
        wandb.log(_wb)

    # Save eval results.
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

    # Save run summary.
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
