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

from src.core.data.encoders.plm import attach_plm_embeddings
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


def _build_train_features(dataset_provider, encoder_cfg=None) -> tuple:
    """Extract raw numpy features and labels from the dataset provider.

    The feature layout is uniform regardless of encoder type. Models
    receive parsed-int news ids and delegate the per-news text lookup
    to their injected :class:`TextEncoder` (GloVe word ids or PLM
    token cache, picked at construction time).

    Returns a dict with at least:
      - ``hist_features`` (B, H) — history news ids
      - ``cand_features`` (B, C) — candidate news ids

    For models that consume packed structured features (PP-Rec, CAUM,
    TCCM), also returns:
      - ``hist_tokens`` (B, H, 1+E+1) — ``[news_idx | entities | category]``
      - ``cand_tokens`` (B, C, 1+E+1)

    Auxiliary side-channels (CTR, recency, subcategory, user_ids) are
    populated when the dataset provider exposes them.
    """
    data = dataset_provider.train_behaviors_data
    features: dict[str, np.ndarray] = {}

    hist_news_idx = np.asarray(data["histories_news_ids"])
    cand_news_idx = np.asarray(data["candidate_news_ids"])
    features["hist_features"] = hist_news_idx
    features["cand_features"] = cand_news_idx

    # Packed structured features for entity/category-using models.
    # Column order matches eval dataloaders: [news_idx | entities | category].
    hist_parts: list[np.ndarray] = [hist_news_idx[..., None]]
    cand_parts: list[np.ndarray] = [cand_news_idx[..., None]]

    if getattr(dataset_provider, "process_entities", False) and (
        "history_news_entities" in data
    ):
        hist_parts.append(np.asarray(data["history_news_entities"]))
        cand_parts.append(np.asarray(data["candidate_news_entities"]))

    if dataset_provider.process_category:
        features["hist_category"] = np.asarray(data["history_news_categories"])
        features["cand_category"] = np.asarray(data["candidate_news_categories"])
        hist_parts.append(np.asarray(data["history_news_categories"])[..., None])
        cand_parts.append(np.asarray(data["candidate_news_categories"])[..., None])

    if dataset_provider.process_subcategory:
        features["hist_subcategory"] = np.asarray(data["history_news_subcategories"])
        features["cand_subcategory"] = np.asarray(data["candidate_news_subcategories"])

    if len(hist_parts) > 1:
        features["hist_tokens"] = np.concatenate(hist_parts, axis=-1)
        features["cand_tokens"] = np.concatenate(cand_parts, axis=-1)

    if dataset_provider.process_user_id:
        features["user_ids"] = np.asarray(data["user_ids"])

    # PP-Rec CTR / recency features.
    if "history_news_ctr" in data:
        ctr = np.asarray(data["history_news_ctr"])
        features["hist_ctr"] = np.minimum(np.ceil(ctr * 200).astype(np.int32), 199)
        features["cand_ctr"] = np.asarray(data["candidate_news_ctr"])
    if "candidate_news_recency" in data:
        features["cand_recency"] = np.asarray(data["candidate_news_recency"])

    labels = np.asarray(data["labels"])
    return features, labels


def _build_eval_dataloaders(dataset_provider, cfg, mode="val"):
    """Build a dict of PyTorch-native eval dataloaders.

    Uniform across encoders: every news (history, candidate, news-table)
    is identified by a parsed-int news id. The model's TextEncoder owns
    the per-news text lookup, so the eval pipeline doesn't care whether
    GloVe or PLM is in use.

    Yielded contracts:
      - ``news_dataloader``: ``{"news_id", "news_features"}`` where
        ``news_features`` is ``(B,)`` int when no auxiliary view is
        packed, else ``(B, k)`` int with column order
        ``[news_idx | entities | category | subcategory]``.
      - ``user_hist_dataloader``: ``(imp_ids, user_ids, history)`` where
        ``history`` is ``(B, H)`` or ``(B, H, k)`` int.
      - ``impression_iterator``: per-impression
        ``(features, labels, imp_id, cand_ids)`` with ``features``
        ``(C,)`` or ``(C, k)`` int.
    """
    pn = dataset_provider.processed_news
    data = (
        dataset_provider.val_behaviors_data
        if mode == "val"
        else dataset_provider.test_behaviors_data
    )
    batch_size = cfg.eval.batch_size

    news_ids_str = np.array(pn["news_ids_original_strings"])
    parsed_news_ids = np.array(
        [
            int(s[1:]) if isinstance(s, str) and s.startswith("N") else int(s)
            for s in news_ids_str
        ],
        dtype=np.int64,
    )

    # Multi-view packing: only when the dataset is configured to expose
    # the corresponding side-channel AND the data actually has it. NRMS
    # has all flags False -> news_features stays (B,) int.
    news_cat = (
        np.asarray(pn["category_indices"])
        if dataset_provider.process_category and "category_indices" in pn
        else None
    )
    news_subcat = (
        np.asarray(pn["subcategory_indices"])
        if dataset_provider.process_subcategory and "subcategory_indices" in pn
        else None
    )
    hist_cat = (
        np.asarray(data["history_news_categories"])
        if dataset_provider.process_category and "history_news_categories" in data
        else None
    )
    hist_subcat = (
        np.asarray(data["history_news_subcategories"])
        if dataset_provider.process_subcategory and "history_news_subcategories" in data
        else None
    )
    cand_cat = (
        data["candidate_news_categories"]
        if dataset_provider.process_category and "candidate_news_categories" in data
        else None
    )
    cand_subcat = (
        data["candidate_news_subcategories"]
        if dataset_provider.process_subcategory
        and "candidate_news_subcategories" in data
        else None
    )

    # Entity features (PP-Rec, CAUM, TCCM).
    news_entity = (
        np.asarray(pn["entity_indices"])
        if getattr(dataset_provider, "process_entities", False)
        and "entity_indices" in pn
        else None
    )
    hist_entity = (
        np.asarray(data["history_news_entities"])
        if getattr(dataset_provider, "process_entities", False)
        and "history_news_entities" in data
        else None
    )
    cand_entity = (
        data["candidate_news_entities"]
        if getattr(dataset_provider, "process_entities", False)
        and "candidate_news_entities" in data
        else None
    )

    news_dl = NewsBatchDataloader(
        news_ids_str=news_ids_str,
        parsed_news_ids=parsed_news_ids,
        batch_size=batch_size,
        category_indices=news_cat,
        subcategory_indices=news_subcat,
        entity_indices=news_entity,
    )
    user_dl = UserHistoryBatchDataloader(
        history_news_ids=np.asarray(data["histories_news_ids"], dtype=np.int64),
        impression_ids=data["impression_ids"],
        user_ids=data.get("user_ids"),
        batch_size=batch_size,
        history_category=hist_cat,
        history_subcategory=hist_subcat,
        history_entity=hist_entity,
    )
    # ``candidate_news_ids`` is ragged across impressions (different
    # candidate counts), so pass through as an object array — the
    # iterator casts per-row.
    imp_iter = ImpressionIterator(
        candidate_news_ids=data["candidate_news_ids"],
        labels=data["labels"],
        impression_ids=data["impression_ids"],
        candidate_ids=data["candidate_news_ids"],
        candidate_category=cand_cat,
        candidate_subcategory=cand_subcat,
        candidate_entity=cand_entity,
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

    # Pre-compute and cache frozen PLM token-level embeddings if requested.
    # GloVe encoder skips this step. The encoder is uniform across models:
    # always caches token-level outputs (and the attention mask). Models
    # decide what to do with them (pool / CNN / MHA / multi-view fusion).
    encoder_cfg = getattr(cfg, "encoder", None)
    if encoder_cfg is not None and encoder_cfg.get("type", "glove") != "glove":
        trainability = encoder_cfg.get("trainability", "frozen_cached")
        if trainability != "frozen_cached":
            raise NotImplementedError(
                f"trainability='{trainability}' is not yet implemented (Pass 2). "
                "Use trainability=frozen_cached for now."
            )
        console.log(
            f"Attaching frozen PLM token-level embeddings "
            f"(plm_name={encoder_cfg.plm_name}, max_length={encoder_cfg.max_length}) ..."
        )
        attach_plm_embeddings(
            processed_news,
            plm_name=encoder_cfg.plm_name,
            text_field=encoder_cfg.text_field,
            max_length=encoder_cfg.max_length,
            batch_size=encoder_cfg.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            id_prefix=getattr(dataset_provider, "id_prefix", "N"),
            level="token",
        )
        # Multi-view models (NAML, etc.) also need the abstract cache.
        # Always built when text_field_abstract is set; models that don't
        # consume it (NRMS, etc.) just ignore the extra keys on
        # processed_news.
        text_field_abstract = encoder_cfg.get("text_field_abstract", "") or ""
        if text_field_abstract:
            console.log(
                f"Attaching second frozen PLM cache for abstract view "
                f"(text_field={text_field_abstract}, "
                f"max_length={encoder_cfg.get('max_length_abstract', 50)}) ..."
            )
            attach_plm_embeddings(
                processed_news,
                plm_name=encoder_cfg.plm_name,
                text_field=text_field_abstract,
                max_length=encoder_cfg.get("max_length_abstract", 50),
                batch_size=encoder_cfg.batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                id_prefix=getattr(dataset_provider, "id_prefix", "N"),
                level="token",
                output_prefix="abstract_",
            )

    # Model
    spec = cfg.spec
    # LSTUR needs num_users for user ID embeddings (auto-computed by dataset)
    extra_kwargs = {}
    if spec.model.name.lower() == "lstur":
        extra_kwargs["num_users"] = processed_news["num_users"]
        console.log(f"Auto-detected num_users: {processed_news['num_users']}")
    model = build_model_from_spec(
        spec, "pytorch", processed_news, encoder=encoder_cfg, **extra_kwargs
    )
    console.log(f"Model {spec.model.name} instantiated for PyTorch.")

    # Model-specific setup (DIGAT, GLORY, TCCM) or standard pipeline.
    # ``encoder_cfg`` is threaded so the hook can pack PLM news-idx
    # features instead of GloVe tokens (TCCM, DIGAT).
    model_setup = setup_model(spec, dataset_provider, processed_news, encoder_cfg)

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
        features, labels = _build_train_features(dataset_provider, encoder_cfg)
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

    # When val == test (dev_as_val), early stopping on val would bias
    # the reported test number — disable it and run the full schedule.
    # The "best-val checkpoint" tracker still promotes the best epoch.
    _es_patience = (
        cfg.train.num_epochs + 1
        if getattr(cfg.dataset, "validation_split_strategy", None) == "dev_as_val"
        else cfg.train.early_stopping.patience
    )
    best_metrics = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        eval_fn=eval_fn if cfg.eval.fast_evaluation else None,
        cfg=cfg,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        early_stopping_patience=_es_patience,
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
    _dev_as_val = (
        getattr(cfg.dataset, "validation_split_strategy", None) == "dev_as_val"
    )
    if cfg.eval.run_test_after_training and _dev_as_val:
        # With dev_as_val, the validation and test sets are the same
        # MIND-dev impressions, so the best-val checkpoint's val metrics
        # ARE the test metrics by construction. Promote them to skip a
        # redundant evaluation pass (~50 s/seed).
        console.log(
            "[dim]dev_as_val: promoting best-val metrics to test metrics "
            "(skipping redundant test eval)[/dim]"
        )
        test_metrics = {
            k.replace("val_", "", 1): v
            for k, v in best_metrics.items()
            if k.startswith("val_")
        }
        log_test_results(test_metrics)
        _test_wall = 0.0
    elif cfg.eval.run_test_after_training:
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
