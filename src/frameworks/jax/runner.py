"""JAX/Flax NNX framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for JAX training,
keeping train.py as a thin dispatcher.

Some models require some especial logic, for instance DIGAT. Thus,
the logic should be adapted to work with them.
"""

import random
import time

import hydra
import numpy as np
from flax import nnx
from omegaconf import DictConfig
from rich.progress import Progress

import wandb
from src.core.data.processing.models.digat import (
    build_digat_train_features,
    build_id_remap,
)
from src.core.data.processing.models.sag import construct_sag
from src.core.data.processing.text.news import read_all_news
from src.core.io.logging import (
    console,
    log_test_results,
    log_training_complete,
    setup_wandb_session,
)
from src.core.io.saving import get_output_run_dir
from src.core.losses import get_loss
from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.models.configs import DIGATConfig
from src.core.models.spec import build_model_from_spec
from src.frameworks.jax.dataloaders import create_train_dataloader
from src.frameworks.jax.evaluation import get_evaluator
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
    setup_wandb_session(cfg)

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

    # Train features: DIGAT assembles its own tensors via SAG preprocessing;
    # every other model follows the standard pipeline.
    sag_data = None
    id_remap = None
    remap_path = None
    if spec.model.name.lower() == "digat":
        sag_config = spec.model.architecture.graph_encoder
        digat_cfg = DIGATConfig(
            sag_hops=sag_config.get("sag_hops", 2),
            sag_neighbors=sag_config.get("sag_neighbors", 5),
            max_title_length=spec.inputs.title.max_length,
            max_history_length=spec.inputs.history.max_length,
            max_impressions_length=spec.inputs.impressions.max_length,
        )
        if hasattr(dataset_provider, "dataset_path"):
            all_news_df = read_all_news(dataset_provider.dataset_path)
            id_map = dataset_provider.news_str_id_to_int_idx
            cache_dir = dataset_provider.dataset_path / "processed"
            sag_data = construct_sag(
                all_news_df,
                id_map,
                sag_hops=digat_cfg.sag_hops,
                sag_neighbors=digat_cfg.sag_neighbors,
                cache_dir=cache_dir,
            )
        else:
            num_news = processed_news["tokens"].shape[0]
            G = digat_cfg.news_graph_size
            sag_data = {
                "news_node_ID": np.zeros((num_news, G), dtype=np.int32),
                "news_graph": np.eye(G, dtype=np.bool_)[None].repeat(num_news, axis=0),
                "news_graph_mask": np.ones((num_news, G), dtype=np.bool_),
            }
        num_categories = int(processed_news.get("num_categories", 18)) + 1

        # Build behaviors→SAG ID remap (the two pipelines use different int mappings)
        remap_path = (
            dataset_provider.dataset_path / "processed" / "behaviors_to_sag_remap.npy"
            if hasattr(dataset_provider, "dataset_path")
            else None
        )
        id_remap = build_id_remap(dataset_provider, processed_news, remap_path)

        features, labels = build_digat_train_features(
            dataset_provider.train_behaviors_data,
            processed_news,
            sag_data,
            max_history=digat_cfg.max_history_length,
            max_impressions=digat_cfg.max_impressions_length,
            num_categories=num_categories,
            news_graph_size=digat_cfg.news_graph_size,
            max_title_length=digat_cfg.max_title_length,
            id_remap=id_remap,
        )
    else:
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

    # Evaluation function (called at end of each epoch). DIGAT uses a
    # dedicated evaluator because its dual graph interaction requires
    # co-computing news and user contexts per impression; standard
    # models go through the registry.
    if spec.model.name.lower() == "digat":
        from src.core.models.evaluations.digat import digat_evaluate
        from src.frameworks.jax.models.adapter import JAXAdapter

        _adapter = JAXAdapter()

        def eval_fn(model, mode="val", **kwargs):
            return digat_evaluate(
                news_encoder=model.news_encoder,
                graph_encoder=model.graph_encoder,
                dataset_provider=dataset_provider,
                processed_news=processed_news,
                sag_data=sag_data,
                adapter=_adapter,
                metrics_calculator=metrics_engine,
                D=model.D,
                num_categories=model.num_categories,
                max_history=model.max_history,
                mode=mode,
                batch_size=cfg.eval.batch_size,
                id_remap=id_remap,
            )
    else:
        evaluate = get_evaluator(spec)

        def eval_fn(model, mode="val", **kwargs):
            news_dl, user_dl, imp_iter = _build_eval_dataloaders(
                dataset_provider, cfg, mode=mode
            )
            behaviors_data = (
                dataset_provider.val_behaviors_data
                if mode == "val"
                else dataset_provider.test_behaviors_data
            )
            with Progress(transient=True) as progress:
                return evaluate(
                    model=model,
                    news_dataloader=news_dl,
                    user_hist_dataloader=user_dl,
                    impression_iterator=imp_iter,
                    behaviors_data=behaviors_data,
                    metrics_calculator=metrics_engine,
                    progress=progress,
                    int_to_news_id_map=dataset_provider.get_int_to_news_id_map(),
                )

    # Loss function from config
    loss_fn = get_loss(
        loss_name=spec.training.loss.name,
        framework="jax",
        from_logits=spec.training.loss.get("from_logits", True),
        label_smoothing=spec.training.loss.get("label_smoothing", 0.0),
    )

    # Auxiliary loss (e.g. CROWN category prediction)
    aux_loss_fn = None
    if hasattr(model, "get_auxiliary_loss"):

        def aux_loss_fn(m):
            return m.get_auxiliary_loss()

    # Train
    best_metrics = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        early_stopping_patience=cfg.train.early_stopping.patience,
        loss_fn=loss_fn,
        get_aux_loss=aux_loss_fn,
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

        # test metrics.
        if spec.model.name.lower() == "digat":
            from src.core.data.processing.models.digat import (
                build_id_remap as _build_remap,
            )

            if remap_path is not None and remap_path.exists():
                remap_path.unlink()
            id_remap = _build_remap(dataset_provider, processed_news, remap_path)

        test_metrics = eval_fn(model, mode="test")
        log_test_results(test_metrics)

    log_training_complete(cfg.model_name, "jax", time.time() - start_time)

    if wandb.run:
        wandb.finish()

    return test_metrics or best_metrics
