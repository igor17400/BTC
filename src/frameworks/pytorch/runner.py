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
    glory_cache: dict | None = None
    if spec.model.name.lower() == "glory":
        from src.core.data.processing.digat import build_id_remap
        from src.core.data.processing.glory import (
            build_news_feature_matrix,
            build_news_graph,
            build_neighbor_dict,
        )
        from src.core.models.configs import GLORYConfig
        from src.frameworks.pytorch.dataloaders import (
            GLORYTrainDataset,
            create_glory_train_dataloader,
        )

        g_cfg = GLORYConfig(
            title_size=spec.inputs.title.max_length,
            entity_size=spec.inputs.get("entity", {}).get("max_length", 5),
            max_history_length=spec.inputs.history.max_length,
            max_impressions_length=spec.inputs.impressions.max_length,
            head_num=spec.model.architecture.news_encoder.head_num,
            head_dim=spec.model.architecture.news_encoder.head_dim,
            attention_hidden_dim=spec.model.architecture.news_encoder.attention_hidden_dim,
            gnn_num_layers=spec.model.architecture.graph_encoder.gnn_num_layers,
            use_graph_type=spec.model.architecture.graph_encoder.use_graph_type,
            directed=spec.model.architecture.graph_encoder.directed,
            k_hops=spec.model.architecture.graph_encoder.k_hops,
            num_neighbors=spec.model.architecture.graph_encoder.num_neighbors,
            dropout_rate=spec.model.dropout_rate,
        )

        # Pack per-news features (title | entity | cat | sub | idx).
        news_features = build_news_feature_matrix(
            processed_news, title_size=g_cfg.title_size, entity_size=g_cfg.entity_size,
        )

        # Build / cache the global news graph from training click trajectories.
        num_news = news_features.shape[0]

        # Behaviors store news IDs parsed from "N####" strings, which
        # live in a different integer space than ``news_features`` row
        # indices (assigned by ``news_str_id_to_int_idx``).  Reuse
        # DIGAT's remap — it matches by token sequence — so every
        # downstream op (graph construction, subgraph sampling, feature
        # lookup) operates in the same news-feature space.
        glory_remap_path = (
            dataset_provider.dataset_path / "processed" / "glory_id_remap.npy"
            if hasattr(dataset_provider, "dataset_path")
            else None
        )
        glory_id_remap = build_id_remap(
            dataset_provider, processed_news, glory_remap_path,
        )

        def _apply_remap(ids: np.ndarray) -> np.ndarray:
            if glory_id_remap is None:
                return np.clip(ids, 0, num_news - 1)
            return np.where(
                ids < len(glory_id_remap), glory_id_remap[ids], 0,
            ).astype(np.int64)

        def _remap_behaviors_in_place(split: str) -> None:
            """Translate behaviors-space ids → feature-space for one split."""
            data = getattr(dataset_provider, f"{split}_behaviors_data", None)
            if not data:
                return
            data = dict(data)
            if "histories_news_ids" in data:
                data["histories_news_ids"] = _apply_remap(
                    np.asarray(data["histories_news_ids"]).astype(np.int64)
                )
            if "candidate_news_ids" in data:
                raw = data["candidate_news_ids"]
                # Python list / object array → almost always ragged
                # (variable-length candidates per impression, val/test).
                # Avoid ``np.asarray(raw)`` — it raises on ragged input.
                if isinstance(raw, list) or (
                    isinstance(raw, np.ndarray) and raw.dtype.kind == "O"
                ):
                    if len(raw) == 0:
                        pass
                    else:
                        sample = np.asarray(raw[0])
                        if sample.dtype.kind in ("i", "u"):
                            data["candidate_news_ids"] = [
                                _apply_remap(np.asarray(c).astype(np.int64))
                                for c in raw
                            ]
                        # else: string IDs — leave as-is (synthetic).
                else:
                    raw_arr = np.asarray(raw)
                    if raw_arr.dtype.kind in ("U", "S"):
                        # Literal string IDs (synthetic homogeneous) —
                        # downstream code handles via string→int map.
                        pass
                    else:
                        # Homogeneous (N, C) int array — train path.
                        data["candidate_news_ids"] = _apply_remap(
                            raw_arr.astype(np.int64),
                        )
            setattr(dataset_provider, f"{split}_behaviors_data", data)

        # Apply remap to train + val up-front; test is lazy-loaded and
        # gets remapped just before the test-eval call.
        _remap_behaviors_in_place("train")
        _remap_behaviors_in_place("val")
        tb = dataset_provider.train_behaviors_data
        graph_cache_dir = (
            dataset_provider.dataset_path / "processed"
            if hasattr(dataset_provider, "dataset_path")
            else None
        )
        graph_path = (
            graph_cache_dir / f"glory_news_graph_type{g_cfg.use_graph_type}.pkl"
            if graph_cache_dir is not None else None
        )
        neighbor_path = (
            graph_cache_dir
            / f"glory_neighbor_dict_type{g_cfg.use_graph_type}_dir{int(g_cfg.directed)}.pkl"
            if graph_cache_dir is not None else None
        )
        glory_graph = build_news_graph(
            dataset_provider.train_behaviors_data,
            num_news=num_news,
            use_graph_type=g_cfg.use_graph_type,
            cache_path=graph_path,
        )
        glory_neighbors = build_neighbor_dict(
            glory_graph, directed=g_cfg.directed, cache_path=neighbor_path,
        )

        # Hold for eval reuse.
        glory_cache = {
            "cfg": g_cfg,
            "news_features": news_features,
            "graph": glory_graph,
            "neighbors": glory_neighbors,
        }

        # Training dataloader (custom collate for variable-size subgraphs).
        tb = dataset_provider.train_behaviors_data
        n_samples = len(tb["labels"])

        # `histories_news_ids` is populated for MIND; synthetic datasets
        # only carry tokens.  Fall back to zero-filled ids so the
        # subgraph sampler degenerates cleanly (history → empty graph).
        if "histories_news_ids" in tb:
            hist_ids = np.asarray(tb["histories_news_ids"]).astype(np.int64)
        else:
            hist_ids = np.zeros(
                (n_samples, g_cfg.max_history_length), dtype=np.int64,
            )

        raw_cand = tb["candidate_news_ids"]
        raw_cand_arr = np.asarray(raw_cand)
        if raw_cand_arr.dtype.kind in ("U", "S", "O"):
            # String IDs (e.g. synthetic emits "N42") — map via the
            # dataset's news_str_id_to_int_idx if available; else zero.
            str_to_int = getattr(dataset_provider, "news_str_id_to_int_idx", None)
            if str_to_int is not None:
                vfunc = np.vectorize(lambda x: str_to_int.get(str(x), 0))
                cand_ids = vfunc(raw_cand_arr).astype(np.int64)
            else:
                cand_ids = np.zeros(raw_cand_arr.shape, dtype=np.int64)
        else:
            # Already feature-space: `_remap_behaviors_in_place("train")`
            # above translated the int candidate IDs.  Re-applying
            # _apply_remap here would double-remap and destroy training.
            cand_ids = raw_cand_arr.astype(np.int64)

        labels = np.asarray(tb["labels"])
        glory_train_ds = GLORYTrainDataset(
            hist_ids=hist_ids,
            cand_ids=cand_ids,
            news_features=news_features,
            graph_edge_index=glory_graph["edge_index"],
            graph_edge_attr=glory_graph["edge_attr"],
            neighbor_dict=glory_neighbors,
            labels=labels,
            his_size=g_cfg.max_history_length,
            k_hops=g_cfg.k_hops,
            num_neighbors=g_cfg.num_neighbors,
        )
        train_dataloader = create_glory_train_dataloader(
            dataset=glory_train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            # Per-sample subgraph sampling dominates single-threaded
            # wall time; run it across worker processes to overlap with
            # GPU compute.
            num_workers=4,
            pin_memory=True,
        )
        # GLORY's dataloader is fully assembled; skip the generic path below.
        features, labels = None, None  # unused
    elif spec.model.name.lower() == "digat":
        from src.core.data.processing.digat import build_digat_train_features
        from src.core.models.configs import DIGATConfig

        sag_config = spec.model.architecture.graph_encoder
        digat_cfg = DIGATConfig(
            sag_hops=sag_config.get("sag_hops", 2),
            sag_neighbors=sag_config.get("sag_neighbors", 5),
            max_title_length=spec.inputs.title.max_length,
            max_history_length=spec.inputs.history.max_length,
            max_impressions_length=spec.inputs.impressions.max_length,
        )
        if hasattr(dataset_provider, "dataset_path"):
            from src.core.data.processing.news import read_all_news
            from src.core.data.processing.sag import construct_sag

            all_news_df = read_all_news(dataset_provider.dataset_path)
            id_map = dataset_provider.news_str_id_to_int_idx
            cache_dir = dataset_provider.dataset_path / "processed"
            sag_data = construct_sag(
                all_news_df, id_map,
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
        from src.core.data.processing.digat import build_id_remap
        remap_path = dataset_provider.dataset_path / "processed" / "behaviors_to_sag_remap.npy" if hasattr(dataset_provider, "dataset_path") else None
        id_remap = build_id_remap(dataset_provider, processed_news, remap_path)

        features, labels = build_digat_train_features(
            dataset_provider.train_behaviors_data,
            processed_news, sag_data,
            max_history=digat_cfg.max_history_length,
            max_impressions=digat_cfg.max_impressions_length,
            num_categories=num_categories,
            news_graph_size=digat_cfg.news_graph_size,
            max_title_length=digat_cfg.max_title_length,
            id_remap=id_remap,
        )
    else:
        features, labels = _build_train_features(dataset_provider)

    # GLORY builds its own DataLoader above; everyone else uses the generic path.
    if spec.model.name.lower() != "glory":
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

    if spec.model.name.lower() == "digat":
        from src.core.models.evaluations.digat import digat_evaluate
        from src.frameworks.pytorch.models.adapter import PyTorchAdapter

        _adapter = PyTorchAdapter()

        def eval_fn(model, mode="val"):
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
    elif spec.model.name.lower() == "glory":
        from src.core.models.evaluations.glory import glory_evaluate
        from src.frameworks.pytorch.models.adapter import PyTorchAdapter

        _adapter = PyTorchAdapter()

        def eval_fn(model, mode="val"):
            return glory_evaluate(
                news_encoder=model.local_news_encoder,
                graph_encoder=model.global_news_encoder,
                click_encoder=model.click_encoder,
                user_encoder=model.user_encoder,
                candidate_encoder=model.candidate_encoder,
                click_predictor=model.click_predictor,
                dataset_provider=dataset_provider,
                processed_news=processed_news,
                news_features=glory_cache["news_features"],
                graph=glory_cache["graph"],
                neighbor_dict=glory_cache["neighbors"],
                adapter=_adapter,
                metrics_calculator=metrics_engine,
                his_size=glory_cache["cfg"].max_history_length,
                mode=mode,
                batch_size=cfg.eval.batch_size,
            )
    else:
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

        # DIGAT: rebuild id_remap now that test data is loaded. It was built
        # at startup before test behaviors existed, so test-exclusive news
        # IDs were mapped to 0 (padding), silently corrupting test metrics.
        if spec.model.name.lower() == "digat":
            if remap_path is not None and remap_path.exists():
                remap_path.unlink()
            id_remap = build_id_remap(dataset_provider, processed_news, remap_path)

        # GLORY: rebuild id_remap AND remap the newly-loaded test split
        # into feature-space (train/val were remapped at init).
        if spec.model.name.lower() == "glory":
            if glory_remap_path is not None and glory_remap_path.exists():
                glory_remap_path.unlink()
            glory_id_remap = build_id_remap(
                dataset_provider, processed_news, glory_remap_path,
            )
            _remap_behaviors_in_place("test")

        test_metrics = eval_fn(model, mode="test")
        if test_metrics:
            log_test_results(test_metrics)

    log_training_complete(cfg.model_name, "pytorch", time.time() - start_time)

    if wandb.run:
        wandb.finish()

    return test_metrics or best_metrics
