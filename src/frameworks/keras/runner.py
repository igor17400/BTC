"""Keras framework runner for NewsReX.

Provides ``run(cfg)`` as the single entry point for Keras training.
Same pattern as JAX/PyTorch: dataset → model → build dataloaders → train.
"""

import os

import numpy as np
from omegaconf import DictConfig

from src.core.io.logging import console, setup_wandb_session
from src.core.io.progress import create_progress
from src.core.io.saving import get_output_run_dir
from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.models.spec import build_model_from_spec

SUPPORTED_BACKENDS = ("jax", "torch")


def _build_train_features(dataset_provider) -> tuple:
    """Extract raw numpy features and labels from the dataset provider."""
    import keras

    data = dataset_provider.train_behaviors_data
    features = {}

    if dataset_provider.process_title:
        features["hist_tokens"] = keras.ops.convert_to_numpy(
            data["history_news_tokens"]
        )
        features["cand_tokens"] = keras.ops.convert_to_numpy(
            data["candidate_news_tokens"]
        )
    if dataset_provider.process_abstract:
        features["hist_abstract_tokens"] = keras.ops.convert_to_numpy(
            data["history_news_abstract_tokens"]
        )
        features["cand_abstract_tokens"] = keras.ops.convert_to_numpy(
            data["candidate_news_abstract_tokens"]
        )
    if dataset_provider.process_category:
        features["hist_category"] = keras.ops.convert_to_numpy(
            data["history_news_categories"]
        )
        features["cand_category"] = keras.ops.convert_to_numpy(
            data["candidate_news_categories"]
        )
    if dataset_provider.process_subcategory:
        features["hist_subcategory"] = keras.ops.convert_to_numpy(
            data["history_news_subcategories"]
        )
        features["cand_subcategory"] = keras.ops.convert_to_numpy(
            data["candidate_news_subcategories"]
        )
    if dataset_provider.process_user_id:
        features["user_ids"] = keras.ops.convert_to_numpy(data["user_ids"])

    # PP-Rec: concatenate entity and category into hist_tokens/cand_tokens
    if "history_news_entities" in data and dataset_provider.process_entities:
        hist_ent = keras.ops.convert_to_numpy(data["history_news_entities"])
        cand_ent = keras.ops.convert_to_numpy(data["candidate_news_entities"])
        # Append entity indices to token features: (B, H, title_len + max_entities)
        if "hist_tokens" in features:
            features["hist_tokens"] = np.concatenate(
                [features["hist_tokens"], hist_ent], axis=-1
            )
            features["cand_tokens"] = np.concatenate(
                [features["cand_tokens"], cand_ent], axis=-1
            )
    if dataset_provider.process_category and dataset_provider.process_entities:
        # Append category index as last feature
        hist_cat = keras.ops.convert_to_numpy(data["history_news_categories"])
        cand_cat = keras.ops.convert_to_numpy(data["candidate_news_categories"])
        if "hist_tokens" in features:
            features["hist_tokens"] = np.concatenate(
                [features["hist_tokens"], np.expand_dims(hist_cat, axis=-1)], axis=-1
            )
            features["cand_tokens"] = np.concatenate(
                [features["cand_tokens"], np.expand_dims(cand_cat, axis=-1)], axis=-1
            )

    # PP-Rec CTR features
    if "history_news_ctr" in data:
        ctr = keras.ops.convert_to_numpy(data["history_news_ctr"])
        # Discretize history CTR for embedding lookup: ceil(ctr * 200), capped at 199
        features["hist_ctr"] = np.minimum(np.ceil(ctr * 200).astype(np.int32), 199)
        features["cand_ctr"] = keras.ops.convert_to_numpy(data["candidate_news_ctr"])
    if "candidate_news_recency" in data:
        features["cand_recency"] = keras.ops.convert_to_numpy(
            data["candidate_news_recency"]
        )

    labels = keras.ops.convert_to_numpy(data["labels"])
    return features, labels


def _setup(cfg: DictConfig):
    """Setup Keras backend and precision."""
    backend = getattr(cfg.device, "keras_backend", "jax")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported Keras backend: '{backend}'. Use one of: {SUPPORTED_BACKENDS}"
        )

    current = os.environ.get("KERAS_BACKEND")
    if current and current != backend:
        raise RuntimeError(
            f"Cannot switch Keras backend from '{current}' to '{backend}' in the same process."
        )
    os.environ["KERAS_BACKEND"] = backend
    import keras

    console.log(f"Keras backend: {keras.backend.backend()}")

    # Precision
    precision = getattr(cfg.device, "precision", "float32")
    precision_map = {
        "float32": "float32",
        "float16": "mixed_float16",
        "bfloat16": "mixed_bfloat16",
    }
    policy_name = precision_map.get(precision, "float32")
    policy = keras.mixed_precision.Policy(policy_name)
    keras.mixed_precision.set_global_policy(policy)

    # Seeds
    keras.utils.set_random_seed(cfg.seed)

    # Device
    from src.frameworks.keras.device import setup_device

    setup_device(
        gpu_ids=cfg.device.gpu_ids if hasattr(cfg.device, "gpu_ids") else [],
        memory_limit=cfg.device.memory_limit
        if hasattr(cfg.device, "memory_limit")
        else 0.9,
    )


def _build_eval_dataloaders(dataset_provider, cfg, mode="val"):
    """Build Keras-native dataloaders for evaluation.

    Isomorphic with ``_build_eval_dataloaders`` in PyTorch and JAX runners.
    """
    from src.frameworks.keras.dataloaders import (
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
        news_abstract_tokens=pn["abstract_tokens"],
        news_category_indices=pn["category_indices"],
        news_subcategory_indices=pn["subcategory_indices"],
        batch_size=batch_size,
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
        news_entity_indices=pn.get("entity_indices"),
    )

    user_dl = UserHistoryBatchDataloader(
        history_tokens=data["history_news_tokens"],
        history_abstract_tokens=data["history_news_abstract_tokens"],
        history_category=data["history_news_categories"],
        history_subcategory=data["history_news_subcategories"],
        impression_ids=data["impression_ids"],
        user_ids=data["user_ids"],
        batch_size=batch_size,
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
        history_entity_indices=data.get("history_news_entities"),
    )

    imp_iter = ImpressionIterator(
        impression_tokens=data["candidate_news_tokens"],
        impression_abstract_tokens=data["candidate_news_abstract_tokens"],
        impression_category=data["candidate_news_categories"],
        impression_subcategory=data["candidate_news_subcategories"],
        labels=data["labels"],
        impression_ids=data["impression_ids"],
        candidate_ids=data["candidate_news_ids"],
        process_title=dataset_provider.process_title,
        process_abstract=dataset_provider.process_abstract,
        process_category=dataset_provider.process_category,
        process_subcategory=dataset_provider.process_subcategory,
    )

    return news_dl, user_dl, imp_iter


def run(cfg: DictConfig):
    """Run training with Keras framework."""
    _setup(cfg)

    import hydra as _hydra
    import keras

    from src.core.losses import get_loss
    from src.frameworks.keras.dataloaders import create_train_dataloader
    from src.frameworks.keras.evaluation import get_evaluator
    from src.frameworks.keras.training import training_loop
    from src.frameworks.keras.utils import LightweightNewsMetrics, create_news_metrics

    setup_wandb_session(cfg)

    # Dataset (same as JAX/PyTorch)
    dataset_provider = _hydra.utils.instantiate(cfg.dataset, mode="train")
    processed_news = dataset_provider.processed_news

    # Model from spec (same as JAX/PyTorch)
    spec = cfg.spec
    # LSTUR needs num_users for user ID embeddings (auto-computed by dataset)
    extra_kwargs = {}
    if spec.model.name.lower() == "lstur":
        extra_kwargs["num_users"] = processed_news["num_users"]
        console.log(f"Auto-detected num_users: {processed_news['num_users']}")
    model = build_model_from_spec(spec, "keras", processed_news, **extra_kwargs)
    console.log(f"Model {spec.model.name} instantiated for Keras.")

    # Compile
    clip_val = cfg.train.get("gradient_clip_val", 0.0)
    optimizer = keras.optimizers.Adam(
        learning_rate=cfg.train.learning_rate,
        global_clipnorm=clip_val if clip_val > 0 else None,
    )
    loss_fn = get_loss(
        loss_name=spec.training.loss.name,
        framework="keras",
        from_logits=spec.training.loss.get("from_logits", True),
        reduction=spec.training.loss.get("reduction", "sum_over_batch_size"),
        label_smoothing=spec.training.loss.get("label_smoothing", 0.0),
    )
    training_metrics = (
        LightweightNewsMetrics.create_training_metrics()
        if LightweightNewsMetrics.should_use_lightweight_metrics(cfg)
        else create_news_metrics(
            NewsRecommenderMetrics(
                **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
            )
        )
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=training_metrics)

    # Build train features and dataloader — DIGAT and GLORY have custom
    # pipelines; standard models use the generic path.
    sag_data = None
    id_remap = None
    remap_path = None
    glory_cache = None
    glory_id_remap = None
    glory_remap_path = None

    if spec.model.name.lower() == "digat":
        from src.core.data.processing.models.digat import (
            build_digat_train_features,
            build_id_remap,
        )
        from src.core.data.processing.models.sag import construct_sag
        from src.core.data.processing.text.news import read_all_news
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
        remap_path = (
            dataset_provider.dataset_path / "processed" / "behaviors_to_sag_remap.npy"
            if hasattr(dataset_provider, "dataset_path")
            else None
        )
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
    elif spec.model.name.lower() == "glory":
        from src.core.data.processing.models.digat import build_id_remap
        from src.core.data.processing.models.glory import (
            build_neighbor_dict,
            build_news_feature_matrix,
            build_news_graph,
        )
        from src.core.models.configs import GLORYConfig
        from src.frameworks.pytorch.dataloaders import (
            GLORYTrainDataset,
            glory_collate,
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
            use_graph_type=spec.model.architecture.graph_encoder.get("use_graph_type", 0),
            directed=spec.model.architecture.graph_encoder.get("directed", True),
            k_hops=spec.model.architecture.graph_encoder.get("k_hops", 2),
            num_neighbors=spec.model.architecture.graph_encoder.get("num_neighbors", 8),
            dropout_rate=spec.model.dropout_rate,
        )
        news_features = build_news_feature_matrix(
            processed_news, g_cfg.title_size, g_cfg.entity_size,
        )
        num_news = news_features.shape[0]
        glory_remap_path = (
            dataset_provider.dataset_path / "processed" / "glory_id_remap.npy"
            if hasattr(dataset_provider, "dataset_path")
            else None
        )
        glory_id_remap = build_id_remap(
            dataset_provider, processed_news, glory_remap_path,
        )

        def _apply_remap(ids):
            if glory_id_remap is None:
                return np.clip(ids, 0, num_news - 1)
            return np.where(
                ids < len(glory_id_remap), glory_id_remap[ids], 0,
            ).astype(np.int64)

        # Remap train in-place for graph builder + dataloader.
        tb = dict(dataset_provider.train_behaviors_data)
        if "histories_news_ids" in tb:
            tb["histories_news_ids"] = _apply_remap(
                np.asarray(tb["histories_news_ids"]).astype(np.int64)
            )
        if "candidate_news_ids" in tb:
            raw = tb["candidate_news_ids"]
            raw_arr = np.asarray(raw)
            if raw_arr.dtype.kind not in ("U", "S"):
                tb["candidate_news_ids"] = _apply_remap(raw_arr.astype(np.int64))
        dataset_provider.train_behaviors_data = tb

        graph_cache_dir = (
            dataset_provider.dataset_path / "processed"
            if hasattr(dataset_provider, "dataset_path")
            else None
        )
        graph_path = (
            graph_cache_dir / f"glory_news_graph_type{g_cfg.use_graph_type}.pkl"
            if graph_cache_dir else None
        )
        neighbor_path = (
            graph_cache_dir / f"glory_neighbor_dict_type{g_cfg.use_graph_type}_dir{int(g_cfg.directed)}.pkl"
            if graph_cache_dir else None
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
        glory_cache = {
            "cfg": g_cfg,
            "news_features": news_features,
            "graph": glory_graph,
            "neighbors": glory_neighbors,
        }

        # Build GLORY train dataset (reuses PyTorch's numpy-based dataset).
        n_samples = len(tb["labels"])
        if "histories_news_ids" in tb:
            hist_ids = np.asarray(tb["histories_news_ids"]).astype(np.int64)
        else:
            hist_ids = np.zeros((n_samples, g_cfg.max_history_length), dtype=np.int64)
        raw_cand = tb["candidate_news_ids"]
        raw_cand_arr = np.asarray(raw_cand)
        if raw_cand_arr.dtype.kind in ("U", "S", "O"):
            str_to_int = getattr(dataset_provider, "news_str_id_to_int_idx", None)
            if str_to_int is not None:
                vfunc = np.vectorize(lambda x: str_to_int.get(str(x), 0))
                cand_ids = vfunc(raw_cand_arr).astype(np.int64)
            else:
                cand_ids = np.zeros(raw_cand_arr.shape, dtype=np.int64)
        else:
            cand_ids = raw_cand_arr.astype(np.int64)

        glory_train_ds = GLORYTrainDataset(
            hist_ids=hist_ids,
            cand_ids=cand_ids,
            news_features=news_features,
            graph_edge_index=glory_graph["edge_index"],
            graph_edge_attr=glory_graph["edge_attr"],
            neighbor_dict=glory_neighbors,
            labels=np.asarray(tb["labels"]),
            his_size=g_cfg.max_history_length,
            k_hops=g_cfg.k_hops,
            num_neighbors=g_cfg.num_neighbors,
        )
        # Set features/labels to None — GLORY uses its own dataloader.
        features, labels = None, None
    else:
        features, labels = _build_train_features(dataset_provider)

    # Build train dataloader — GLORY uses PyTorch DataLoader with custom
    # collate; DIGAT and standard models use the Keras Sequence.
    if spec.model.name.lower() == "glory":
        import keras
        if keras.backend.backend() == "jax":
            # JAX backend: Keras Sequence with padded collate.
            # Can't use PyTorch DataLoader — it calls .cpu() on outputs.
            from src.frameworks.jax.dataloaders import _glory_collate_jax

            class _GLORYKerasSequence(keras.utils.Sequence):
                def __init__(self, ds, bs, collate):
                    self._ds, self._bs, self._collate = ds, bs, collate
                    self._indices = np.arange(len(ds))
                def __len__(self):
                    return len(self._ds) // self._bs
                def __getitem__(self, idx):
                    start = idx * self._bs
                    batch = [self._ds[int(self._indices[i])]
                             for i in range(start, start + self._bs)]
                    return self._collate(batch)
                def on_epoch_end(self):
                    np.random.shuffle(self._indices)

            train_dataloader = _GLORYKerasSequence(
                glory_train_ds, cfg.train.batch_size, _glory_collate_jax,
            )
        else:
            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(
                glory_train_ds,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=glory_collate,
                drop_last=True,
            )
    else:
        train_dataloader = create_train_dataloader(
            features=features,
            labels=labels,
            batch_size=cfg.train.batch_size,
            model_name=spec.model.name.lower(),
        )

    # Metrics engine for evaluation
    metrics_engine = NewsRecommenderMetrics(
        **cfg.metrics.params if hasattr(cfg.metrics, "params") else {}
    )

    # Output directory
    output_run_dir = get_output_run_dir(cfg)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation function — DIGAT and GLORY use dedicated evaluators;
    # standard models go through the registry.
    if spec.model.name.lower() == "digat":
        from src.core.models.evaluations.digat import digat_evaluate
        from src.frameworks.keras.models.adapter import KerasAdapter

        _adapter = KerasAdapter()

        def eval_fn(model, mode="val"):
            result = digat_evaluate(
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
            # Free eval memory so training can resume without fragmentation.
            import keras
            if keras.backend.backend() == "torch":
                import torch; torch.cuda.empty_cache()
            return result
    elif spec.model.name.lower() == "glory":
        from src.core.models.evaluations.glory import glory_evaluate
        from src.frameworks.keras.models.adapter import KerasAdapter

        _adapter = KerasAdapter()

        def eval_fn(model, mode="val"):
            result = glory_evaluate(
                news_encoder=model.local_news_encoder,
                graph_encoder=model.global_news_encoder,
                click_encoder=model.click_encoder,
                user_encoder=model.user_encoder,
                candidate_encoder=model.candidate_encoder,
                click_predictor=None,
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
                id_remap=glory_id_remap,
            )
            import keras
            if keras.backend.backend() == "torch":
                import torch; torch.cuda.empty_cache()
            return result
    else:
        evaluate = get_evaluator(spec)

        def eval_fn(model, mode="val"):
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
                )

    # Test function
    def test_fn(model, best_model_path):
        if best_model_path.exists():
            model.load_weights(best_model_path)
        else:
            console.log(
                "[yellow]Best weights not found, using current weights.[/yellow]"
            )
        # Load test data (not loaded during mode="train" init)
        if not dataset_provider.test_behaviors_data:
            dataset_provider._load_data("test")

        # Rebuild ID remap now that test data is loaded.
        nonlocal id_remap, glory_id_remap
        if spec.model.name.lower() == "digat":
            from src.core.data.processing.models.digat import (
                build_id_remap as _build_remap,
            )
            if remap_path is not None and remap_path.exists():
                remap_path.unlink()
            id_remap = _build_remap(dataset_provider, processed_news, remap_path)

        if spec.model.name.lower() == "glory":
            if glory_remap_path is not None and glory_remap_path.exists():
                glory_remap_path.unlink()
            _saved_train = dataset_provider.train_behaviors_data
            dataset_provider.train_behaviors_data = {}
            glory_id_remap = build_id_remap(
                dataset_provider, processed_news, glory_remap_path,
            )
            dataset_provider.train_behaviors_data = _saved_train

        return eval_fn(model, mode="test")

    # Train
    with create_progress(console=console) as progress:
        best_epoch_metrics, test_metrics = training_loop(
            model=model,
            train_dataset=train_dataloader,
            eval_fn=eval_fn,
            test_fn=test_fn,
            cfg=cfg,
            metrics_engine=metrics_engine,
            progress=progress,
            output_directory=output_run_dir,
        )

    return test_metrics or best_epoch_metrics or {}
