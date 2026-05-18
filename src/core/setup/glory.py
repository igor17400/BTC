"""GLORY-specific setup: graph construction, ID remap, and eval closure."""

from __future__ import annotations

import numpy as np

from src.core.data.processing.models.digat import build_id_remap
from src.core.data.processing.models.glory import (
    build_csr_in_adjacency,
    build_entity_graph,
    build_entity_neighbor_dict,
    build_neighbor_dict,
    build_news_feature_matrix,
    build_news_graph,
)
from src.core.models.configs import GLORYConfig
from src.core.models.evaluations.custom.glory import glory_evaluate
from src.frameworks.pytorch.dataloaders import GLORYTrainDataset

from .types import ModelSetupResult


def setup_glory(spec, dataset_provider, processed_news) -> ModelSetupResult:
    """Prepare GLORY training data, graphs, and eval closure factory.

    This is the framework-agnostic core of GLORY setup, shared by JAX,
    PyTorch, and Keras runners.
    """
    use_entity = spec.model.get("use_entity", False)
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
        use_entity=use_entity,
        entity_emb_dim=spec.model.get("entity_emb_dim", 100),
        entity_neighbors=spec.model.architecture.graph_encoder.get(
            "entity_neighbors", 10
        ),
    )

    # Build news feature matrix
    news_features = build_news_feature_matrix(
        processed_news,
        g_cfg.title_size,
        g_cfg.entity_size,
    )
    num_news = news_features.shape[0]

    # Build ID remap
    glory_remap_path = (
        dataset_provider.dataset_path / "processed" / "glory_id_remap.npy"
        if hasattr(dataset_provider, "dataset_path")
        else None
    )
    glory_id_remap = build_id_remap(
        dataset_provider,
        processed_news,
        glory_remap_path,
    )

    def _apply_remap(ids):
        if glory_id_remap is None:
            return np.clip(ids, 0, num_news - 1)
        safe_ids = np.clip(ids, 0, len(glory_id_remap) - 1)
        return np.where(
            ids < len(glory_id_remap),
            glory_id_remap[safe_ids],
            0,
        ).astype(np.int64)

    # Remap train behaviors in-place
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

    # Build graphs
    graph_cache_dir = (
        dataset_provider.dataset_path / "processed"
        if hasattr(dataset_provider, "dataset_path")
        else None
    )
    graph_path = (
        graph_cache_dir / f"glory_news_graph_type{g_cfg.use_graph_type}.pkl"
        if graph_cache_dir
        else None
    )
    neighbor_path = (
        graph_cache_dir
        / f"glory_neighbor_dict_type{g_cfg.use_graph_type}_dir{int(g_cfg.directed)}.pkl"
        if graph_cache_dir
        else None
    )
    glory_graph = build_news_graph(
        dataset_provider.train_behaviors_data,
        num_news=num_news,
        use_graph_type=g_cfg.use_graph_type,
        cache_path=graph_path,
    )
    glory_neighbors = build_neighbor_dict(
        glory_graph,
        directed=g_cfg.directed,
        cache_path=neighbor_path,
    )

    # Entity graph + neighbors (optional)
    entity_neighbor_dict = None
    if use_entity:
        entity_graph_path = (
            graph_cache_dir / "glory_entity_graph.pkl" if graph_cache_dir else None
        )
        entity_neighbor_path = (
            graph_cache_dir / "glory_entity_neighbor_dict.pkl"
            if graph_cache_dir
            else None
        )
        entity_graph = build_entity_graph(
            glory_graph,
            news_features,
            title_size=g_cfg.title_size,
            entity_size=g_cfg.entity_size,
            cache_path=entity_graph_path,
        )
        entity_neighbor_dict = build_entity_neighbor_dict(
            entity_graph,
            cache_path=entity_neighbor_path,
        )

    # Build GLORYTrainDataset
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
        entity_neighbor_dict=entity_neighbor_dict,
        entity_size=g_cfg.entity_size,
        entity_neighbors=g_cfg.entity_neighbors,
        title_size=g_cfg.title_size,
    )

    # GLORY eval mode: "precomputed" (default; one global GNN pass on
    # the full 65k-node graph) vs "per_impression" (k-hop subgraph per
    # impression, matching the reference ValidGraphDataset).
    eval_mode = str(
        spec.model.architecture.graph_encoder.get("eval_mode", "precomputed")
    ).lower()

    # Padded-subgraph mode (for JAX). When set, every per-impression
    # subgraph is padded/truncated to a fixed (N, E) shape so XLA
    # compiles the GNN forward once. ``None`` (default) keeps the
    # variable-size path that PyTorch uses.
    max_subgraph_nodes = spec.model.architecture.graph_encoder.get(
        "max_subgraph_nodes", None
    )
    max_subgraph_edges = spec.model.architecture.graph_encoder.get(
        "max_subgraph_edges", None
    )

    # CSR adjacency is only needed when we sample subgraphs at eval time.
    # Built lazily to skip the cost when eval_mode="precomputed".
    csr_in_adjacency = None
    if eval_mode == "per_impression":
        csr_in_adjacency = build_csr_in_adjacency(
            np.asarray(glory_graph["edge_index"]),
            int(glory_graph["num_nodes"]),
        )

    # Mutable context for eval_fn and remap rebuild
    ctx = {
        "cfg": g_cfg,
        "news_features": news_features,
        "graph": glory_graph,
        "neighbors": glory_neighbors,
        "entity_neighbor_dict": entity_neighbor_dict,
        "id_remap": glory_id_remap,
        "remap_path": glory_remap_path,
        "use_entity": use_entity,
        "eval_mode": eval_mode,
        "csr_in_adjacency": csr_in_adjacency,
        "max_subgraph_nodes": max_subgraph_nodes,
        "max_subgraph_edges": max_subgraph_edges,
    }

    def make_eval_fn(
        model,
        adapter,
        metrics_engine,
        dataset_provider,
        processed_news,
        eval_batch_size,
        output_run_dir,
        **_,
    ):
        def eval_fn(model, mode="val", epoch=None, **kwargs):
            return glory_evaluate(
                news_encoder=model.local_news_encoder,
                graph_encoder=model.global_news_encoder,
                click_encoder=model.click_encoder,
                user_encoder=model.user_encoder,
                candidate_encoder=model.candidate_encoder,
                click_predictor=getattr(model, "click_predictor", None),
                dataset_provider=dataset_provider,
                processed_news=processed_news,
                news_features=ctx["news_features"],
                graph=ctx["graph"],
                neighbor_dict=ctx["neighbors"],
                adapter=adapter,
                metrics_calculator=metrics_engine,
                his_size=ctx["cfg"].max_history_length,
                mode=mode,
                batch_size=eval_batch_size,
                id_remap=ctx["id_remap"],
                use_entity=ctx["use_entity"],
                entity_encoder=getattr(model, "local_entity_encoder", None),
                global_entity_encoder=getattr(model, "global_entity_encoder", None),
                entity_embedding=getattr(model, "entity_embedding", None),
                entity_neighbor_dict=ctx.get("entity_neighbor_dict"),
                entity_size=ctx["cfg"].entity_size,
                entity_neighbors=ctx["cfg"].entity_neighbors,
                title_size=ctx["cfg"].title_size,
                save_predictions_path=str(output_run_dir / "predictions"),
                epoch=epoch,
                eval_mode=ctx["eval_mode"],
                k_hops=ctx["cfg"].k_hops,
                num_neighbors=ctx["cfg"].num_neighbors,
                csr_in_adjacency=ctx["csr_in_adjacency"],
                max_subgraph_nodes=ctx["max_subgraph_nodes"],
                max_subgraph_edges=ctx["max_subgraph_edges"],
            )

        return eval_fn

    def rebuild_test_remap(dataset_provider, processed_news):
        if ctx["remap_path"] is not None and ctx["remap_path"].exists():
            ctx["remap_path"].unlink()
        # Hide remapped train data to avoid ID namespace collisions
        _saved_train = dataset_provider.train_behaviors_data
        dataset_provider.train_behaviors_data = {}
        ctx["id_remap"] = build_id_remap(
            dataset_provider,
            processed_news,
            ctx["remap_path"],
        )
        dataset_provider.train_behaviors_data = _saved_train

    return ModelSetupResult(
        features=None,
        labels=None,
        train_dataset=glory_train_ds,
        eval_context=ctx,
        make_eval_fn=make_eval_fn,
        rebuild_test_remap=rebuild_test_remap,
    )
