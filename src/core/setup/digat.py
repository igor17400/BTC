"""DIGAT-specific setup: SAG construction, ID remap, and eval closure."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.data.processing.models.digat import (
    build_digat_train_features,
    build_id_remap,
)
from src.core.data.processing.models.sag import construct_sag
from src.core.data.processing.text.news import read_all_news
from src.core.models.configs import DIGATConfig
from src.core.models.evaluations.custom.digat import digat_evaluate

from .types import ModelSetupResult


def setup_digat(spec, dataset_provider, processed_news) -> ModelSetupResult:
    """Prepare DIGAT training data, SAG graphs, and eval closure factory.

    This is the framework-agnostic core of DIGAT setup, shared by JAX,
    PyTorch, and Keras runners.
    """
    sag_config = spec.model.architecture.graph_encoder
    digat_cfg = DIGATConfig(
        sag_hops=sag_config.get("sag_hops", 2),
        sag_neighbors=sag_config.get("sag_neighbors", 5),
        max_title_length=spec.inputs.title.max_length,
        max_history_length=spec.inputs.history.max_length,
        max_impressions_length=spec.inputs.impressions.max_length,
    )

    # Build SAG data
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

    # Build ID remap
    remap_path = (
        dataset_provider.dataset_path / "processed" / "behaviors_to_sag_remap.npy"
        if hasattr(dataset_provider, "dataset_path")
        else None
    )
    id_remap = build_id_remap(dataset_provider, processed_news, remap_path)

    # Build train features
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

    # Mutable context — eval_fn reads from here, rebuild_test_remap writes to it
    ctx = {
        "sag_data": sag_data,
        "id_remap": id_remap,
        "remap_path": remap_path,
        "num_categories": num_categories,
    }

    def make_eval_fn(model, adapter, metrics_engine, dataset_provider,
                     processed_news, eval_batch_size, output_run_dir, **_):
        def eval_fn(model, mode="val", epoch=None, **kwargs):
            return digat_evaluate(
                news_encoder=model.news_encoder,
                graph_encoder=model.graph_encoder,
                dataset_provider=dataset_provider,
                processed_news=processed_news,
                sag_data=ctx["sag_data"],
                adapter=adapter,
                metrics_calculator=metrics_engine,
                D=model.D,
                num_categories=model.num_categories,
                max_history=model.max_history,
                mode=mode,
                batch_size=eval_batch_size,
                id_remap=ctx["id_remap"],
                save_predictions_path=str(output_run_dir / "predictions"),
                epoch=epoch,
            )
        return eval_fn

    def rebuild_test_remap(dataset_provider, processed_news):
        if ctx["remap_path"] is not None and ctx["remap_path"].exists():
            ctx["remap_path"].unlink()
        ctx["id_remap"] = build_id_remap(
            dataset_provider, processed_news, ctx["remap_path"]
        )

    return ModelSetupResult(
        features=features,
        labels=labels,
        train_dataset=None,
        eval_context=ctx,
        make_eval_fn=make_eval_fn,
        rebuild_test_remap=rebuild_test_remap,
    )
