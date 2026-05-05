"""TCCM-specific setup: token-CTR tables, per-impression buckets, eval closure."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.data.processing.models.digat import build_id_remap
from src.core.data.processing.models.tccm import (
    build_per_token_buckets,
    cache_token_ctr_tables,
    compute_token_ctr_tables,
    load_token_ctr_tables,
)
from src.core.io.progress import create_progress
from src.core.models.evaluations.custom.tccm import tccm_fast_evaluate

from .types import ModelSetupResult


def _build_tccm_cache(dataset_provider, processed_news, spec) -> dict:
    """Build / load the TCCM token-CTR tables and dataset metadata."""
    import pandas as pd

    arch = spec.model.architecture
    pop_cfg = arch.popularity_encoder
    time_cfg = arch.time_module
    bins = int(pop_cfg.get("token_embedding_bins", 200))
    bucket_hours = int(getattr(dataset_provider, "popularity_bucket_hours", 1) or 1)
    timeliness_bins = int(time_cfg.get("embedding_bins", 505))

    cache_dir = (
        dataset_provider.dataset_path / "processed"
        if hasattr(dataset_provider, "dataset_path")
        else None
    )

    cached = load_token_ctr_tables(cache_dir) if cache_dir is not None else None
    if cached is not None:
        word_pop, entity_pop, dataset_start, c_bucket_h, num_buckets, _bins = cached
        if c_bucket_h != bucket_hours:
            cached = None  # bucket_hours mismatch — recompute

    if cached is None:
        col_names = ["impression_id", "user_id", "time", "history", "impressions"]
        train_path = dataset_provider.dataset_path / "train" / "behaviors.tsv"
        valid_path = dataset_provider.dataset_path / "valid" / "behaviors.tsv"
        df = pd.read_csv(train_path, sep="\t", header=None, names=col_names)
        # Include valid behaviors so wall-clock buckets cover the test
        # window. Causal ``-1`` shift in build_per_token_buckets keeps
        # the lookup strictly past — no leakage.
        if valid_path.exists():
            df_val = pd.read_csv(valid_path, sep="\t", header=None, names=col_names)
            df = pd.concat([df, df_val], ignore_index=True)
        news_ids_str = processed_news["news_ids_original_strings"]
        news_str_to_idx = {nid: i for i, nid in enumerate(news_ids_str)}
        word_pop, entity_pop, dataset_start, num_buckets = compute_token_ctr_tables(
            df,
            news_str_to_idx=news_str_to_idx,
            news_title_tokens=processed_news["tokens"],
            news_entity_indices=processed_news["entity_indices"],
            vocab_size=int(processed_news["vocab_size"]),
            entity_vocab_size=int(processed_news["entity_vocab_size"]),
            bucket_hours=bucket_hours,
            bins=bins,
        )
        if cache_dir is not None:
            cache_token_ctr_tables(
                cache_dir,
                word_pop=word_pop,
                entity_pop=entity_pop,
                dataset_start=dataset_start,
                bucket_hours=bucket_hours,
                num_buckets=num_buckets,
                bins=bins,
            )

    return {
        "word_pop": word_pop,
        "entity_pop": entity_pop,
        "dataset_start": dataset_start,
        "bucket_hours": bucket_hours,
        "num_buckets": int(num_buckets),
        "bins": bins,
        "timeliness_bins": timeliness_bins,
    }


def _build_features(dataset_provider) -> tuple[dict, np.ndarray]:
    """Replicate the standard PyTorch feature builder used by the runner.

    Kept inline here so the setup hook can return a complete
    ``features`` dict (the runner's ``_build_train_features`` is private
    and not part of the framework-agnostic surface).
    """
    data = dataset_provider.train_behaviors_data
    features: dict = {}

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

    # PP-Rec entity + category concat (matches runner._build_train_features)
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

    # PP-Rec CTR features: history CTR discretized to int buckets,
    # candidate CTR stays float for the scaler.
    if "history_news_ctr" in data:
        ctr = np.asarray(data["history_news_ctr"])
        features["hist_ctr"] = np.minimum(np.ceil(ctr * 200).astype(np.int32), 199)
        features["cand_ctr"] = np.asarray(data["candidate_news_ctr"])
    if "candidate_news_recency" in data:
        features["cand_recency"] = np.asarray(data["candidate_news_recency"])

    labels = np.asarray(data["labels"])
    return features, labels


def setup_tccm(spec, dataset_provider, processed_news) -> ModelSetupResult:
    """Prepare TCCM training: token-CTR tables, ID remap, per-impression
    bucket lookups, and the popularity-encoder eval closure factory."""
    tccm_cache = _build_tccm_cache(dataset_provider, processed_news, spec)

    remap_path = (
        dataset_provider.dataset_path / "processed" / "tccm_id_remap.npy"
        if hasattr(dataset_provider, "dataset_path")
        else None
    )
    id_remap = build_id_remap(dataset_provider, processed_news, remap_path)
    tccm_cache["id_remap"] = id_remap

    def _remap_ids(ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if id_remap is None:
            return ids
        return np.where(ids < len(id_remap), id_remap[ids], 0).astype(np.int64)

    features, labels = _build_features(dataset_provider)
    cand_pop_buckets, cand_news_exist_time = build_per_token_buckets(
        candidate_news_ids=_remap_ids(
            dataset_provider.train_behaviors_data["candidate_news_ids"]
        ),
        impression_times=np.asarray(
            dataset_provider.train_behaviors_data["impression_times"]
        ),
        news_title_tokens=processed_news["tokens"],
        news_entity_indices=processed_news["entity_indices"],
        word_pop=tccm_cache["word_pop"],
        entity_pop=tccm_cache["entity_pop"],
        news_publish_time=processed_news["news_publish_time"],
        dataset_start=tccm_cache["dataset_start"],
        bucket_hours=tccm_cache["bucket_hours"],
        timeliness_bins=tccm_cache["timeliness_bins"],
    )
    features["cand_pop_buckets"] = cand_pop_buckets
    features["cand_news_exist_time"] = cand_news_exist_time

    # Mutable context — eval_fn reads, rebuild_test_remap writes.
    ctx = {
        "tccm_cache": tccm_cache,
        "id_remap": id_remap,
        "remap_path": remap_path,
    }

    def make_eval_fn(model, adapter, metrics_engine, dataset_provider,
                     processed_news, eval_batch_size, output_run_dir,
                     build_eval_dataloaders=None, **_):
        """Build the TCCM eval closure.

        ``build_eval_dataloaders`` is provided by the framework runner
        (PyTorch, Keras, JAX) so this hook stays framework-agnostic. Each
        runner's builder may return either a 3-tuple
        ``(news_dl, user_dl, imp_iter)`` (Keras / JAX style) or a dict
        with the same three keys (PyTorch style); both shapes are handled.
        """
        if build_eval_dataloaders is None:
            raise ValueError(
                "TCCM setup needs ``build_eval_dataloaders`` from the runner "
                "to build framework-native eval data."
            )

        from omegaconf import OmegaConf
        _eval_cfg = OmegaConf.create({"eval": {"batch_size": int(eval_batch_size)}})

        int_to_news_id_map = (
            dataset_provider.get_int_to_news_id_map()
            if hasattr(dataset_provider, "get_int_to_news_id_map")
            else None
        )

        def _unpack(provider):
            if isinstance(provider, dict):
                return (
                    provider["news_dataloader"],
                    provider["user_hist_dataloader"],
                    provider["impression_iterator"],
                )
            return provider  # already a 3-tuple

        def eval_fn(model, mode="val", epoch=None, **kwargs):
            news_dl, user_dl, imp_iter = _unpack(
                build_eval_dataloaders(dataset_provider, _eval_cfg, mode=mode)
            )
            behaviors_data = (
                dataset_provider.val_behaviors_data
                if mode == "val"
                else dataset_provider.test_behaviors_data
            )
            with create_progress(transient=True) as progress:
                return tccm_fast_evaluate(
                    model=model,
                    news_dataloader=news_dl,
                    user_hist_dataloader=user_dl,
                    impression_iterator=imp_iter,
                    behaviors_data=behaviors_data,
                    metrics_calculator=metrics_engine,
                    progress=progress,
                    adapter=adapter,
                    int_to_news_id_map=int_to_news_id_map,
                    tccm_cache=ctx["tccm_cache"],
                    processed_news=processed_news,
                    save_predictions_path=str(output_run_dir / "predictions"),
                    epoch=epoch,
                    mode=mode,
                )

        return eval_fn

    def rebuild_test_remap(dataset_provider, processed_news):
        if ctx["remap_path"] is not None and ctx["remap_path"].exists():
            ctx["remap_path"].unlink()
        ctx["id_remap"] = build_id_remap(
            dataset_provider, processed_news, ctx["remap_path"]
        )
        ctx["tccm_cache"]["id_remap"] = ctx["id_remap"]

    return ModelSetupResult(
        features=features,
        labels=labels,
        train_dataset=None,
        eval_context=ctx,
        make_eval_fn=make_eval_fn,
        rebuild_test_remap=rebuild_test_remap,
    )
