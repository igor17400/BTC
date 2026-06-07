"""CROWN-specific setup: standard training features + custom eval closure.

CROWN's user representation is **candidate-aware** (reference
``seongeunryu/crown-www25/userEncoders.py``), so the default fast
evaluator — which precomputes a single user vector per impression and
dot-products against candidates — produces incorrect scores. This setup
hook keeps the standard training path (CROWN consumes the same packed
``[news_idx | cat | subcat]`` features as every other model) but
replaces the evaluator with :func:`crown_fast_evaluate`, which:

1. Precomputes news vectors (full news_repr).
2. Precomputes per-impression GCN-updated history reps + masks (the
   expensive news_encoder + 1-layer SAGE runs once per impression).
3. Per impression, gathers candidate news vectors and runs
   candidate-aware attention + dot product.
"""

from __future__ import annotations

import numpy as np

from src.core.io.progress import create_progress
from src.core.models.evaluations.custom.crown import crown_fast_evaluate

from .types import ModelSetupResult


def _build_features(dataset_provider) -> tuple[dict, np.ndarray]:
    """Build CROWN training features from dataset_provider.

    Mirrors the CROWN branch of the standard runners' ``_build_train_features``:
    history + candidate ``news_idx`` plus category + subcategory ids.
    The news encoder reads ``news_idx`` and delegates per-news token
    lookup to its injected :class:`TextEncoder` (GloVe or PLM).
    """
    data = dataset_provider.train_behaviors_data
    features: dict = {
        "hist_features": np.asarray(data["histories_news_ids"]),
        "cand_features": np.asarray(data["candidate_news_ids"]),
    }
    if "history_news_categories" in data:
        features["hist_category"] = np.asarray(data["history_news_categories"])
        features["cand_category"] = np.asarray(data["candidate_news_categories"])
    if "history_news_subcategories" in data:
        features["hist_subcategory"] = np.asarray(data["history_news_subcategories"])
        features["cand_subcategory"] = np.asarray(data["candidate_news_subcategories"])
    if dataset_provider.process_user_id:
        features["user_ids"] = np.asarray(data["user_ids"])
    labels = np.asarray(data["labels"])
    return features, labels


def setup_crown(
    spec, dataset_provider, processed_news, encoder_cfg=None
) -> ModelSetupResult:
    """Wire CROWN to use :func:`crown_fast_evaluate` for evaluation."""
    features, labels = _build_features(dataset_provider)

    def make_eval_fn(
        model,
        adapter,
        metrics_engine,
        dataset_provider,
        processed_news,
        eval_batch_size,
        output_run_dir,
        build_eval_dataloaders=None,
        **_,
    ):
        if build_eval_dataloaders is None:
            raise ValueError(
                "CROWN setup needs ``build_eval_dataloaders`` from the runner."
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
            return provider

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
                return crown_fast_evaluate(
                    model=model,
                    news_dataloader=news_dl,
                    user_hist_dataloader=user_dl,
                    impression_iterator=imp_iter,
                    metrics_calculator=metrics_engine,
                    progress=progress,
                    adapter=adapter,
                    behaviors_data=behaviors_data,
                    int_to_news_id_map=int_to_news_id_map,
                    save_predictions_path=str(output_run_dir / "predictions"),
                    epoch=epoch,
                    mode=mode,
                )

        return eval_fn

    return ModelSetupResult(
        features=features,
        labels=labels,
        train_dataset=None,
        eval_context={},
        make_eval_fn=make_eval_fn,
        rebuild_test_remap=None,
    )
