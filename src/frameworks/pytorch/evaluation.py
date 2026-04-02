"""Fast evaluation with precomputed vectors for PyTorch models.

Delegates to ``BaseModel.fast_evaluate`` which already handles the full
pipeline (precompute news/user vectors, score impressions, compute metrics).
This module provides a thin convenience wrapper matching the expected call
signature from the training loop.
"""

from typing import Any, Dict, Optional

import torch
from rich.progress import Progress


def run_fast_evaluation(
    model: torch.nn.Module,
    dataset_provider: Any,
    metrics_engine: Any,
    progress: Progress,
    cfg: Any,
    mode: str = "validate",
    save_predictions_path: Optional[str] = None,
    epoch: Optional[int] = None,
    int_to_news_id_map: Optional[Dict] = None,
) -> Dict[str, float]:
    """Run fast evaluation using precomputed news and user vectors.

    This is the PyTorch equivalent of calling ``model.fast_evaluate(...)``
    from the Keras training pipeline.

    Args:
        model: A PyTorch ``BaseModel`` subclass (NRMS / NAML / LSTUR).
        dataset_provider: Object that exposes ``user_hist_dataloader``,
            ``news_dataloader``, and ``impression_iterator`` attributes
            (or a callable returning them).
        metrics_engine: Core metrics calculator with ``compute_metrics``
            and ``METRIC_NAMES``.
        progress: Rich progress bar manager.
        cfg: Configuration object (used for any runtime flags).
        mode: ``"validate"`` or ``"test"``.
        save_predictions_path: Optional path to save predictions.
        epoch: Current epoch number (for file naming).
        int_to_news_id_map: Optional mapping from int news ids to string ids.

    Returns:
        Dictionary of metric name -> float value.
    """
    model.eval()

    with torch.no_grad():
        # The dataset_provider is expected to expose these three iterables.
        # If it is a dict, unpack; otherwise assume attribute access.
        if isinstance(dataset_provider, dict):
            user_hist_dl = dataset_provider["user_hist_dataloader"]
            news_dl = dataset_provider["news_dataloader"]
            impression_iter = dataset_provider["impression_iterator"]
        else:
            user_hist_dl = dataset_provider.user_hist_dataloader
            news_dl = dataset_provider.news_dataloader
            impression_iter = dataset_provider.impression_iterator

        metrics = model.fast_evaluate(
            user_hist_dataloader=user_hist_dl,
            news_dataloader=news_dl,
            impression_iterator=impression_iter,
            metrics_calculator=metrics_engine,
            progress=progress,
            mode=mode,
            save_predictions_path=save_predictions_path,
            epoch=epoch,
            int_to_news_id_map=int_to_news_id_map,
        )

    return metrics
