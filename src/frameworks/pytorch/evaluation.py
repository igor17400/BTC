"""Fast evaluation with precomputed vectors for PyTorch models.

Provides a standalone ``fast_evaluate`` function isomorphic with the
Keras and JAX evaluation modules.
"""

from typing import Any

import torch
from rich.progress import Progress


def fast_evaluate(
    model: torch.nn.Module,
    dataset_provider: Any,
    metrics_engine: Any,
    progress: Progress,
    cfg: Any,
    mode: str = "validate",
    save_predictions_path: str | None = None,
    epoch: int | None = None,
    int_to_news_id_map: dict | None = None,
) -> dict[str, float]:
    """Run fast evaluation using precomputed news and user vectors.

    Isomorphic with ``fast_evaluate`` in Keras and JAX evaluation modules.

    Args:
        model: A PyTorch ``BaseModel`` subclass (NRMS / NAML / LSTUR).
        dataset_provider: Dict with ``user_hist_dataloader``,
            ``news_dataloader``, and ``impression_iterator`` keys.
        metrics_engine: Core metrics calculator.
        progress: Rich progress bar manager.
        cfg: Configuration object.
        mode: ``"validate"`` or ``"test"``.
        save_predictions_path: Optional path to save predictions.
        epoch: Current epoch number.
        int_to_news_id_map: Optional mapping from int news ids to string ids.

    Returns:
        Dictionary of metric name -> float value.
    """
    model.eval()

    eval_mode = "val" if mode == "validate" else mode
    batch_size = getattr(cfg.eval, "batch_size", 64)

    with torch.no_grad():
        if isinstance(dataset_provider, dict):
            user_hist_dl = dataset_provider["user_hist_dataloader"]
            news_dl = dataset_provider["news_dataloader"]
            impression_iter = dataset_provider["impression_iterator"]
        else:
            user_hist_dl = dataset_provider.user_history_dataloader(
                mode=eval_mode, batch_size=batch_size
            )
            news_dl = dataset_provider.news_dataloader(batch_size=batch_size)
            impression_iter = dataset_provider.impression_dataloader(mode=eval_mode)

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


# Keep old name as alias for backwards compatibility
run_fast_evaluation = fast_evaluate
