import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import keras
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from src.core.metrics.functions import NewsRecommenderMetrics
from src.core.io.saving import save_predictions_to_file_fn, get_output_run_dir
from src.frameworks.keras.utils import test_step_fn
from src.core.io.logging import log_metrics_to_console_fn

console = Console()


def run_evaluation_epoch(
        model: keras.Model,
        eval_dataloader,
        custom_metrics_calculator: NewsRecommenderMetrics,
        num_total_impressions: int,
        progress: Progress,
        mode: str = "val",  # "val" or "test"
        save_predictions_dir: Optional[Path] = None,
        epoch_idx: Optional[int] = None,  # 0-indexed
) -> Dict[str, float]:
    """
    Run validation or testing for one epoch (non-fast evaluation).
    This path is used if `cfg.eval.fast_evaluation` is False.
    It iterates through the dataloader, calls `test_step_fn`.
    """
    eval_progress_task = progress.add_task(
        f"Running {mode} (epoch {epoch_idx + 1 if epoch_idx is not None else 'N/A'})...",
        total=num_total_impressions,
        visible=True,
    )

    for m in model.metrics:
        m.reset_state()

    all_labels_batched = []
    all_predictions_batched = []
    all_impression_ids_batched = []

    processed_impressions_count = 0
    total_loss = 0.0

    for batch_features, batch_labels, batch_impression_ids in eval_dataloader:
        # `test_step_fn` updates compiled metrics and returns batch loss + compiled metrics results
        step_results = test_step_fn(model, (batch_features, batch_labels))
        total_loss += step_results["loss"]

        # Get raw predictions from the model for custom metrics calculation
        # The model.call(training=False) should return scores for the slate.
        batch_raw_predictions = model(batch_features, training=False)

        all_labels_batched.append(batch_labels.numpy())
        all_predictions_batched.append(batch_raw_predictions.numpy())
        all_impression_ids_batched.extend(  # Assuming IDs are string tensors or convertable
            [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in batch_impression_ids.numpy()
            ]
        )

        current_batch_size = batch_labels.shape[0]
        processed_impressions_count += current_batch_size
        progress.update(
            eval_progress_task,
            advance=current_batch_size,
            description=f"{mode} (Batch Loss: {step_results['loss']:.4f})",
        )
        if processed_impressions_count >= num_total_impressions:
            break

    progress.remove_task(eval_progress_task)

    if not all_labels_batched:
        console.log(f"[yellow][WARNING {mode}] No data processed in evaluation epoch.[/yellow]")
        # Return default zeroed metrics
        return {m.name: 0.0 for m in model.metrics} | {
            "loss": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "num_impressions": 0,
        }

    # Concatenate results from all batches
    concatenated_labels = np.concatenate(all_labels_batched, axis=0)
    concatenated_predictions = np.concatenate(all_predictions_batched, axis=0)

    # Get final results from compiled Keras metrics
    final_eval_metrics = {m.name: float(m.result().numpy()) for m in model.metrics} | {
        "loss": total_loss / len(all_labels_batched) if all_labels_batched else 0.0
    }

    # Calculate custom metrics (MRR, nDCG) using the custom calculator
    custom_eval_metrics = custom_metrics_calculator.compute_metrics(
        y_true=concatenated_labels, y_pred_logits=concatenated_predictions, progress=progress
    )

    final_eval_metrics = (
            final_eval_metrics | custom_eval_metrics | {"num_impressions": processed_impressions_count}
    )

    if save_predictions_dir:
        predictions_to_save = {
            imp_id: (gt.tolist(), pred.tolist())
            for imp_id, gt, pred in zip(
                all_impression_ids_batched, concatenated_labels, concatenated_predictions
            )
        }
        save_predictions_to_file_fn(predictions_to_save, save_predictions_dir, epoch_idx, mode)

    return final_eval_metrics


def get_main_comparison_metric(
        validation_metrics: Dict[str, float],
        min_improvement: float = 0.01,  # Minimum improvement required to consider it better
) -> Tuple[float, bool]:
    """Extracts the primary metric used for comparing epochs and checks if improvement is significant.

    Args:
        validation_metrics: Dictionary of validation metrics
        primary_metric_key: Key for the primary metric (default: "ndcg@10")
        min_improvement: Minimum improvement required to consider it better (default: 0.01)

    Returns:
        Tuple of (average_metric_value, is_significant_improvement)
    """
    # Define the main metrics we care about
    main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]

    # Get values for all main metrics
    metric_values = []
    for metric in main_metrics:
        value = validation_metrics.get(metric)
        if value is not None:
            metric_values.append(float(value))

    if not metric_values:
        console.log(
            "[yellow]No main metrics found in validation metrics. Using 0.0 as fallback.[/yellow]"
        )
        return 0.0, False

    # Calculate average of main metrics
    average_metric = sum(metric_values) / len(metric_values)

    # Log the individual metrics and their average
    console.log("[bold blue]Main metrics values:[/bold blue]")
    for metric, value in zip(main_metrics, metric_values):
        console.log(f"  {metric}: {value:.4f}")
    console.log(f"[bold green]Average metric value: {average_metric:.4f}[/bold green]")

    # Check if the improvement is significant
    is_significant = average_metric > min_improvement

    return average_metric, is_significant


def _run_initial_validation(
        model: keras.Model,
        dataset_provider: Any,
        custom_metrics_engine: NewsRecommenderMetrics,
        progress_bar_manager: Progress,
        cfg: DictConfig,
) -> Dict[str, float]:
    """Run initial validation before training starts."""
    console.log("[bold yellow]Running Initial Validation (before training starts)...[/bold yellow]")

    # Setup output directory based on Hydra's current run path
    output_run_dir = get_output_run_dir(cfg)
    predictions_save_dir = output_run_dir / "predictions" / "initial_val"

    if cfg.eval.save_predictions:
        predictions_save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.eval.fast_evaluation:
        metrics = model.fast_evaluate(
            user_hist_dataloader=dataset_provider.user_history_dataloader(mode="val", batch_size=cfg.eval.batch_size),
            impression_iterator=dataset_provider.impression_dataloader(mode="val"),
            news_dataloader=dataset_provider.news_dataloader(batch_size=cfg.eval.batch_size),
            metrics_calculator=custom_metrics_engine,
            progress=progress_bar_manager,
            mode="initial_val",
            save_predictions_path=predictions_save_dir if cfg.eval.save_predictions else None,
            epoch=None,
            int_to_news_id_map=dataset_provider.get_int_to_news_id_map(),
        )
    else:
        metrics = run_evaluation_epoch(
            model,
            dataset_provider.val_dataloader(batch_size=cfg.eval.batch_size),
            custom_metrics_engine,
            dataset_provider.val_size,
            progress_bar_manager,
            mode="initial_val",
            save_predictions_dir=predictions_save_dir if cfg.eval.save_predictions else None,
            epoch_idx=None,
        )
    log_metrics_to_console_fn(metrics, "Initial Validation")
    return metrics


def _run_epoch_evaluation(
        model: keras.Model,
        dataset_provider: Any,
        custom_metrics_engine: NewsRecommenderMetrics,
        progress_bar_manager: Progress,
        cfg: DictConfig,
        epoch_idx: int,
        predictions_save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run evaluation for a single epoch."""
    if cfg.eval.fast_evaluation:
        # Create mode-specific directory for predictions
        mode_specific_dir = predictions_save_dir / "val" if predictions_save_dir else None
        if mode_specific_dir:
            mode_specific_dir.mkdir(parents=True, exist_ok=True)

        return model.fast_evaluate(
            user_hist_dataloader=dataset_provider.user_history_dataloader(mode="val"),
            impression_iterator=dataset_provider.impression_dataloader(mode="val"),
            news_dataloader=dataset_provider.news_dataloader(),
            metrics_calculator=custom_metrics_engine,
            progress=progress_bar_manager,
            mode="val",
            save_predictions_path=mode_specific_dir if cfg.eval.save_predictions else None,
            epoch=epoch_idx,
            int_to_news_id_map=dataset_provider.get_int_to_news_id_map(),
        )

    # Create mode-specific directory for predictions
    mode_specific_dir = predictions_save_dir / "val" if predictions_save_dir else None
    if mode_specific_dir:
        mode_specific_dir.mkdir(parents=True, exist_ok=True)

    return run_evaluation_epoch(
        model,
        dataset_provider.val_dataloader(batch_size=cfg.eval.batch_size),
        custom_metrics_engine,
        dataset_provider.val_size,
        progress_bar_manager,
        mode="val",
        save_predictions_dir=mode_specific_dir if cfg.eval.save_predictions else None,
        epoch_idx=epoch_idx,
    )


def _run_final_testing(
        model: keras.Model,
        dataset_provider: Any,
        custom_metrics_engine: NewsRecommenderMetrics,
        progress_bar_manager: Progress,
        cfg: DictConfig,
        best_model_weights_filepath: Path,
        last_model_weights_filepath: Path,
        best_epoch_metrics_tracking: Dict[str, Any],
        predictions_save_dir: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """Run final testing phase after training."""
    console.log("[bold]Loading best model for final testing...[/bold]")

    # Loading weights from the model
    model.load_weights(best_model_weights_filepath)

    # Load test data
    console.log("Loading test data for final evaluation...")
    dataset_provider._load_data(mode="test")

    # Create mode-specific directory for predictions
    mode_specific_dir = predictions_save_dir / "test" if predictions_save_dir else None
    if mode_specific_dir:
        mode_specific_dir.mkdir(parents=True, exist_ok=True)

    if cfg.eval.fast_evaluation:
        return model.fast_evaluate(
            user_hist_dataloader=dataset_provider.user_history_dataloader(mode="test", batch_size=cfg.eval.batch_size),
            impression_iterator=dataset_provider.impression_dataloader(mode="test"),
            news_dataloader=dataset_provider.news_dataloader(batch_size=cfg.eval.batch_size),
            metrics_calculator=custom_metrics_engine,
            progress=progress_bar_manager,
            mode="test",
            save_predictions_path=mode_specific_dir if cfg.eval.save_predictions else None,
            epoch=best_epoch_metrics_tracking.get("epoch_number", -1) - 1,
            int_to_news_id_map=dataset_provider.get_int_to_news_id_map(),
        )

    return run_evaluation_epoch(
        model,
        dataset_provider.test_dataloader(batch_size=cfg.eval.batch_size),
        custom_metrics_engine,
        dataset_provider.test_size,
        progress_bar_manager,
        mode="test",
        save_predictions_dir=mode_specific_dir if cfg.eval.save_predictions else None,
        epoch_idx=best_epoch_metrics_tracking.get("epoch_number", -1) - 1,
    )
