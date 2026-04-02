import time
from pathlib import Path
from typing import Any

import keras
import wandb
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from src.core.io.config import save_model_config
from src.core.io.logging import (
    log_metrics_to_console_fn,
    log_metrics_to_wandb_fn,
)
from src.core.io.saving import save_run_summary_fn
from src.core.metrics.functions import NewsRecommenderMetrics
from src.frameworks.keras.evaluation import (
    _run_final_testing,
    _run_initial_validation,
)

console = Console()


# =============================================================================
# Callback Classes
# =============================================================================


class FastEvaluationCallback(keras.callbacks.Callback):
    """Custom Keras callback for fast evaluation during training."""

    def __init__(
        self,
        dataset_provider: Any,
        custom_metrics_engine: NewsRecommenderMetrics,
        cfg: DictConfig,
        progress_bar_manager: Progress,
        predictions_save_dir: Path | None = None,
        wandb_history: dict[str, Any] | None = None,
        timing_metrics: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.dataset_provider = dataset_provider
        self.custom_metrics_engine = custom_metrics_engine
        self.cfg = cfg
        self.progress_bar_manager = progress_bar_manager
        self.predictions_save_dir = predictions_save_dir
        self.wandb_history = wandb_history or {}
        self.timing_metrics = timing_metrics or {}

        # Best epoch tracking
        self.best_epoch_metrics = {
            "average_metric_value": -float("inf"),
            "patience": cfg.train.early_stopping.patience,
        }
        self.wait = 0

        # Validation timing
        self.validation_start_time = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None):
        """Run fast evaluation at the end of each epoch."""
        if not self.cfg.eval.fast_evaluation:
            return

        # Start validation timing
        self.validation_start_time = time.time()

        # Create mode-specific directory for predictions
        mode_specific_dir = (
            self.predictions_save_dir / "val" if self.predictions_save_dir else None
        )
        if mode_specific_dir:
            mode_specific_dir.mkdir(parents=True, exist_ok=True)

        # Run fast evaluation
        val_metrics = self.model.fast_evaluate(
            user_hist_dataloader=self.dataset_provider.user_history_dataloader(
                mode="val", batch_size=self.cfg.eval.batch_size
            ),
            impression_iterator=self.dataset_provider.impression_dataloader(mode="val"),
            news_dataloader=self.dataset_provider.news_dataloader(
                batch_size=self.cfg.eval.batch_size
            ),
            metrics_calculator=self.custom_metrics_engine,
            progress=self.progress_bar_manager,
            mode="val",
            save_predictions_path=mode_specific_dir
            if self.cfg.eval.save_predictions
            else None,
            epoch=epoch,
            int_to_news_id_map=self.dataset_provider.get_int_to_news_id_map(),
        )

        # Calculate validation duration
        validation_duration = (
            time.time() - self.validation_start_time
            if self.validation_start_time
            else 0
        )

        # Store validation timing
        if "epoch_validation_times" in self.timing_metrics:
            self.timing_metrics["epoch_validation_times"].append(validation_duration)

        # Log metrics to console
        log_metrics_to_console_fn(val_metrics, f"Epoch {epoch + 1} Validation")

        # Log validation timing to console
        console.log(
            f"[dim]🔍 Epoch {epoch + 1} validation time: {validation_duration:.2f}s ({validation_duration / 60:.2f}min)[/dim]"
        )

        # Log to WandB if enabled
        if wandb.run and self.wandb_history is not None:
            # Add validation timing to metrics
            metrics_with_timing = {f"val/{k}": v for k, v in val_metrics.items()}
            metrics_with_timing["validation_duration_seconds"] = validation_duration
            metrics_with_timing["validation_duration_minutes"] = (
                validation_duration / 60.0
            )

            log_metrics_to_wandb_fn(
                metrics_with_timing,
                epoch + 1,
                self.wandb_history,
            )

        # Update best epoch metrics and check for early stopping
        is_new_best = self._update_best_epoch_metrics(epoch, val_metrics)

        if is_new_best:
            self.wait = 0
            # Save best model weights
            if hasattr(self, "best_model_path"):
                self.model.save_weights(str(self.best_model_path))
                # Save model configuration alongside weights
                save_model_config(self.cfg, self.best_model_path)
        else:
            self.wait += 1

        # Check for early stopping
        if self.wait >= self.cfg.train.early_stopping.patience:
            console.log(
                f"[bold red]Early stopping triggered after epoch {epoch + 1}.[/bold red]"
            )
            self.model.stop_training = True

    def _update_best_epoch_metrics(
        self,
        epoch_idx: int,
        val_metrics: dict[str, float],
        min_improvement: float = 0.01,
    ) -> bool:
        """Update best epoch metrics tracking."""
        # Calculate main comparison metric (average of AUC, MRR, nDCG@5, nDCG@10)
        main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
        metric_values = []

        for metric in main_metrics:
            value = val_metrics.get(metric)
            if value is not None:
                metric_values.append(float(value))

        if not metric_values:
            console.log(
                "[yellow]No main metrics found in validation metrics. Using 0.0 as fallback.[/yellow]"
            )
            return False

        current_avg_metric = sum(metric_values) / len(metric_values)
        previous_best = self.best_epoch_metrics.get(
            "average_metric_value", -float("inf")
        )

        # Check if this is a new best epoch
        is_new_best = (
            current_avg_metric > previous_best and current_avg_metric > min_improvement
        )

        if is_new_best:
            self.best_epoch_metrics.clear()
            self.best_epoch_metrics.update(
                {
                    "epoch_number": epoch_idx + 1,
                    "train_loss_at_best": float(
                        self.model.history.history.get("loss", [0.0])[-1]
                    ),
                    "val_loss_at_best": float(val_metrics.get("loss", float("nan"))),
                    "average_metric_value": current_avg_metric,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            console.log(
                f"[bold green]New best epoch! Average metric improved from {previous_best:.4f} to {current_avg_metric:.4f}[/bold green]"
            )
        else:
            console.log(
                f"[yellow]No improvement. Current average: {current_avg_metric:.4f}, Previous best: {previous_best:.4f}[/yellow]"
            )

        return is_new_best

    def set_model_save_path(self, best_model_path: Path):
        """Set the path where the best model should be saved."""
        self.best_model_path = best_model_path


class SlowEvaluationCallback(keras.callbacks.Callback):
    """Custom Keras callback for traditional batch-wise evaluation during training."""

    def __init__(
        self,
        dataset_provider: Any,
        custom_metrics_engine: NewsRecommenderMetrics,
        cfg: DictConfig,
        progress_bar_manager: Progress,
        predictions_save_dir: Path | None = None,
        wandb_history: dict[str, Any] | None = None,
        timing_metrics: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.dataset_provider = dataset_provider
        self.custom_metrics_engine = custom_metrics_engine
        self.cfg = cfg
        self.progress_bar_manager = progress_bar_manager
        self.predictions_save_dir = predictions_save_dir
        self.wandb_history = wandb_history or {}
        self.timing_metrics = timing_metrics or {}

        # Best epoch tracking
        self.best_epoch_metrics = {
            "average_metric_value": -float("inf"),
            "patience": cfg.train.early_stopping.patience,
        }
        self.wait = 0

        # Validation timing
        self.validation_start_time = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None):
        """Run slow evaluation at the end of each epoch."""
        if self.cfg.eval.fast_evaluation:
            return

        # Start validation timing
        self.validation_start_time = time.time()

        # Create mode-specific directory for predictions
        mode_specific_dir = (
            self.predictions_save_dir / "val" if self.predictions_save_dir else None
        )
        if mode_specific_dir:
            mode_specific_dir.mkdir(parents=True, exist_ok=True)

        # Run traditional evaluation
        from src.frameworks.keras.evaluation import run_evaluation_epoch

        val_metrics = run_evaluation_epoch(
            self.model,
            self.dataset_provider.val_dataloader(batch_size=self.cfg.eval.batch_size),
            self.custom_metrics_engine,
            self.dataset_provider.val_size,
            self.progress_bar_manager,
            mode="val",
            save_predictions_dir=mode_specific_dir
            if self.cfg.eval.save_predictions
            else None,
            epoch_idx=epoch,
        )

        # Calculate validation duration
        validation_duration = (
            time.time() - self.validation_start_time
            if self.validation_start_time
            else 0
        )

        # Store validation timing
        if "epoch_validation_times" in self.timing_metrics:
            self.timing_metrics["epoch_validation_times"].append(validation_duration)

        # Log metrics to console
        log_metrics_to_console_fn(val_metrics, f"Epoch {epoch + 1} Validation")

        # Log validation timing to console
        console.log(
            f"[dim]🔍 Epoch {epoch + 1} validation time: {validation_duration:.2f}s ({validation_duration / 60:.2f}min)[/dim]"
        )

        # Log to WandB if enabled
        if wandb.run and self.wandb_history is not None:
            # Add validation timing to metrics
            metrics_with_timing = {f"val/{k}": v for k, v in val_metrics.items()}
            metrics_with_timing["validation_duration_seconds"] = validation_duration
            metrics_with_timing["validation_duration_minutes"] = (
                validation_duration / 60.0
            )

            log_metrics_to_wandb_fn(
                metrics_with_timing,
                epoch + 1,
                self.wandb_history,
            )

        # Update best epoch metrics and check for early stopping
        is_new_best = self._update_best_epoch_metrics(epoch, val_metrics)

        if is_new_best:
            self.wait = 0
            # Save best model weights
            if hasattr(self, "best_model_path"):
                self.model.save_weights(str(self.best_model_path))
                # Save model configuration alongside weights
                save_model_config(self.cfg, self.best_model_path)
        else:
            self.wait += 1

        # Check for early stopping
        if self.wait >= self.cfg.train.early_stopping.patience:
            console.log(
                f"[bold red]Early stopping triggered after epoch {epoch + 1}.[/bold red]"
            )
            self.model.stop_training = True

    def _update_best_epoch_metrics(
        self,
        epoch_idx: int,
        val_metrics: dict[str, float],
        min_improvement: float = 0.01,
    ) -> bool:
        """Update best epoch metrics tracking."""
        # Calculate main comparison metric (average of AUC, MRR, nDCG@5, nDCG@10)
        main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
        metric_values = []

        for metric in main_metrics:
            value = val_metrics.get(metric)
            if value is not None:
                metric_values.append(float(value))

        if not metric_values:
            console.log(
                "[yellow]No main metrics found in validation metrics. Using 0.0 as fallback.[/yellow]"
            )
            return False

        current_avg_metric = sum(metric_values) / len(metric_values)
        previous_best = self.best_epoch_metrics.get(
            "average_metric_value", -float("inf")
        )

        # Check if this is a new best epoch
        is_new_best = (
            current_avg_metric > previous_best and current_avg_metric > min_improvement
        )

        if is_new_best:
            self.best_epoch_metrics.clear()
            self.best_epoch_metrics.update(
                {
                    "epoch_number": epoch_idx + 1,
                    "train_loss_at_best": float(
                        self.model.history.history.get("loss", [0.0])[-1]
                    ),
                    "val_loss_at_best": float(val_metrics.get("loss", float("nan"))),
                    "average_metric_value": current_avg_metric,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            console.log(
                f"[bold green]New best epoch! Average metric improved from {previous_best:.4f} to {current_avg_metric:.4f}[/bold green]"
            )
        else:
            console.log(
                f"[yellow]No improvement. Current average: {current_avg_metric:.4f}, Previous best: {previous_best:.4f}[/yellow]"
            )

        return is_new_best

    def set_model_save_path(self, best_model_path: Path):
        """Set the path where the best model should be saved."""
        self.best_model_path = best_model_path


class TrainingMetricsCallback(keras.callbacks.Callback):
    """Callback to track and log training metrics (loss + timing) per epoch."""

    def __init__(
        self,
        timing_metrics: dict[str, Any],
        wandb_history: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.timing_metrics = timing_metrics
        self.wandb_history = wandb_history or {}
        self.epoch_training_start_time = None

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None):
        """Record epoch training start time."""
        self.epoch_training_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None):
        """Record and log epoch training metrics."""
        # Calculate epoch training time (excluding validation)
        epoch_training_time = 0
        if self.epoch_training_start_time:
            epoch_training_time = time.time() - self.epoch_training_start_time
            self.timing_metrics["epoch_training_times"].append(epoch_training_time)

        # Extract training loss from logs
        training_loss = logs.get("loss", 0.0) if logs else 0.0

        # Log training metrics to console
        console.log(
            f"[dim]⏱️ Epoch {epoch + 1} training: {epoch_training_time:.2f}s ({epoch_training_time / 60:.2f}min) | Loss: {training_loss:.4f}[/dim]"
        )

        # Log to WandB if enabled
        if wandb.run and self.wandb_history is not None:
            training_metrics = {
                "train/loss": training_loss,
                "train/epoch_training_time_seconds": epoch_training_time,
                "train/epoch_training_time_minutes": epoch_training_time / 60.0,
            }

            # Log training metrics to WandB
            wandb.log(training_metrics, step=epoch + 1)

            # Store in wandb_history for consistency
            for key, value in training_metrics.items():
                self.wandb_history[f"epoch_{epoch + 1}_{key}"] = value


class ComprehensiveTimingCallback(keras.callbacks.Callback):
    """Callback to track overall experiment timing phases."""

    def __init__(
        self,
        timing_metrics: dict[str, Any],
        wandb_history: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.timing_metrics = timing_metrics
        self.wandb_history = wandb_history or {}
        self.training_phase_start_time = None

    def on_train_begin(self, logs: dict[str, float] | None = None):
        """Record training phase start time."""
        self.training_phase_start_time = time.time()
        console.log("[bold blue]🏋️ Training phase started![/bold blue]")

    def on_train_end(self, logs: dict[str, float] | None = None):
        """Calculate total training time and summary statistics."""
        if self.training_phase_start_time:
            self.timing_metrics["total_training_time"] = (
                time.time() - self.training_phase_start_time
            )

            # Calculate average epoch training time
            if self.timing_metrics["epoch_training_times"]:
                avg_epoch_training_time = sum(
                    self.timing_metrics["epoch_training_times"]
                ) / len(self.timing_metrics["epoch_training_times"])
                console.log(
                    f"[bold cyan]📊 Average epoch training time: {avg_epoch_training_time:.2f}s ({avg_epoch_training_time / 60:.2f}min)[/bold cyan]"
                )

            console.log(
                f"[bold green]✅ Training phase completed! Total training time: {self.timing_metrics['total_training_time']:.2f}s ({self.timing_metrics['total_training_time'] / 60:.2f}min)[/bold green]"
            )


class RichProgressCallback(keras.callbacks.Callback):
    """Keras callback to integrate with Rich progress bars."""

    def __init__(
        self,
        progress_manager: Progress,
        num_epochs: int,
        steps_per_epoch: int | None = None,
    ):
        super().__init__()
        self.progress_manager = progress_manager
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.overall_task = None
        self.epoch_task = None
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        """Initialize overall progress tracking."""
        self.overall_task = self.progress_manager.add_task(
            "Overall Training Progress", total=self.num_epochs
        )

    def on_epoch_begin(self, epoch, logs=None):
        """Initialize epoch progress tracking."""
        self.current_epoch = epoch

        # Use the provided steps_per_epoch if available
        total_steps = self.steps_per_epoch

        self.epoch_task = self.progress_manager.add_task(
            f"Epoch {epoch + 1}/{self.num_epochs}", total=total_steps, visible=True
        )

    def on_batch_end(self, batch, logs=None):
        """Update epoch progress after each batch."""
        if self.epoch_task is not None:
            loss = logs.get("loss", 0.0) if logs else 0.0
            self.progress_manager.update(
                self.epoch_task,
                advance=1,
                description=f"Epoch {self.current_epoch + 1}/{self.num_epochs} (Loss: {loss:.4f})",
            )

    def on_epoch_end(self, epoch, logs=None):
        """Clean up epoch progress and update overall progress."""
        if self.epoch_task is not None:
            self.progress_manager.remove_task(self.epoch_task)
            self.epoch_task = None

        if self.overall_task is not None:
            description = f"Completed Epoch {epoch + 1}/{self.num_epochs}"

            self.progress_manager.update(
                self.overall_task, advance=1, description=description
            )

    def on_train_end(self, logs=None):
        """Clean up overall progress tracking."""
        if self.overall_task is not None:
            self.progress_manager.remove_task(self.overall_task)


# =============================================================================
# Helper Functions
# =============================================================================


def _format_time_duration(seconds: float) -> str:
    """Format time duration in a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def _log_comprehensive_timing_summary(
    timing_metrics: dict[str, Any], wandb_history: dict[str, Any] | None = None
):
    """Log comprehensive timing summary to console and WandB."""
    console.log("\n" + "=" * 70)
    console.log("[bold cyan]📊 COMPREHENSIVE TIMING SUMMARY[/bold cyan]")
    console.log("=" * 70)

    # Calculate derived metrics
    epoch_training_times = timing_metrics.get("epoch_training_times", [])
    epoch_validation_times = timing_metrics.get("epoch_validation_times", [])

    if epoch_training_times:
        avg_epoch_training_time = sum(epoch_training_times) / len(epoch_training_times)
        total_epoch_training_time = sum(epoch_training_times)
        console.log(
            f"[green]⏱️  Average epoch training time:[/green] {avg_epoch_training_time:.2f}s ({_format_time_duration(avg_epoch_training_time)})"
        )
        console.log(
            f"[green]🏋️  Total epoch training time:[/green] {total_epoch_training_time:.2f}s ({_format_time_duration(total_epoch_training_time)})"
        )

    if epoch_validation_times:
        avg_epoch_validation_time = sum(epoch_validation_times) / len(
            epoch_validation_times
        )
        total_epoch_validation_time = sum(epoch_validation_times)
        console.log(
            f"[blue]🔍 Average epoch validation time:[/blue] {avg_epoch_validation_time:.2f}s ({_format_time_duration(avg_epoch_validation_time)})"
        )
        console.log(
            f"[blue]🔬 Total epoch validation time:[/blue] {total_epoch_validation_time:.2f}s ({_format_time_duration(total_epoch_validation_time)})"
        )

    # Major phase timings
    total_training_time = timing_metrics.get("total_training_time", 0)
    total_testing_time = timing_metrics.get("total_testing_time", 0)
    total_experiment_time = timing_metrics.get("total_experiment_time", 0)

    console.log(
        f"[yellow]🎯 Total training phase time:[/yellow] {total_training_time:.2f}s ({_format_time_duration(total_training_time)})"
    )
    if total_testing_time > 0:
        console.log(
            f"[magenta]🧪 Total testing phase time:[/magenta] {total_testing_time:.2f}s ({_format_time_duration(total_testing_time)})"
        )
    console.log(
        f"[bold red]🚀 Total experiment time:[/bold red] {total_experiment_time:.2f}s ({_format_time_duration(total_experiment_time)})"
    )

    # Calculate percentages
    if total_experiment_time > 0:
        training_percentage = (total_training_time / total_experiment_time) * 100
        testing_percentage = (total_testing_time / total_experiment_time) * 100
        console.log(
            f"[dim]📈 Training: {training_percentage:.1f}% | Testing: {testing_percentage:.1f}% of total experiment time[/dim]"
        )

    console.log("=" * 70)

    # Log to WandB if enabled
    if wandb.run and wandb_history is not None:
        timing_summary = {
            "timing/total_experiment_time_seconds": total_experiment_time,
            "timing/total_experiment_time_minutes": total_experiment_time / 60.0,
            "timing/total_experiment_time_hours": total_experiment_time / 3600.0,
            "timing/total_training_time_seconds": total_training_time,
            "timing/total_training_time_minutes": total_training_time / 60.0,
            "timing/total_testing_time_seconds": total_testing_time,
            "timing/total_testing_time_minutes": total_testing_time / 60.0,
        }

        if epoch_training_times:
            timing_summary.update(
                {
                    "timing/avg_epoch_training_time_seconds": avg_epoch_training_time,
                    "timing/avg_epoch_training_time_minutes": avg_epoch_training_time
                    / 60.0,
                }
            )

        if epoch_validation_times:
            timing_summary.update(
                {
                    "timing/avg_epoch_validation_time_seconds": avg_epoch_validation_time,
                    "timing/avg_epoch_validation_time_minutes": avg_epoch_validation_time
                    / 60.0,
                }
            )

        # Log to WandB
        wandb.log(timing_summary)

        # Store in wandb_history
        wandb_history.update(timing_summary)

        console.log("[dim]📤 Comprehensive timing metrics logged to WandB[/dim]")


def _setup_training_directories(
    output_directory: Path, model_name: str
) -> tuple[Path, Path, Path]:
    """
    Setup directories for models and predictions.

    Args:
        output_directory: Path to the output directory
        model_name: Name of the model
    Returns:
        Tuple[Path, Path, Path]: Path to the best model, path to the last model, and path to the predictions
    """
    models_save_dir = output_directory / "models"
    models_save_dir.mkdir(parents=True, exist_ok=True)
    predictions_save_dir = output_directory / "predictions"
    predictions_save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_save_dir / f"{model_name}_best.weights.h5"
    last_model_path = models_save_dir / f"{model_name}_last.weights.h5"

    return best_model_path, last_model_path, predictions_save_dir


# =============================================================================
# Main Training Orchestrator
# =============================================================================


def training_loop_orchestrator(
    model: keras.Model,
    dataset_provider: Any,
    cfg: DictConfig,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """Orchestrates training using Keras 3 model.fit() with custom callbacks."""

    # Start timing the entire experiment
    experiment_start_time = time.time()
    console.log("[bold blue]🚀 Starting experiment timing...[/bold blue]")

    # Setup directories
    best_model_path, last_model_path, predictions_save_dir = (
        _setup_training_directories(output_directory, model.name)
    )

    # Setup WandB history tracking
    wandb_history = {} if cfg.logging.enable_wandb else None

    # Initialize timing tracking dictionary
    timing_metrics = {
        "epoch_training_times": [],
        "epoch_validation_times": [],
        "total_training_time": 0.0,
        "total_testing_time": 0.0,
        "total_experiment_time": 0.0,
    }

    # JIT compilation warmup to avoid slow first epoch
    try:
        from src.frameworks.keras.utils import warmup_jit_compilation

        console.log(
            "[bold]Warming up JIT compilation to avoid slow first epoch...[/bold]"
        )

        # Get a sample batch for warmup
        model_name = (
            cfg.spec.model.name.lower()
            if hasattr(cfg, "spec") and cfg.spec is not None
            else cfg.model._target_.split(".")[-1].lower()
        )
        sample_dataloader = dataset_provider.train_dataloader(
            batch_size=min(2, cfg.train.batch_size),  # Small batch for warmup
            model_name=model_name,
        )

        # Get first batch
        for sample_batch in sample_dataloader:
            warmup_jit_compilation(model, sample_batch)
            break

        console.log("[green]JIT warmup completed![/green]")
    except Exception as e:
        console.log(f"[yellow]JIT warmup skipped: {e}[/yellow]")

    initial_metrics = {}
    if cfg.eval.run_initial_eval:
        # Initial validation
        initial_metrics = _run_initial_validation(
            model, dataset_provider, custom_metrics_engine, progress_bar_manager, cfg
        )
        if wandb.run and wandb_history is not None:
            log_metrics_to_wandb_fn(
                {f"initial_val/{k}": v for k, v in initial_metrics.items()},
                0,
                wandb_history,
            )

    # Model Summary
    if hasattr(model, "training_model") and model.training_model is not None:
        console.log(
            f"[bold cyan]Summary of {model.name} Training Model (internal):[/bold cyan]"
        )
        model.training_model.summary(print_fn=lambda s: console.log(s))
    if hasattr(model, "scorer_model") and model.scorer_model is not None:
        console.log(
            f"[bold cyan]Summary of {model.name} Scorer Model (internal):[/bold cyan]"
        )
        model.scorer_model.summary(print_fn=lambda s: console.log(s))

    console.log(f"[bold cyan]Summary of {model.name} Model (main wrapper):[/bold cyan]")
    model.summary(print_fn=lambda s: console.log(s))

    # Prepare training and validation datasets
    console.log("[bold]Preparing datasets for Keras 3 training...[/bold]")

    # Extract model name from config
    model_name = (
        cfg.spec.model.name.lower()
        if hasattr(cfg, "spec") and cfg.spec is not None
        else cfg.model._target_.split(".")[-1].lower()
    )

    train_dataset = dataset_provider.train_dataloader(
        batch_size=cfg.train.batch_size, model_name=model_name
    )

    # Setup callbacks
    callbacks = []

    # Calculate steps per epoch
    steps_per_epoch = None
    if hasattr(train_dataset, "__len__"):
        try:
            steps_per_epoch = len(train_dataset)
        except:
            steps_per_epoch = None

    # Add training metrics callback (handles training loss + training epoch timing)
    training_metrics_callback = TrainingMetricsCallback(
        timing_metrics=timing_metrics, wandb_history=wandb_history
    )
    callbacks.append(training_metrics_callback)

    # Add comprehensive timing callback (handles overall training phase timing)
    timing_callback = ComprehensiveTimingCallback(
        timing_metrics=timing_metrics, wandb_history=wandb_history
    )
    callbacks.append(timing_callback)

    # Add Rich progress callback
    progress_callback = RichProgressCallback(
        progress_bar_manager, cfg.train.num_epochs, steps_per_epoch
    )
    callbacks.append(progress_callback)

    # Add evaluation callback (fast or slow)
    if cfg.eval.fast_evaluation:
        eval_callback = FastEvaluationCallback(
            dataset_provider=dataset_provider,
            custom_metrics_engine=custom_metrics_engine,
            cfg=cfg,
            progress_bar_manager=progress_bar_manager,
            predictions_save_dir=predictions_save_dir,
            wandb_history=wandb_history,
            timing_metrics=timing_metrics,
        )
    else:
        eval_callback = SlowEvaluationCallback(
            dataset_provider=dataset_provider,
            custom_metrics_engine=custom_metrics_engine,
            cfg=cfg,
            progress_bar_manager=progress_bar_manager,
            predictions_save_dir=predictions_save_dir,
            wandb_history=wandb_history,
            timing_metrics=timing_metrics,
        )

    eval_callback.set_model_save_path(best_model_path)
    callbacks.append(eval_callback)

    # Add ModelCheckpoint callback for last model
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(last_model_path),
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    callbacks.append(checkpoint_callback)

    # Add ReduceLROnPlateau if configured
    if (
        hasattr(cfg.train, "reduce_lr_on_plateau")
        and cfg.train.reduce_lr_on_plateau.enabled
    ):
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=cfg.train.reduce_lr_on_plateau.factor,
            patience=cfg.train.reduce_lr_on_plateau.patience,
            min_lr=cfg.train.reduce_lr_on_plateau.min_lr,
            verbose=1,
        )
        callbacks.append(reduce_lr_callback)

    # Verify model is compiled (should always be true after model initialization)
    if not model.compiled:
        raise RuntimeError(
            "Model is not compiled! This should not happen as the model should be "
            "compiled during initialization in initialize_model_and_dataset(). "
            "Please check the model initialization process."
        )

    # Main training using model.fit()
    console.log("[bold]Starting Keras 3 model.fit() training...[/bold]")

    try:
        model.fit(
            train_dataset,
            epochs=cfg.train.num_epochs,
            callbacks=callbacks,
            verbose=0,  # We handle progress with Rich
            validation_data=None,  # We handle validation with custom callbacks
        )

        console.log("[bold green]Training completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.log("[bold red]Training interrupted by user.[/bold red]")
        history = None
    except Exception as e:
        console.log(f"[bold red]Training failed with error: {e}[/bold red]")
        raise

    # Get best epoch metrics from the evaluation callback
    best_epoch_metrics = eval_callback.best_epoch_metrics

    # Final testing with timing
    test_metrics = None
    if cfg.eval.run_test_after_training:
        console.log("[bold blue]🧪 Starting testing phase...[/bold blue]")
        testing_start_time = time.time()

        test_metrics = _run_final_testing(
            model,
            dataset_provider,
            custom_metrics_engine,
            progress_bar_manager,
            cfg,
            best_model_path,
            last_model_path,
            best_epoch_metrics,
            predictions_save_dir,
        )

        timing_metrics["total_testing_time"] = time.time() - testing_start_time
        console.log(
            f"[bold green]✅ Testing phase completed! Testing time: {timing_metrics['total_testing_time']:.2f}s ({timing_metrics['total_testing_time'] / 60:.2f}min)[/bold green]"
        )

    # Calculate total experiment time
    timing_metrics["total_experiment_time"] = time.time() - experiment_start_time

    # Log comprehensive timing summary
    _log_comprehensive_timing_summary(timing_metrics, wandb_history)

    # Save results and cleanup
    if test_metrics:
        log_metrics_to_console_fn(test_metrics, "Final Test")
        if wandb.run and wandb_history is not None:
            log_metrics_to_wandb_fn(
                {f"test/{k}": v for k, v in test_metrics.items()},
                cfg.train.num_epochs + 1,
                wandb_history,
            )

    save_run_summary_fn(
        output_directory,
        cfg,
        initial_metrics,
        best_epoch_metrics,
        test_metrics,
        wandb_history,
    )

    if wandb.run:
        wandb.summary.update(
            {
                "best_epoch": best_epoch_metrics.get("epoch_number"),
                "best_val_metric": best_epoch_metrics.get("average_metric_value"),
                **{f"best/{k}": v for k, v in best_epoch_metrics.items()},
            }
        )
        wandb.finish()
        console.log("Wandb session finished.")

    return best_epoch_metrics, test_metrics
