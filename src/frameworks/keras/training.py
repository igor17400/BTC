"""Keras 3 training loop with model.fit() and custom evaluation callbacks."""

import time
from pathlib import Path
from typing import Any

import keras
import wandb
from omegaconf import DictConfig
from rich.progress import Progress

from src.core.io.config import save_model_config
from src.core.io.logging import (
    console,
    log_early_stopping,
    log_epoch_end,
    log_metrics_to_wandb_fn,
    log_test_results,
    log_training_complete,
    log_training_start,
)
from src.core.io.saving import save_run_summary_fn
from src.core.metrics.functions import NewsRecommenderMetrics
from src.frameworks.keras.evaluation import (
    _run_final_testing,
    _run_initial_validation,
)

# =============================================================================
# Callbacks
# =============================================================================


class EvaluationCallback(keras.callbacks.Callback):
    """Run fast evaluation at the end of each epoch, track best model."""

    def __init__(
        self,
        dataset_provider,
        metrics_engine: NewsRecommenderMetrics,
        cfg: DictConfig,
        progress: Progress,
        best_model_path: Path,
        wandb_history: dict | None = None,
    ):
        super().__init__()
        self.dataset_provider = dataset_provider
        self.metrics_engine = metrics_engine
        self.cfg = cfg
        self.progress = progress
        self.best_model_path = best_model_path
        self.wandb_history = wandb_history

        self.best_epoch_metrics: dict[str, Any] = {
            "average_metric_value": -float("inf"),
            "patience": cfg.train.early_stopping.patience,
        }
        self.wait = 0
        self.epoch_start_time: float = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        train_time = time.time() - self.epoch_start_time
        train_loss = logs.get("loss", 0.0) if logs else 0.0

        # Run validation
        val_metrics: dict[str, float] | None = None
        val_time = None
        is_best = False

        if self.cfg.eval.fast_evaluation:
            eval_start = time.time()
            val_metrics = self.model.fast_evaluate(
                user_hist_dataloader=self.dataset_provider.user_history_dataloader(
                    mode="val", batch_size=self.cfg.eval.batch_size
                ),
                impression_iterator=self.dataset_provider.impression_dataloader(
                    mode="val"
                ),
                news_dataloader=self.dataset_provider.news_dataloader(
                    batch_size=self.cfg.eval.batch_size
                ),
                metrics_calculator=self.metrics_engine,
                progress=self.progress,
                mode="val",
                epoch=epoch,
                int_to_news_id_map=self.dataset_provider.get_int_to_news_id_map(),
            )
            val_time = time.time() - eval_start

            # Best tracking
            main = ["auc", "mrr", "ndcg@5", "ndcg@10"]
            vals = [float(val_metrics[m]) for m in main if m in val_metrics]
            avg = sum(vals) / len(vals) if vals else 0.0

            if avg > self.best_epoch_metrics["average_metric_value"]:
                is_best = True
                self.wait = 0
                self.best_epoch_metrics = {
                    "epoch_number": epoch + 1,
                    "average_metric_value": avg,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                self.model.save_weights(str(self.best_model_path))
                save_model_config(self.cfg, self.best_model_path)
            else:
                self.wait += 1

            # WandB
            if wandb.run and self.wandb_history is not None:
                payload = {f"val/{k}": v for k, v in val_metrics.items()}
                payload["train/loss"] = train_loss
                log_metrics_to_wandb_fn(payload, epoch + 1, self.wandb_history)

        # Shared log format
        log_epoch_end(
            epoch=epoch + 1,
            num_epochs=self.cfg.train.num_epochs,
            train_loss=train_loss,
            train_time=train_time,
            val_metrics=val_metrics,
            val_time=val_time,
            is_best=is_best,
        )

        # Early stopping
        if self.wait >= self.cfg.train.early_stopping.patience:
            log_early_stopping(epoch + 1, self.cfg.train.early_stopping.patience)
            self.model.stop_training = True


class RichProgressCallback(keras.callbacks.Callback):
    """Rich progress bar for Keras training."""

    def __init__(
        self, progress: Progress, num_epochs: int, steps_per_epoch: int | None = None
    ):
        super().__init__()
        self.progress = progress
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.overall_task = None
        self.epoch_task = None
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        self.overall_task = self.progress.add_task("Training", total=self.num_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_task = self.progress.add_task(
            f"Epoch {epoch + 1}/{self.num_epochs}",
            total=self.steps_per_epoch,
        )

    def on_batch_end(self, batch, logs=None):
        if self.epoch_task is not None:
            loss = logs.get("loss", 0.0) if logs else 0.0
            self.progress.update(
                self.epoch_task,
                advance=1,
                description=f"Epoch {self.current_epoch + 1}/{self.num_epochs} (loss={loss:.4f})",
            )

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_task is not None:
            self.progress.remove_task(self.epoch_task)
            self.epoch_task = None
        if self.overall_task is not None:
            self.progress.update(self.overall_task, advance=1)

    def on_train_end(self, logs=None):
        if self.overall_task is not None:
            self.progress.remove_task(self.overall_task)


# =============================================================================
# Helpers
# =============================================================================


def _setup_training_directories(output_directory: Path, model_name: str):
    models_dir = output_directory / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_directory / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f"{model_name}_best.weights.h5"
    last_path = models_dir / f"{model_name}_last.weights.h5"
    return best_path, last_path, predictions_dir


# =============================================================================
# Main Orchestrator
# =============================================================================


def training_loop_orchestrator(
    model: keras.Model,
    dataset_provider,
    cfg: DictConfig,
    custom_metrics_engine: NewsRecommenderMetrics,
    progress_bar_manager: Progress,
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """Train a Keras model using model.fit() with evaluation callbacks."""

    experiment_start = time.time()

    best_model_path, last_model_path, predictions_save_dir = (
        _setup_training_directories(output_directory, model.name)
    )

    wandb_history = {} if cfg.logging.enable_wandb else None
    framework = cfg.get("framework", "keras")
    log_training_start(cfg.model_name, framework, cfg.train.num_epochs)

    # JIT warmup
    try:
        from src.frameworks.keras.utils import warmup_jit_compilation

        model_name = (
            cfg.spec.model.name.lower()
            if hasattr(cfg, "spec") and cfg.spec is not None
            else cfg.model._target_.split(".")[-1].lower()
        )
        sample_dl = dataset_provider.train_dataloader(
            batch_size=min(2, cfg.train.batch_size), model_name=model_name
        )
        first_batch = next(iter(sample_dl), None)
        if first_batch is not None:
            warmup_jit_compilation(model, first_batch)
            console.log("JIT warmup completed.")
    except Exception as e:
        console.log(f"[yellow]JIT warmup skipped: {e}[/yellow]")

    # Initial eval
    initial_metrics = {}
    if cfg.eval.run_initial_eval:
        initial_metrics = _run_initial_validation(
            model, dataset_provider, custom_metrics_engine, progress_bar_manager, cfg
        )

    # Train dataset
    model_name = (
        cfg.spec.model.name.lower()
        if hasattr(cfg, "spec") and cfg.spec is not None
        else cfg.model._target_.split(".")[-1].lower()
    )
    train_dataset = dataset_provider.train_dataloader(
        batch_size=cfg.train.batch_size, model_name=model_name
    )

    # Steps per epoch
    steps_per_epoch = len(train_dataset) if hasattr(train_dataset, "__len__") else None

    # Callbacks
    eval_callback = EvaluationCallback(
        dataset_provider=dataset_provider,
        metrics_engine=custom_metrics_engine,
        cfg=cfg,
        progress=progress_bar_manager,
        best_model_path=best_model_path,
        wandb_history=wandb_history,
    )

    callbacks = [
        RichProgressCallback(
            progress_bar_manager, cfg.train.num_epochs, steps_per_epoch
        ),
        eval_callback,
        keras.callbacks.ModelCheckpoint(
            filepath=str(last_model_path),
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        ),
    ]

    # Verify compiled
    if not model.compiled:
        raise RuntimeError("Model is not compiled.")

    # Train
    try:
        model.fit(
            train_dataset,
            epochs=cfg.train.num_epochs,
            callbacks=callbacks,
            verbose=0,
        )
    except KeyboardInterrupt:
        console.log("[bold red]Training interrupted.[/bold red]")
    except Exception as e:
        console.log(f"[bold red]Training failed: {e}[/bold red]")
        raise

    best_epoch_metrics = eval_callback.best_epoch_metrics

    # Test
    test_metrics = None
    if cfg.eval.run_test_after_training:
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
        if test_metrics:
            log_test_results(test_metrics)

    total_time = time.time() - experiment_start
    log_training_complete(cfg.model_name, framework, total_time)

    # Save summary
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
                **{f"best/{k}": v for k, v in best_epoch_metrics.items()},
            }
        )
        wandb.finish()

    return best_epoch_metrics, test_metrics
