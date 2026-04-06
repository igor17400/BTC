"""Functional training loop for Flax NNX news recommendation models.

Key design choices:
* ``@nnx.jit``-compiled ``train_step`` for maximum throughput.
* ``optax`` for gradient-based optimisation via ``nnx.Optimizer``.
* Early stopping and optional WandB logging.
* Rich progress bars for real-time feedback.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from rich.console import Console
from rich.progress import Progress

from src.core.io.logging import (
    log_early_stopping,
    log_epoch_end,
)

from .losses import categorical_cross_entropy

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# JIT-compiled train step
# ---------------------------------------------------------------------------


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_features: dict[str, jnp.ndarray],
    batch_labels: jnp.ndarray,
) -> jnp.ndarray:
    """Single JIT-compiled training step.

    Args:
        model: Flax NNX model (mutable state is updated in-place).
        optimizer: ``nnx.Optimizer`` wrapping an ``optax`` optimiser.
        batch_features: Dictionary of input JAX arrays.
        batch_labels: Ground-truth labels ``(B, C)``.

    Returns:
        Scalar loss value for this step.
    """

    def loss_fn(model):
        preds = model(batch_features, training=True)
        return categorical_cross_entropy(batch_labels, preds)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------


def warmup_jit(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    sample_batch: tuple[dict[str, jnp.ndarray], jnp.ndarray],
) -> None:
    """Run a single forward + backward pass to trigger XLA compilation.

    This avoids a slow first training step.
    """
    console.log("Warming up JIT compilation...")
    features, labels = sample_batch
    try:
        _ = train_step(model, optimizer, features, labels)
        jax.block_until_ready(jax.device_put(0))
        console.log("[green]JIT warmup completed.[/green]")
    except Exception as exc:
        console.log(
            f"[yellow]JIT warmup encountered a non-critical error: {exc}[/yellow]"
        )


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Simple early-stopping tracker."""

    def __init__(self, patience: int = 5, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_metric: float = -float("inf")
        self.wait: int = 0
        self.best_epoch: int = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Return ``True`` if training should stop."""
        if metric > self.best_metric + self.min_improvement:
            self.best_metric = metric
            self.wait = 0
            self.best_epoch = epoch
            return False
        self.wait += 1
        return self.wait >= self.patience


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def training_loop(
    model: nnx.Module,
    train_dataloader,
    *,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 5,
    # Optional evaluation hooks
    eval_fn=None,
    eval_kwargs: dict[str, Any] | None = None,
    # Logging
    progress: Progress | None = None,
    enable_wandb: bool = False,
    # Saving
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Main Flax NNX training entry point.

    Args:
        model: A ``BaseModel`` subclass (NRMS / NAML / LSTUR).
        train_dataloader: Iterable yielding ``(features_dict, labels)`` per
            batch.  It is re-iterated every epoch.
        num_epochs: Maximum number of training epochs.
        learning_rate: Peak learning rate for the optimiser.
        weight_decay: L2 weight-decay coefficient (0 disables).
        early_stopping_patience: Number of epochs without improvement
            before stopping.
        eval_fn: Optional callable ``(model, **eval_kwargs) -> metrics_dict``
            to run at the end of each epoch.
        eval_kwargs: Keyword arguments forwarded to *eval_fn*.
        progress: Rich ``Progress`` bar manager.
        enable_wandb: Whether to log metrics to Weights & Biases.
        save_dir: Directory for saving model checkpoints.

    Returns:
        Dictionary with ``"best_epoch_metrics"`` and timing information.
    """
    # ---- Optimiser -------------------------------------------------------
    if weight_decay > 0:
        tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    else:
        tx = optax.adam(learning_rate)

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # ---- JIT warmup ------------------------------------------------------
    try:
        first_batch = next(iter(train_dataloader), None)
        if first_batch is not None:
            warmup_jit(model, optimizer, first_batch)
    except Exception as exc:
        logger.warning("JIT warmup skipped: %s", exc)

    # ---- Early stopping --------------------------------------------------
    stopper = EarlyStopping(patience=early_stopping_patience)

    # ---- WandB -----------------------------------------------------------
    wandb_run = wandb.run if enable_wandb else None

    # ---- Progress --------------------------------------------------------
    own_progress = False
    if progress is None:
        progress = Progress()
        progress.start()
        own_progress = True

    # ---- Timing ----------------------------------------------------------
    timing: dict[str, Any] = {
        "epoch_training_times": [],
        "epoch_validation_times": [],
    }
    experiment_start = time.time()

    best_metrics: dict[str, Any] = {"average_metric_value": -float("inf")}

    # ---- Epoch loop ------------------------------------------------------
    overall_task = progress.add_task("Training", total=num_epochs)

    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            # Batch loop
            batch_task = progress.add_task(
                f"Epoch {epoch + 1}/{num_epochs}",
                total=len(train_dataloader)
                if hasattr(train_dataloader, "__len__")
                else None,
                visible=True,
            )

            for batch_features, batch_labels in train_dataloader:
                loss = train_step(model, optimizer, batch_features, batch_labels)
                epoch_loss += float(loss)
                num_batches += 1
                progress.update(
                    batch_task,
                    advance=1,
                    description=(
                        f"Epoch {epoch + 1}/{num_epochs} "
                        f"(Loss: {epoch_loss / num_batches:.4f})"
                    ),
                )

            progress.remove_task(batch_task)

            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_train_time = time.time() - epoch_start
            timing["epoch_training_times"].append(epoch_train_time)

            # ---- Evaluation ----------------------------------------------
            val_metrics: dict[str, float] | None = None
            eval_time = None
            is_best = False

            if eval_fn is not None:
                eval_start = time.time()
                val_metrics = eval_fn(model, **(eval_kwargs or {}))
                eval_time = time.time() - eval_start
                timing["epoch_validation_times"].append(eval_time)

                # Best tracking
                main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
                vals = [val_metrics[m] for m in main_metrics if m in val_metrics]
                avg_metric = sum(vals) / len(vals) if vals else 0.0

                if avg_metric > best_metrics["average_metric_value"]:
                    is_best = True
                    best_metrics = {
                        "epoch_number": epoch + 1,
                        "train_loss": avg_loss,
                        "average_metric_value": avg_metric,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }

            # Log epoch (shared format)
            log_epoch_end(
                epoch=epoch + 1,
                num_epochs=num_epochs,
                train_loss=avg_loss,
                train_time=epoch_train_time,
                val_metrics=val_metrics,
                val_time=eval_time,
                is_best=is_best,
            )

            # WandB
            if wandb_run is not None:
                log_data = {"train/loss": avg_loss, "epoch": epoch + 1}
                if val_metrics:
                    log_data.update({f"val/{k}": v for k, v in val_metrics.items()})
                wandb.log(log_data)

            # Early stopping
            if val_metrics and stopper.step(avg_metric, epoch + 1):
                log_early_stopping(epoch + 1, early_stopping_patience)
                break

            progress.update(overall_task, advance=1)

    finally:
        progress.remove_task(overall_task)
        if own_progress:
            progress.stop()

    timing["total_training_time"] = time.time() - experiment_start
    best_metrics["timing"] = timing

    return best_metrics
