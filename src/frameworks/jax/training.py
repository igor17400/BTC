"""Functional training loop for Flax NNX news recommendation models.

Key design choices:
* ``@nnx.jit``-compiled ``train_step`` for maximum throughput.
* ``optax`` for gradient-based optimisation via ``nnx.Optimizer``.
* Early stopping and optional WandB logging.
* Rich progress bars for real-time feedback.
"""

from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx
from rich.console import Console
from safetensors.numpy import save_file as save_safetensors

from src.core.io.logging import (
    log_early_stopping,
    log_epoch_end,
)
from src.core.io.progress import ProgressManager, create_progress

logger = logging.getLogger(__name__)

# Flax NNX's ``Optimizer.update`` signature changed in 0.10.5:
#   - older:  update(self, grads, **kwargs)
#   - newer:  update(self, model, grads, **kwargs)
# Detect at import time so the hot path inside ``@nnx.jit`` stays clean.
_OPT_UPDATE_TAKES_MODEL = "model" in inspect.signature(nnx.Optimizer.update).parameters
console = Console()


# ---------------------------------------------------------------------------
# JIT-compiled train step
# ---------------------------------------------------------------------------


def make_train_step(loss_fn, get_aux_loss=None, use_jit: bool = True):
    """Create a training step that uses the given loss function.

    Args:
        loss_fn: Callable ``(y_true, y_pred) -> scalar`` loss function.
        get_aux_loss: Optional callable ``(model) -> scalar`` that returns
            an auxiliary loss term (e.g. CROWN category prediction).
        use_jit: Whether to JIT-compile the step.  Set ``False`` for
            models with variable-sized inputs (e.g. GLORY subgraphs)
            where shapes change every batch.

    Returns:
        A ``train_step(model, optimizer, features, labels)`` function,
        optionally JIT-compiled.
    """

    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch_features: dict[str, jnp.ndarray],
        batch_labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single training step."""

        def _loss(model):
            preds = model(batch_features, training=True)
            total = loss_fn(batch_labels, preds)
            if get_aux_loss is not None:
                total = total + get_aux_loss(model)
            return total

        loss, grads = nnx.value_and_grad(_loss)(model)
        if _OPT_UPDATE_TAKES_MODEL:
            optimizer.update(model, grads)
        else:
            optimizer.update(grads)
        return loss

    if use_jit:
        train_step = nnx.jit(train_step)

    return train_step


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------


def warmup_jit(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_step,
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
    gradient_clip_norm: float = 0.0,
    early_stopping_patience: int = 5,
    early_stopping_min_improvement: float = 0.01,
    warmup_ratio: float = 0.0,
    loss_fn=None,
    get_aux_loss=None,
    use_jit: bool = True,
    # Optional evaluation hooks
    eval_fn=None,
    eval_kwargs: dict[str, Any] | None = None,
    # Logging
    progress: ProgressManager | None = None,
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
        loss_fn: Loss callable ``(y_true, y_pred) -> scalar``. Defaults to
            ``categorical_cross_entropy(from_logits=True)``.
        eval_fn: Optional callable ``(model, **eval_kwargs) -> metrics_dict``
            to run at the end of each epoch.
        eval_kwargs: Keyword arguments forwarded to *eval_fn*.
        progress: Progress bar manager.
        enable_wandb: Whether to log metrics to Weights & Biases.
        save_dir: Directory for saving model checkpoints.

    Returns:
        Dictionary with ``"best_epoch_metrics"`` and timing information.
    """
    # ---- Loss function ---------------------------------------------------
    if loss_fn is None:
        from .losses import categorical_cross_entropy

        loss_fn = categorical_cross_entropy

    train_step = make_train_step(loss_fn, get_aux_loss=get_aux_loss, use_jit=use_jit)

    # ---- Optimiser -------------------------------------------------------
    components = []
    if gradient_clip_norm > 0:
        components.append(optax.clip_by_global_norm(gradient_clip_norm))

    # LR schedule: optional linear warmup then constant
    if warmup_ratio > 0.0 and hasattr(train_dataloader, "__len__"):
        total_steps = num_epochs * len(train_dataloader)
        warmup_steps = int(total_steps * warmup_ratio)
        schedule = optax.warmup_constant_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
        )
        if weight_decay > 0:
            components.append(optax.adamw(schedule, weight_decay=weight_decay))
        else:
            components.append(optax.adam(schedule))
        logger.info(
            f"LR warmup: {warmup_steps} steps ({warmup_ratio * 100:.0f}% of {total_steps})"
        )
    else:
        if weight_decay > 0:
            components.append(optax.adamw(learning_rate, weight_decay=weight_decay))
        else:
            components.append(optax.adam(learning_rate))

    tx = optax.chain(*components) if len(components) > 1 else components[0]

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # ---- JIT warmup ------------------------------------------------------
    if use_jit:
        try:
            first_batch = next(iter(train_dataloader), None)
            if first_batch is not None:
                warmup_jit(model, optimizer, train_step, first_batch)
        except Exception as exc:
            logger.warning("JIT warmup skipped: %s", exc)

    # ---- Early stopping --------------------------------------------------
    stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_improvement=early_stopping_min_improvement,
    )

    # ---- WandB -----------------------------------------------------------
    wandb_run = wandb.run if enable_wandb else None

    # ---- Progress --------------------------------------------------------
    own_progress = False
    if progress is None:
        progress = create_progress()
        progress.start()
        own_progress = True

    # ---- Best checkpoint (in-memory) ----------------------------------------
    best_state = None

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
                val_metrics = eval_fn(model, epoch=epoch, **(eval_kwargs or {}))
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
                    # Deep-copy best model state so later NaN corruption
                    # doesn't propagate into the saved checkpoint.
                    best_state = jax.tree.map(lambda x: x.copy(), nnx.state(model))

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

    # Restore best model state so test eval uses the best weights.
    if best_state is not None:
        nnx.update(model, best_state)
        logger.info(
            "Restored best model state from epoch %d",
            best_metrics.get("epoch_number", "?"),
        )
        # Save best weights to disk in safetensors format (HF-compatible).
        if save_dir:
            ckpt_dir = Path(save_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "model.safetensors"

            # Flatten the state dict to dot-separated keys and convert
            # all arrays to numpy.  PRNGKey arrays are skipped (not
            # model weights).
            flat = {}
            for key_path, leaf in jax.tree_util.tree_leaves_with_path(best_state):
                name = ".".join(str(k) for k in key_path)
                if hasattr(leaf, "dtype") and jnp.issubdtype(
                    leaf.dtype, jax.dtypes.prng_key
                ):
                    continue  # skip RNG state
                flat[name] = np.asarray(leaf)
            save_safetensors(flat, str(ckpt_path))
            logger.info("Saved best model weights to %s", ckpt_path)

    timing["total_training_time"] = time.time() - experiment_start
    best_metrics["timing"] = timing

    return best_metrics
