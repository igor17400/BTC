"""Manual training loop for PyTorch news recommendation models.

Provides ``training_loop`` as the main entry point with:
- Forward -> loss -> backward -> optimizer.step()
- Pluggable eval_fn for validation (via ``fast_evaluate``)
- Early stopping and model checkpointing
- Rich progress bars
- Optional WandB logging
"""

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader

from src.core.io.logging import log_early_stopping, log_epoch_end
from src.core.io.progress import create_progress

from .device import setup_device

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _move_batch_to_device(
    batch_features: dict[str, torch.Tensor],
    batch_labels: torch.Tensor,
    device: torch.device,
):
    """Move all tensors in a batch dict to *device*."""
    features = {k: v.to(device, non_blocking=True) for k, v in batch_features.items()}
    labels = batch_labels.to(device, non_blocking=True)
    return features, labels


def _build_progress():
    return create_progress(columns="training", expand=True)


class EarlyStopping:
    """Mirrors the JAX ``EarlyStopping`` (see ``jax/training.py``).

    Decoupled from best-checkpoint tracking: ``step`` returns ``True`` once
    ``patience`` consecutive epochs fail to beat ``best + min_improvement``.
    """

    def __init__(self, patience: int = 5, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_metric: float = -float("inf")
        self.wait: int = 0

    def step(self, metric: float) -> bool:
        if metric > self.best_metric + self.min_improvement:
            self.best_metric = metric
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def training_loop(
    model: nn.Module,
    train_dataloader: DataLoader,
    *,
    eval_fn=None,
    cfg: Any = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 3,
    early_stopping_min_improvement: float = 0.01,
    enable_wandb: bool = False,
    save_dir: str | Path | None = None,
    gpu_ids: list[int] | None = None,
    loss_fn: nn.Module | None = None,
    get_aux_loss=None,
) -> dict[str, Any]:
    """Run a full training loop.

    Isomorphic with ``training_loop`` in Keras and JAX.

    Args:
        model: PyTorch ``BaseModel`` subclass.
        train_dataloader: DataLoader yielding ``(features_dict, labels)``.
        eval_fn: Optional callable ``(model) -> metrics_dict``
            to run at the end of each epoch.
        cfg: Global configuration object (optional, for extra context).
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularisation.
        early_stopping_patience: Epochs without improvement before stopping.
        enable_wandb: Whether to log to Weights & Biases.
        save_dir: Directory for saving model checkpoints.
        gpu_ids: GPU IDs to use.
        loss_fn: Loss function (defaults to CategoricalCrossEntropyLoss).
        get_aux_loss: Optional callable ``(model) -> scalar`` auxiliary loss
            (e.g. CROWN category prediction, MINER disagreement).

    Returns:
        Dictionary with ``best_epoch_metrics`` and timing information.
    """
    # ---- Device ----
    device = setup_device(gpu_ids)
    model = model.to(device)

    # ---- Loss / Optimiser ----
    if loss_fn is None:
        from .losses import CategoricalCrossEntropyLoss

        loss_fn = CategoricalCrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # ---- LR warmup schedule ----
    warmup_ratio = 0.0
    if cfg and hasattr(cfg, "train"):
        warmup_ratio = getattr(cfg.train, "warmup_ratio", 0.0)
    scheduler = None
    if warmup_ratio > 0.0:
        total_steps = num_epochs * len(train_dataloader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
            ),
        )

    # ---- WandB ----
    wandb_run = wandb.run if enable_wandb else None

    # ---- Tracking ----
    best_metrics: dict[str, Any] = {"average_metric_value": -float("inf")}
    stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_improvement=early_stopping_min_improvement,
    )
    timing: dict[str, Any] = {
        "epoch_training_times": [],
        "epoch_validation_times": [],
    }
    experiment_start = time.time()

    # ---- Mixed precision setup (once, outside the epoch loop) ----
    # ``cfg.device.precision`` selects the autocast dtype:
    #   - ``"bfloat16"`` / ``"bf16"`` — bf16 autocast, no GradScaler
    #     (bf16 has fp32-equivalent range, scaler would be a no-op).
    #   - ``"float16"`` / ``"fp16"`` / ``"half"`` — fp16 autocast +
    #     ``GradScaler`` for loss scaling. Matches the reference GLORY
    #     (and most production training) on Ampere/Ada GPUs.
    amp_dtype: torch.dtype | None = None
    if cfg and hasattr(cfg, "device") and hasattr(cfg.device, "precision"):
        precision = str(cfg.device.precision).lower()
        if precision in ("bfloat16", "bf16"):
            amp_dtype = torch.bfloat16
        elif precision in ("float16", "fp16", "half"):
            amp_dtype = torch.float16
    use_autocast = amp_dtype is not None and device.type == "cuda"
    use_scaler = use_autocast and amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # ---- Main loop ----
    with _build_progress() as progress:
        epoch_task = progress.add_task("Epochs", total=num_epochs)

        for epoch in range(1, num_epochs + 1):
            # ---------- Training ----------
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            num_batches = 0

            batch_task = progress.add_task(
                f"Epoch {epoch}/{num_epochs} - training", total=len(train_dataloader)
            )

            grad_accum_steps = 1
            if cfg and hasattr(cfg, "train") and hasattr(cfg.train, "grad_accum_steps"):
                grad_accum_steps = max(int(cfg.train.grad_accum_steps), 1)
            total_micro = len(train_dataloader)

            optimizer.zero_grad(set_to_none=True)
            for micro_idx, (batch_features, batch_labels) in enumerate(
                train_dataloader
            ):
                batch_features, batch_labels = _move_batch_to_device(
                    batch_features, batch_labels, device
                )

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if use_autocast
                    else nullcontext()
                )
                with autocast_ctx:
                    predictions = model(batch_features, training=True)
                    loss = loss_fn(predictions, batch_labels)
                    if get_aux_loss is not None:
                        loss = loss + get_aux_loss(model)

                display_loss = loss.detach()
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_step = (micro_idx + 1) % grad_accum_steps == 0 or (
                    micro_idx + 1
                ) == total_micro
                if is_step:
                    if (
                        cfg
                        and hasattr(cfg, "train")
                        and hasattr(cfg.train, "gradient_clip_val")
                    ):
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.train.gradient_clip_val
                        )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += display_loss.item()
                num_batches += 1

                progress.update(batch_task, advance=1)

            progress.remove_task(batch_task)

            avg_train_loss = running_loss / max(num_batches, 1)
            epoch_train_time = time.time() - epoch_start
            timing["epoch_training_times"].append(epoch_train_time)

            # ---------- Validation ----------
            val_metrics: dict[str, float] | None = None
            val_time = None
            is_best = False

            if eval_fn is not None:
                eval_start = time.time()
                val_metrics = eval_fn(model, epoch=epoch)
                val_time = time.time() - eval_start
                timing["epoch_validation_times"].append(val_time)

                # Best tracking
                main_metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
                vals = [val_metrics[m] for m in main_metrics if m in val_metrics]
                avg_metric = sum(vals) / len(vals) if vals else 0.0

                if avg_metric > best_metrics["average_metric_value"]:
                    is_best = True
                    best_metrics = {
                        "epoch_number": epoch,
                        "train_loss": avg_train_loss,
                        "average_metric_value": avg_metric,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }

                    if save_dir:
                        ckpt_path = Path(save_dir) / "model.safetensors"
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        flat = {
                            k: v.detach().cpu().contiguous()
                            for k, v in model.state_dict().items()
                        }
                        save_safetensors(flat, str(ckpt_path))

            # Log epoch (shared format)
            log_epoch_end(
                epoch=epoch,
                num_epochs=num_epochs,
                train_loss=avg_train_loss,
                train_time=epoch_train_time,
                val_metrics=val_metrics,
                val_time=val_time,
                is_best=is_best,
            )

            # Early stopping
            if val_metrics is not None and stopper.step(avg_metric):
                log_early_stopping(epoch, early_stopping_patience)
                break

            # WandB
            if wandb_run is not None:
                log_data = {"train/loss": avg_train_loss, "epoch": epoch}
                if val_metrics:
                    log_data.update({f"val/{k}": v for k, v in val_metrics.items()})
                wandb.log(log_data)

            progress.update(epoch_task, advance=1)

    # ---- Final ----
    timing["total_training_time"] = time.time() - experiment_start
    best_metrics["timing"] = timing

    return best_metrics
