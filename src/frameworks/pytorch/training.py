"""Manual training loop for PyTorch news recommendation models.

Provides ``training_loop(cfg)`` as the main entry point, mirroring the
Keras ``model.fit()`` workflow with:
- Forward -> loss -> backward -> optimizer.step()
- Core metrics for validation (via ``run_fast_evaluation``)
- Early stopping and model checkpointing
- Rich progress bars
- Optional WandB logging
- Automatic device handling (CUDA / MPS / CPU)
"""

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from src.core.io.logging import log_early_stopping, log_epoch_end

from .device import setup_device
from .evaluation import run_fast_evaluation
from .losses import CategoricalCrossEntropyLoss

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


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def training_loop(
    cfg: Any,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataset_provider: Any = None,
    metrics_engine: Any = None,
    *,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    gpu_ids: str = "",
    patience: int = 3,
    checkpoint_dir: str | None = None,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    loss_fn: nn.Module | None = None,
    int_to_news_id_map: dict | None = None,
) -> dict[str, Any]:
    """Run a full training loop.

    Args:
        cfg: Global configuration object (passed through to evaluation).
        model: PyTorch ``BaseModel`` subclass.
        train_dataloader: DataLoader yielding ``(features_dict, labels)``.
        val_dataset_provider: Provides validation dataloaders for fast eval.
        metrics_engine: Core metrics calculator.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularisation.
        gpu_ids: Comma-separated GPU IDs (empty = auto-detect).
        patience: Early stopping patience (epochs without improvement).
        checkpoint_dir: Directory to save best model checkpoint.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: WandB project name.
        wandb_run_name: WandB run name.
        loss_fn: Loss function (defaults to CategoricalCrossEntropyLoss).
        int_to_news_id_map: Mapping for news IDs during evaluation.

    Returns:
        Dictionary with ``best_metrics``, ``best_epoch``, ``history``.
    """
    # ---- Device ----
    device = setup_device(gpu_ids)
    model = model.to(device)

    # ---- Loss / Optimiser ----
    if loss_fn is None:
        loss_fn = CategoricalCrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # ---- WandB ----
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(cfg) if hasattr(cfg, "__dict__") else {},
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # ---- Tracking ----
    best_val_metric = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: dict[str, list] = {"train_loss": [], "val_metrics": []}

    # ---- Main loop ----
    with _build_progress() as progress:
        epoch_task = progress.add_task("Epochs", total=epochs)

        for epoch in range(1, epochs + 1):
            # ---------- Training ----------
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            num_batches = 0

            batch_task = progress.add_task(
                f"Epoch {epoch}/{epochs} - training", total=len(train_dataloader)
            )

            for batch_features, batch_labels in train_dataloader:
                batch_features, batch_labels = _move_batch_to_device(
                    batch_features, batch_labels, device
                )

                optimizer.zero_grad()
                predictions = model(batch_features, training=True)
                loss = loss_fn(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

                progress.update(batch_task, advance=1)

            progress.remove_task(batch_task)

            avg_train_loss = running_loss / max(num_batches, 1)
            epoch_train_time = time.time() - epoch_start
            history["train_loss"].append(avg_train_loss)

            # ---------- Validation ----------
            val_metrics: dict[str, float] = {}
            val_time = None
            is_best = False

            if val_dataset_provider is not None and metrics_engine is not None:
                eval_start = time.time()
                val_metrics = run_fast_evaluation(
                    model=model,
                    dataset_provider=val_dataset_provider,
                    metrics_engine=metrics_engine,
                    progress=progress,
                    cfg=cfg,
                    mode="validate",
                    epoch=epoch,
                    int_to_news_id_map=int_to_news_id_map,
                )
                val_time = time.time() - eval_start
                history["val_metrics"].append(val_metrics)

                # Early stopping / checkpointing
                monitor_value = val_metrics.get("auc", val_metrics.get("loss", 0.0))
                if "auc" not in val_metrics:
                    monitor_value = -monitor_value

                if monitor_value > best_val_metric:
                    best_val_metric = monitor_value
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    is_best = True

                    if checkpoint_dir:
                        ckpt_path = Path(checkpoint_dir) / "best_model.pt"
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), ckpt_path)
                else:
                    epochs_without_improvement += 1

            # Log epoch (shared format)
            log_epoch_end(
                epoch=epoch,
                num_epochs=epochs,
                train_loss=avg_train_loss,
                train_time=epoch_train_time,
                val_metrics=val_metrics or None,
                val_time=val_time,
                is_best=is_best,
            )

            if epochs_without_improvement >= patience:
                log_early_stopping(epoch, patience)
                break

            # WandB
            if use_wandb:
                log_dict = {"epoch": epoch, "train_loss": avg_train_loss}
                for k, v in val_metrics.items():
                    log_dict[f"val_{k}"] = v
                wandb.log(log_dict)

            progress.update(epoch_task, advance=1)

    # ---- Final ----
    if use_wandb:
        wandb.finish()

    return {
        "best_metrics": history["val_metrics"][best_epoch - 1]
        if best_epoch > 0 and history["val_metrics"]
        else {},
        "best_epoch": best_epoch,
        "history": history,
    }
