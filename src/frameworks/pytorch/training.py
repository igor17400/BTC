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
from src.core.io.saving import cleanup_val_staging, promote_best_val_predictions
from src.core.io.timing import PhaseStats

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
    best_metric_key = "auc"
    if cfg and hasattr(cfg, "train"):
        best_metric_key = str(getattr(cfg.train, "best_metric", "auc")).lower()
    best_metrics: dict[str, Any] = {"average_metric_value": -float("inf")}
    stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_improvement=early_stopping_min_improvement,
    )

    def _score_for_tracking(metrics: dict[str, float]) -> float:
        """Pick the scalar used for best-checkpoint + early-stopping."""
        if best_metric_key == "avg":
            keys = ["auc", "mrr", "ndcg@5", "ndcg@10"]
            vs = [metrics[k] for k in keys if k in metrics]
            return sum(vs) / len(vs) if vs else 0.0
        # default — and reference's behavior — is raw AUC
        return float(metrics.get(best_metric_key, 0.0))

    timing: dict[str, Any] = {
        "train_epochs": [],
        "val_epochs": [],
        "time_to_first_step_seconds": None,
    }
    experiment_start = time.time()

    # ---- Step-based validation config (mirrors reference GLORY's
    # ``val_steps`` + ``val_skip_epochs``). When ``cfg.train.val_steps``
    # is set, validation runs every N optimizer steps instead of (only)
    # at epoch boundaries. ``val_skip_epochs`` defers step-based vals
    # until ``val_skip_epochs * batches_per_epoch`` steps have elapsed.
    val_steps_cfg: int | None = None
    val_skip_epochs_cfg: float = 0.0
    if cfg and hasattr(cfg, "train"):
        v = getattr(cfg.train, "val_steps", None)
        if v is not None:
            try:
                val_steps_cfg = int(v) if int(v) > 0 else None
            except (TypeError, ValueError):
                val_steps_cfg = None
        val_skip_epochs_cfg = float(getattr(cfg.train, "val_skip_epochs", 0.0))

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
            samples_seen = 0

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
                    # Reference GLORY (main.py:50-55) only steps the LR
                    # scheduler when AMP's GradScaler did NOT downshift —
                    # i.e. fp16 overflow did not occur in this micro-step.
                    # During warmup with fp16, the scaler can downshift
                    # tens of times in the first ~500 steps; NewsReX
                    # without this guard burns those steps off the warmup
                    # schedule even though optimizer.step() was a no-op.
                    if scaler is not None:
                        scaler.step(optimizer)
                        old_scale = scaler.get_scale()
                        scaler.update()
                        new_scale = scaler.get_scale()
                        scaler_step_succeeded = new_scale >= old_scale
                    else:
                        optimizer.step()
                        scaler_step_succeeded = True

                    if scheduler is not None and scaler_step_succeeded:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # First completed gradient-update step in the whole
                # run captures the "time-to-first-step" metric.
                if is_step and timing["time_to_first_step_seconds"] is None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    timing["time_to_first_step_seconds"] = (
                        time.time() - experiment_start
                    )

                running_loss += display_loss.item()
                num_batches += 1
                samples_seen += int(batch_labels.shape[0])

                progress.update(batch_task, advance=1)

                # ---------- Step-based validation ----------
                # When val_steps_cfg is set, run validation every N
                # optimizer steps (mirrors reference GLORY's main loop:
                # reference_codes/GLORY/src/main.py:69-86).
                if val_steps_cfg is not None and is_step and eval_fn is not None:
                    global_step_count = (epoch - 1) * total_micro + (micro_idx + 1)
                    skip_until = int(val_skip_epochs_cfg * total_micro)
                    if (
                        global_step_count > skip_until
                        and global_step_count % val_steps_cfg == 0
                    ):
                        model.eval()
                        step_eval_start = time.time()
                        step_val_metrics = eval_fn(model, epoch=epoch)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        step_val_time = time.time() - step_eval_start
                        model.train()

                        n_imps_step = (
                            int(step_val_metrics.get("_num_impressions", 0))
                            if isinstance(step_val_metrics, dict)
                            else 0
                        )
                        timing["val_epochs"].append(
                            PhaseStats(
                                wall_seconds=step_val_time,
                                n_samples=n_imps_step or None,
                                throughput_samples_per_s=(
                                    n_imps_step / step_val_time
                                    if (n_imps_step and step_val_time > 0)
                                    else None
                                ),
                            ).to_dict()
                        )

                        avg_metric_step = _score_for_tracking(step_val_metrics)

                        is_best_step = (
                            avg_metric_step > best_metrics["average_metric_value"]
                        )
                        if is_best_step:
                            best_metrics = {
                                "epoch_number": epoch,
                                "step_number": global_step_count,
                                "train_loss": running_loss / max(num_batches, 1),
                                "average_metric_value": avg_metric_step,
                                **{f"val_{k}": v for k, v in step_val_metrics.items()},
                            }
                            if save_dir:
                                ckpt_path = Path(save_dir) / "model.safetensors"
                                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                                flat = {
                                    k: v.detach().cpu().contiguous()
                                    for k, v in model.state_dict().items()
                                }
                                save_safetensors(flat, str(ckpt_path))
                                promote_best_val_predictions(
                                    Path(save_dir).parent / "predictions"
                                )

                        # Inline log so progress doesn't swallow it.
                        print(
                            f"[step val] epoch={epoch} step={global_step_count} "
                            f"auc={step_val_metrics.get('auc', float('nan')):.4f} "
                            f"mrr={step_val_metrics.get('mrr', float('nan')):.4f} "
                            f"ndcg@10={step_val_metrics.get('ndcg@10', float('nan')):.4f} "
                            f"{'*BEST*' if is_best_step else ''} "
                            f"best={best_metrics['average_metric_value']:.4f}",
                            flush=True,
                        )

                        if wandb_run is not None:
                            wandb_log = {
                                f"val/{k}": v
                                for k, v in step_val_metrics.items()
                                if not k.startswith("_")
                            }
                            wandb_log["epoch"] = epoch
                            wandb_log["step"] = global_step_count
                            wandb.log(wandb_log)

                        # Step-based early stop.
                        if stopper.step(avg_metric_step):
                            log_early_stopping(epoch, early_stopping_patience)
                            break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            progress.remove_task(batch_task)

            avg_train_loss = running_loss / max(num_batches, 1)
            epoch_train_time = time.time() - epoch_start

            train_stats = PhaseStats(
                wall_seconds=epoch_train_time,
                n_steps=num_batches,
                n_samples=samples_seen,
                throughput_samples_per_s=(
                    samples_seen / epoch_train_time if epoch_train_time > 0 else None
                ),
                throughput_steps_per_s=(
                    num_batches / epoch_train_time if epoch_train_time > 0 else None
                ),
            )
            timing["train_epochs"].append(train_stats.to_dict())

            # ---------- Validation ----------
            val_metrics: dict[str, float] | None = None
            val_time = None
            is_best = False

            # Propagate step-based early stop out of the epoch loop.
            if stopper.wait >= stopper.patience:
                break

            # End-of-epoch validation only when step-based is NOT active.
            # When val_steps_cfg is set, vals fire inside the batch loop.
            if eval_fn is not None and val_steps_cfg is None:
                eval_start = time.time()
                val_metrics = eval_fn(model, epoch=epoch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                val_time = time.time() - eval_start

                n_imps = (
                    int(val_metrics.get("_num_impressions", 0))
                    if isinstance(val_metrics, dict)
                    else 0
                )
                val_stats = PhaseStats(
                    wall_seconds=val_time,
                    n_samples=n_imps or None,
                    throughput_samples_per_s=(
                        n_imps / val_time if (n_imps and val_time > 0) else None
                    ),
                )
                timing["val_epochs"].append(val_stats.to_dict())

                # Best tracking
                avg_metric = _score_for_tracking(val_metrics)

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
                        promote_best_val_predictions(
                            Path(save_dir).parent / "predictions"
                        )

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

    if save_dir:
        cleanup_val_staging(Path(save_dir).parent / "predictions")

    best_metrics["timing"] = timing
    return best_metrics
