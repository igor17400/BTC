import datetime
from typing import Dict, List, Optional

import logging

from rich.console import Console
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf

import wandb

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration with rich handler.

    Args:
        level: Logging level (default: "INFO")
    """
    # Remove any existing handlers
    logging.root.handlers = []

    # Configure logging with rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                console=console,
                show_time=True,
                show_path=False,
                enable_link_path=False,
            )
        ],
        force=True,  # Override any existing configuration
    )

    # Configure backend-specific logging
    logging.getLogger("hydra").setLevel(logging.WARNING)
    logging.getLogger("hydra").propagate = False

    # JAX backend logging configuration
    logging.getLogger("jax").setLevel(logging.WARNING)  # Reduce JAX compilation messages
    logging.getLogger("jaxlib").setLevel(logging.WARNING)  # Reduce JAXlib messages

    # Keras 3 logging
    logging.getLogger("keras").setLevel(logging.INFO)  # Show important Keras messages


def setup_wandb_session(cfg: DictConfig) -> None:
    """Initialize Weights & Biases logging."""
    if cfg.logging.enable_wandb:
        run_name = (
            cfg.logging.experiment_name
            or f"{cfg.get('model_name', 'model')}_run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        try:
            wandb.init(
                project=cfg.logging.project_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=run_name,
            )
            console.log(
                f"Wandb initialized for project '{cfg.logging.project_name}', run '{run_name}'"
            )
        except Exception as e:
            console.log(f"[red]Failed to initialize Wandb: {e}[/red]")


def log_metrics_to_console_fn(metrics_dict: Dict[str, float], header_prefix: str = "") -> None:
    if header_prefix:
        console.log(f"[bold blue]{header_prefix} Metrics:[/bold blue]")
    for name, value in metrics_dict.items():
        try:
            console.log(f"  [green]{name}: {float(value):.4f}[/green]")
        except (ValueError, TypeError):
            console.log(f"  [green]{name}: {value}[/green]")


def log_epoch_summary_fn(
    current_epoch_idx: int,
    epoch_train_metrics_results: Dict[str, float],
    epoch_val_metrics_results: Dict[str, float],
    is_best_epoch: bool,
    wandb_cache: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Logs comprehensive summary for an epoch to console and WandB."""
    console.rule(f"[bold magenta]Epoch {current_epoch_idx + 1} Completed[/bold magenta]")
    log_metrics_to_console_fn(epoch_train_metrics_results, "Average Training")
    log_metrics_to_console_fn(epoch_val_metrics_results, "Validation")
    if is_best_epoch:
        console.log("[bold green]✨ New best epoch based on validation metric! ✨[/bold green]")
    console.rule()

    if wandb.run and wandb_cache is not None:
        wandb_payload = {f"train/{k}": v for k, v in epoch_train_metrics_results.items()} | {
            f"val/{k}": v for k, v in epoch_val_metrics_results.items()
        }
        log_metrics_to_wandb_fn(wandb_payload, current_epoch_idx + 1, wandb_cache)


def log_metrics_to_wandb_fn(
    metrics_payload: Dict[str, float],  # Flat dict with prefixes like "train/loss"
    commit_step: int,  # Usually epoch index
    wandb_history_cache: Dict[str, List[float]],  # Local cache for full history
) -> None:
    if wandb.run:  # Check if wandb session is active
        wandb.log(metrics_payload, step=commit_step)
        for key, value in metrics_payload.items():
            wandb_history_cache.setdefault(key, []).append(float(value))
