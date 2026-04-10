import datetime
import logging

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

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
    logging.getLogger("jax").setLevel(
        logging.WARNING
    )  # Reduce JAX compilation messages
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
        group = cfg.logging.get("wandb_group", None)
        tags = [
            cfg.get("framework", "unknown"),
            cfg.get("dataset", {}).get("name", "unknown"),
            cfg.get("model_name", "unknown"),
        ]
        try:
            wandb.init(
                project=cfg.logging.project_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=run_name,
                group=group,
                tags=tags,
            )
            console.log(
                f"Wandb initialized for project '{cfg.logging.project_name}', "
                f"group '{group}', run '{run_name}'"
            )
        except Exception as e:
            console.log(f"[red]Failed to initialize Wandb: {e}[/red]")


def log_metrics_to_console_fn(
    metrics_dict: dict[str, float], header_prefix: str = ""
) -> None:
    if header_prefix:
        console.log(f"[bold blue]{header_prefix} Metrics:[/bold blue]")
    for name, value in metrics_dict.items():
        try:
            console.log(f"  [green]{name}: {float(value):.4f}[/green]")
        except (ValueError, TypeError):
            console.log(f"  [green]{name}: {value}[/green]")


def log_epoch_summary_fn(
    current_epoch_idx: int,
    epoch_train_metrics_results: dict[str, float],
    epoch_val_metrics_results: dict[str, float],
    is_best_epoch: bool,
    wandb_cache: dict[str, list[float]] | None = None,
) -> None:
    """Logs comprehensive summary for an epoch to console and WandB."""
    console.rule(
        f"[bold magenta]Epoch {current_epoch_idx + 1} Completed[/bold magenta]"
    )
    log_metrics_to_console_fn(epoch_train_metrics_results, "Average Training")
    log_metrics_to_console_fn(epoch_val_metrics_results, "Validation")
    if is_best_epoch:
        console.log(
            "[bold green]✨ New best epoch based on validation metric! ✨[/bold green]"
        )
    console.rule()

    if wandb.run and wandb_cache is not None:
        wandb_payload = {
            f"train/{k}": v for k, v in epoch_train_metrics_results.items()
        } | {f"val/{k}": v for k, v in epoch_val_metrics_results.items()}
        log_metrics_to_wandb_fn(wandb_payload, current_epoch_idx + 1, wandb_cache)


def log_metrics_to_wandb_fn(
    metrics_payload: dict[str, float],
    commit_step: int,
    wandb_history_cache: dict[str, list[float]],
) -> None:
    if wandb.run:
        wandb.log(metrics_payload, step=commit_step)
        for key, value in metrics_payload.items():
            wandb_history_cache.setdefault(key, []).append(float(value))


# ---------------------------------------------------------------------------
# Shared training lifecycle logging (framework-agnostic)
# ---------------------------------------------------------------------------


def log_training_start(model_name: str, framework: str, num_epochs: int) -> None:
    console.log(
        f"[bold blue]Training {model_name} ({framework}) for {num_epochs} epochs[/bold blue]"
    )


def log_epoch_end(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    train_time: float,
    val_metrics: dict[str, float] | None = None,
    val_time: float | None = None,
    is_best: bool = False,
) -> None:
    """Log a consistent epoch summary across all frameworks."""
    parts = [f"[bold]Epoch {epoch}/{num_epochs}[/bold]"]
    parts.append(f"loss={train_loss:.4f}")
    parts.append(f"time={train_time:.1f}s")
    console.log("  ".join(parts))

    if val_metrics:
        val_parts = []
        for k, v in val_metrics.items():
            if k != "num_impressions":
                val_parts.append(f"{k}={v:.4f}")
        val_str = "  ".join(val_parts)
        time_str = f"  time={val_time:.1f}s" if val_time else ""
        console.log(f"  val: {val_str}{time_str}")

    if is_best:
        console.log("[bold green]  New best epoch![/bold green]")


def log_early_stopping(epoch: int, patience: int) -> None:
    console.log(
        f"[bold red]Early stopping at epoch {epoch} (no improvement for {patience} epochs)[/bold red]"
    )


def log_test_results(metrics: dict[str, float]) -> None:
    parts = []
    for k, v in metrics.items():
        if k != "num_impressions":
            parts.append(f"{k}={v:.4f}")
    console.log(f"[bold green]Test results:[/bold green] {'  '.join(parts)}")

    if wandb.run:
        wandb.log(
            {f"test/{k}": v for k, v in metrics.items() if k != "num_impressions"}
        )


def log_training_complete(model_name: str, framework: str, total_time: float) -> None:
    console.log(f"--- {model_name} ({framework}) finished in {total_time:.1f}s ---")
