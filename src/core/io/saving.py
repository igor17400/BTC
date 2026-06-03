import json
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()

# Per-epoch validation predictions are written to a single staging file that
# each epoch overwrites (no accumulation). The training loop promotes the
# best epoch's file to the canonical name via ``promote_best_val_predictions``,
# so the kept ``val_predictions.txt`` always matches the saved checkpoint.
_VAL_STAGING_NAME = "val_predictions.staging.txt"
_VAL_BEST_NAME = "val_predictions.txt"


def get_output_run_dir(cfg):
    """
    Returns the output directory for the current run.
    Checks for an explicit ``_output_run_dir`` override first (used by
    multi-seed training), then falls back to Hydra's working directory.
    """
    override = cfg.get("_output_run_dir", None) if hasattr(cfg, "get") else None
    if override:
        output_run_dir = Path(override)
    else:
        try:
            output_run_dir = Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except ValueError:
            # Running outside @hydra.main (e.g., smoke tests with hydra.compose)
            base = getattr(cfg, "output_base_dir", "outputs")
            dataset_name = (
                cfg.get("dataset", {}).get("name", "unknown")
                if hasattr(cfg, "get")
                else "unknown"
            )
            model_name = getattr(cfg, "model_name", "run")
            framework = getattr(cfg, "framework", "unknown")
            seed = getattr(cfg, "seed", 0)
            output_run_dir = (
                Path(base)
                / "train"
                / dataset_name
                / model_name
                / framework
                / f"seed_{seed}"
            )

    output_run_dir.mkdir(parents=True, exist_ok=True)
    return output_run_dir


def save_predictions_to_file_fn(
    predictions_dict: dict[str, tuple[list, list]],
    output_dir: Path,
    epoch_idx: int | None = None,
    mode: str = "val",
) -> None:
    console = Console()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Val predictions go to a single staging file (overwritten every epoch);
    # the best epoch is promoted to ``val_predictions.txt`` by the training
    # loop. Test predictions are final and written to their canonical name.
    # ``epoch_idx`` is no longer baked into the filename — we keep at most one
    # val file per run instead of one per epoch.
    filename = _VAL_STAGING_NAME if mode == "val" else f"{mode}_predictions.txt"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        f.write("ImpressionID\tGroundTruth\tPredictionScores\n")
        for imp_id, (gt, pred_scores) in predictions_dict.items():
            gt_str = json.dumps(gt)
            pred_scores_str = json.dumps(pred_scores)
            f.write(f"{imp_id}\t{gt_str}\t{pred_scores_str}\n")
    console.log(f"Saved {mode} predictions to {filepath}")


def promote_best_val_predictions(predictions_dir: Path | str) -> None:
    """Promote the staged val-predictions file to the canonical best file.

    Called by the training loop right after it saves a new-best checkpoint, so
    ``val_predictions.txt`` always holds the predictions of the epoch whose
    weights are in ``models/model.safetensors``. No-op when staging is absent
    (e.g. ``eval.save_predictions=false``).
    """
    predictions_dir = Path(predictions_dir)
    staging = predictions_dir / _VAL_STAGING_NAME
    if staging.exists():
        os.replace(staging, predictions_dir / _VAL_BEST_NAME)


def cleanup_val_staging(predictions_dir: Path | str) -> None:
    """Remove a leftover val staging file (when the last epoch wasn't best)."""
    staging = Path(predictions_dir) / _VAL_STAGING_NAME
    if staging.exists():
        staging.unlink()


def save_run_summary_fn(
    summary_output_dir: Path,
    hydra_cfg: DictConfig,
    initial_metrics_dict: dict[str, float],
    best_metrics_summary_dict: dict[str, Any],
    test_metrics_dict: dict[str, float] | None = None,
    wandb_full_history: dict[str, list[float]] | None = None,
) -> None:
    """Saves training config, key metrics, and history to a JSON file."""
    data_to_save = {
        "configuration": OmegaConf.to_container(
            hydra_cfg, resolve=True, throw_on_missing=True
        ),
        "initial_validation_metrics": {
            k: float(v) for k, v in initial_metrics_dict.items()
        },
        "best_validation_summary": {
            k: (float(v) if isinstance(v, (int, float, np.float32, np.float64)) else v)
            for k, v in best_metrics_summary_dict.items()
        },
    }
    if test_metrics_dict:
        data_to_save["final_test_metrics"] = {
            k: float(v) for k, v in test_metrics_dict.items()
        }
    if wandb_full_history:
        data_to_save["wandb_run_history"] = wandb_full_history

    summary_filepath = summary_output_dir / "training_run_summary.json"
    try:
        with open(summary_filepath, "w") as f:
            json.dump(
                data_to_save,
                f,
                indent=4,
                default=lambda o: str(o) if isinstance(o, Path) else None,
            )  # Handle Path objects, raise for others
        # Silent — summary saved alongside other outputs
    except TypeError as e:
        console.log(
            f"[red]Error saving summary to JSON: {e}. Data causing issues might be in complex objects.[/red]"
        )
        # Fallback: try to save parts or print
        console.log(f"Problematic data (partial): {str(data_to_save)[:500]}")
