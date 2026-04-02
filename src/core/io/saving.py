import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
import hydra

console = Console()


def get_output_run_dir(cfg):
    """
    Returns the output directory for the current run.
    Uses Hydra's working directory to avoid duplicate folder structures.
    """
    # Get Hydra's current working directory
    output_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    return output_run_dir


def save_predictions_to_file_fn(
        predictions_dict: Dict[str, Tuple[List, List]],
        output_dir: Path,
        epoch_idx: Optional[int] = None,
        mode: str = "val",
) -> None:
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mode}_predictions"
    if epoch_idx is not None:
        filename += f"_epoch_{epoch_idx + 1}"
    filename += ".txt"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        f.write("ImpressionID\tGroundTruth\tPredictionScores\n")
        for imp_id, (gt, pred_scores) in predictions_dict.items():
            gt_str = json.dumps(gt)
            pred_scores_str = json.dumps(pred_scores)
            f.write(f"{imp_id}\t{gt_str}\t{pred_scores_str}\n")
    console.log(f"Saved {mode} predictions to {filepath}")


def save_run_summary_fn(
        summary_output_dir: Path,
        hydra_cfg: DictConfig,
        initial_metrics_dict: Dict[str, float],
        best_metrics_summary_dict: Dict[str, Any],
        test_metrics_dict: Optional[Dict[str, float]] = None,
        wandb_full_history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Saves training config, key metrics, and history to a JSON file."""
    data_to_save = {
        "configuration": OmegaConf.to_container(hydra_cfg, resolve=True, throw_on_missing=True),
        "initial_validation_metrics": {k: float(v) for k, v in initial_metrics_dict.items()},
        "best_validation_summary": {
            k: (float(v) if isinstance(v, (int, float, np.float32, np.float64)) else v)
            for k, v in best_metrics_summary_dict.items()
        },
    }
    if test_metrics_dict:
        data_to_save["final_test_metrics"] = {k: float(v) for k, v in test_metrics_dict.items()}
    if wandb_full_history:
        data_to_save["wandb_run_history"] = wandb_full_history

    summary_filepath = summary_output_dir / "training_run_summary.json"
    try:
        with open(summary_filepath, "w") as f:
            json.dump(
                data_to_save, f, indent=4, default=lambda o: str(o) if isinstance(o, Path) else None
            )  # Handle Path objects, raise for others
        console.log(f"Training run summary saved to {summary_filepath}")
    except TypeError as e:
        console.log(
            f"[red]Error saving summary to JSON: {e}. Data causing issues might be in complex objects.[/red]"
        )
        # Fallback: try to save parts or print
        console.log(f"Problematic data (partial): {str(data_to_save)[:500]}")
