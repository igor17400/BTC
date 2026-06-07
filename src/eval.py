"""NewsReX evaluation entry point.

Load pre-trained weights and run test evaluation without training.

Usage:
    # Local weights
    uv run python src/eval.py experiment=mind/glove/nrms framework=jax \
        weights=outputs/MIND-small/NRMS/jax/seed_42/.../models/model.safetensors

    # HuggingFace Hub weights
    uv run python src/eval.py experiment=mind/glove/nrms framework=jax \
        weights=hf://newsrex/NRMS-JAX-MIND-small-seed42/model.safetensors
"""

import importlib
import sys

import hydra
from omegaconf import DictConfig, open_dict

from src.core.io.logging import console, setup_logging
from src.core.io.progress import set_default_backend

# Override Hydra output to inference/ instead of train/.
sys.argv.append(
    "hydra.run.dir=${output_base_dir}/inference/${dataset.name}/${model_name}/${framework}/${now:%Y-%m-%d-%H-%M-%S}"
)

FRAMEWORK_MODULES = {
    "pytorch": "src.frameworks.pytorch.runner",
    "jax": "src.frameworks.jax.runner",
}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run evaluation with pre-trained weights."""
    setup_logging(level=cfg.logging.level if hasattr(cfg.logging, "level") else "INFO")
    set_default_backend(cfg.logging.get("progress_backend", "rich"))

    framework = getattr(cfg, "framework", "jax")
    weights = getattr(cfg, "weights", None)

    if not weights:
        raise ValueError(
            "weights= is required. Pass a local path or hf://org/repo/file."
        )

    if framework not in FRAMEWORK_MODULES:
        raise ValueError(f"Unknown framework: '{framework}'.")

    # Force eval-only mode: skip training, run test, no wandb, no output.
    with open_dict(cfg):
        cfg.train.num_epochs = 0
        cfg.eval.run_test_after_training = True
        cfg.eval.save_predictions = False
        cfg.logging.enable_wandb = False
        cfg._eval_only = True

    console.log(f"--- {cfg.model_name} Evaluation (framework: {framework}) ---")
    console.log(f"Weights: {weights}")

    runner = importlib.import_module(FRAMEWORK_MODULES[framework])
    metrics = runner.run(cfg)

    console.rule("[bold]Evaluation Results")
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
