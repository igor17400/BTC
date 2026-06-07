"""NewsReX training entry point.

Thin Hydra dispatcher — all framework logic lives in
``src.frameworks.{pytorch,jax}.runner.run(cfg)``.

When ``multi_seed.enabled`` is true, trains the model multiple times
with different seeds, collects metrics, and reports mean ± std.
"""

import importlib
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.core.io.logging import console, setup_logging
from src.core.io.progress import set_default_backend

FRAMEWORK_MODULES = {
    "pytorch": "src.frameworks.pytorch.runner",
    "jax": "src.frameworks.jax.runner",
}


def _run_single(cfg: DictConfig) -> dict:
    """Run a single training pass and return metrics."""
    framework = getattr(cfg, "framework", "jax")
    runner = importlib.import_module(FRAMEWORK_MODULES[framework])
    return runner.run(cfg) or {}


def _run_multi_seed(cfg: DictConfig) -> None:
    """Run training across multiple seeds and report aggregated results."""
    seeds = list(cfg.multi_seed.seeds)
    framework = getattr(cfg, "framework", "jax")
    model_name = cfg.model_name

    console.rule(
        f"[bold]Multi-Seed Training: {model_name} / {framework} ({len(seeds)} seeds)"
    )

    all_metrics = []
    for i, seed in enumerate(seeds):
        console.rule(f"Seed {seed} ({i + 1}/{len(seeds)})")

        # Override the seed and output dir for this run.
        # Hydra's runtime.output_dir is frozen at startup (seed 42),
        # so we build a per-seed output path explicitly. The split
        # strategy is part of the path so dev_as_val and random/
        # chronological runs don't collide.
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        cfg_copy["seed"] = seed
        split_strategy = cfg_copy.get("dataset", {}).get(
            "validation_split_strategy", "random"
        )
        encoder_slug = (
            cfg_copy.get("encoder", {}).get("slug", "glove")
            if isinstance(cfg_copy.get("encoder"), dict)
            else "glove"
        )
        cfg_copy["_output_run_dir"] = str(
            Path(cfg_copy["output_base_dir"])
            / "train"
            / cfg_copy.get("dataset", {}).get("name", "unknown")
            / model_name
            / framework
            / encoder_slug
            / split_strategy
            / f"seed_{seed}"
        )
        cfg_run = OmegaConf.create(cfg_copy)

        metrics = _run_single(cfg_run)
        if metrics:
            metrics["seed"] = seed
            all_metrics.append(metrics)
            console.log(f"Seed {seed} results: {metrics}")

    if not all_metrics:
        console.log("[red]No seeds produced metrics.[/red]")
        return

    # Aggregate results
    results_df = pd.DataFrame(all_metrics)
    metric_cols = [c for c in results_df.columns if c != "seed"]

    console.rule("[bold]Multi-Seed Results")
    console.print(results_df.to_string(index=False))

    # Compute mean ± std
    summary = {}
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(results_df[col]):
            mean = results_df[col].mean()
            std = results_df[col].std()
            summary[col] = f"{mean:.4f} ± {std:.4f}"

    console.print()
    console.rule("[bold]Mean ± Std")
    for metric, value in summary.items():
        console.print(f"  {metric}: {value}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training, configured by Hydra."""
    setup_logging(level=cfg.logging.level if hasattr(cfg.logging, "level") else "INFO")
    set_default_backend(cfg.logging.get("progress_backend", "rich"))

    framework = getattr(cfg, "framework", "jax")

    if framework not in FRAMEWORK_MODULES:
        raise ValueError(
            f"Unknown framework: '{framework}'. Available: {list(FRAMEWORK_MODULES.keys())}"
        )

    console.log(
        f"--- {cfg.model_name} Training Run Initializing (framework: {framework}) ---"
    )
    console.log("Configuration used:")
    console.log(OmegaConf.to_yaml(cfg))
    console.log("------------------------------------")

    if getattr(cfg, "multi_seed", None) and cfg.multi_seed.enabled:
        _run_multi_seed(cfg)
    else:
        _run_single(cfg)


if __name__ == "__main__":
    main()
