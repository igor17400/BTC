"""NewsReX training entry point.

Thin Hydra dispatcher — all framework logic lives in
``src.frameworks.{keras,pytorch,jax}.runner.run(cfg)``.
"""

import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.io.logging import setup_logging, console


FRAMEWORK_MODULES = {
    "keras": "src.frameworks.keras.runner",
    "pytorch": "src.frameworks.pytorch.runner",
    "jax": "src.frameworks.jax.runner",
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training, configured by Hydra."""
    setup_logging(level=cfg.logging.level if hasattr(cfg.logging, "level") else "INFO")

    framework = getattr(cfg, "framework", "keras")

    if framework not in FRAMEWORK_MODULES:
        raise ValueError(
            f"Unknown framework: '{framework}'. Available: {list(FRAMEWORK_MODULES.keys())}"
        )

    console.log(f"--- {cfg.model_name} Training Run Initializing (framework: {framework}) ---")
    console.log("Configuration used:")
    console.log(OmegaConf.to_yaml(cfg))
    console.log("------------------------------------")

    runner = importlib.import_module(FRAMEWORK_MODULES[framework])
    runner.run(cfg)


if __name__ == "__main__":
    main()
