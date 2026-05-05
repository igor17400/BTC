"""NewsReX hyperparameter search entry point.

Thin Hydra dispatcher — all search logic lives in
``src.core.search.optimizer``.

Usage::

    # Phase 1: training dynamics on MIND-small
    python src/search.py search.model=nrms search.framework=pytorch search.n_trials=12

    # Override epochs and disable WandB
    python src/search.py search.model=naml search.framework=pytorch search.epochs=3 search.wandb=false

    # Phase 2: architecture search
    python src/search.py search.model=nrms search.framework=pytorch search.phase=2 search.n_trials=50

    # Custom study name (for resumption)
    python src/search.py search.model=lstur search.framework=pytorch search.study_name=lstur_v2

    # View results interactively
    optuna-dashboard sqlite:///outputs/search/optuna.db
"""

import hydra
import optuna
from omegaconf import DictConfig

from src.core.io.logging import console, setup_logging
from src.core.io.progress import set_default_backend
from src.core.search.optimizer import print_results, run_search


@hydra.main(version_base=None, config_path="../configs", config_name="search")
def main(cfg: DictConfig) -> None:
    """Main entry point for hyperparameter search, configured by Hydra."""
    setup_logging(level=cfg.logging.level if hasattr(cfg.logging, "level") else "INFO")
    set_default_backend(cfg.logging.get("progress_backend", "rich"))

    # Suppress noisy Optuna logs unless explicitly verbose
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search = cfg.search
    console.log("[bold]Starting hyperparameter search[/bold]")
    console.log(f"  Model: {search.model} | Framework: {search.framework}")
    console.log(
        f"  Phase: {search.phase} | Trials: {search.n_trials} | Epochs: {search.epochs}"
    )
    console.log(f"  WandB: {'enabled' if search.wandb else 'disabled'}")
    console.log(f"  Storage: {search.storage}")

    study = run_search(cfg)
    print_results(study, cfg)


if __name__ == "__main__":
    main()
