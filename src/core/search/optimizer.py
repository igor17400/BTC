"""Optuna-based hyperparameter optimizer.

Core search logic: creates studies, runs objectives, reports results.
Called by the thin ``src/search.py`` entry point.
"""

from __future__ import annotations

import gc
import importlib
import logging
from pathlib import Path

import optuna
from hydra import compose
from omegaconf import DictConfig

from src.core.search.spaces import get_search_space

logger = logging.getLogger(__name__)

FRAMEWORK_MODULES = {
    "pytorch": "src.frameworks.pytorch.runner",
    "jax": "src.frameworks.jax.runner",
}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    *,
    model: str,
    framework: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    seed: int,
    phase: int,
    wandb: bool,
) -> float:
    """Optuna objective function.

    Builds a Hydra config from trial suggestions, runs training via the
    existing framework runner, and returns the average validation metric.
    """
    # 1. Get search space overrides for this trial
    suggest_fn = get_search_space(phase, model)
    search_overrides = suggest_fn(trial)

    # 2. Build full Hydra override list
    overrides = [
        f"experiment={dataset}/{model}",
        f"framework={framework}",
        f"spec.training.num_epochs={epochs}",
        f"spec.training.batch_size={batch_size}",
        f"seed={seed}",
    ]
    overrides += search_overrides

    # WandB configuration
    if wandb:
        overrides += [
            "logging.enable_wandb=true",
            f"logging.wandb_group=search/{model}/{framework}",
            f"logging.experiment_name=trial_{trial.number}",
        ]
    else:
        overrides += ["logging.enable_wandb=false"]

    # 3. Compose config
    cfg = compose(config_name="config", overrides=overrides)

    # Log trial params
    params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
    logger.info("Trial %d: %s", trial.number, params_str)

    # 4. Run training
    try:
        runner = importlib.import_module(FRAMEWORK_MODULES[framework])
        metrics = runner.run(cfg)
    except Exception as exc:
        logger.error("Trial %d failed: %s", trial.number, exc)
        raise optuna.TrialPruned(str(exc)) from exc
    finally:
        _cleanup_gpu(framework)

    # 5. Extract objective value
    if isinstance(metrics, dict):
        avg = metrics.get("average_metric_value", 0.0)
        if avg <= 0:
            vals = [
                metrics.get(f"val_{m}", metrics.get(m, 0))
                for m in ["auc", "mrr", "ndcg@5", "ndcg@10"]
            ]
            avg = sum(v for v in vals if v) / max(len([v for v in vals if v]), 1)
    else:
        avg = 0.0

    logger.info("Trial %d result: %.4f", trial.number, avg)
    return avg


def _cleanup_gpu(framework: str) -> None:
    """Release GPU memory between trials."""
    gc.collect()
    try:
        if framework == "pytorch":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif framework == "jax":
            import jax

            jax.clear_caches()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


def run_search(cfg: DictConfig) -> optuna.Study:
    """Run an Optuna hyperparameter search from a Hydra config.

    Args:
        cfg: Hydra config with ``search`` section containing model, framework,
            n_trials, epochs, phase, storage, study_name, wandb.

    Returns:
        The completed Optuna study.
    """
    search = cfg.search
    model = search.model
    framework = search.framework
    phase = search.get("phase", 1)
    n_trials = search.get("n_trials", 20)
    epochs = search.get("epochs", 5)
    batch_size = search.get("batch_size", 16)
    seed = cfg.get("seed", 42)
    wandb = search.get("wandb", True)
    storage = search.get("storage", "sqlite:///outputs/search/optuna.db")
    study_name = search.get("study_name", f"{model}_{framework}_phase{phase}")

    # Ensure storage directory exists
    storage_path = storage.replace("sqlite:///", "")
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )

    # Run
    study.optimize(
        lambda trial: objective(
            trial,
            model=model,
            framework=framework,
            dataset=search.get("dataset", "mind"),
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            phase=phase,
            wandb=wandb,
        ),
        n_trials=n_trials,
        catch=(Exception,),
    )

    return study


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def print_results(study: optuna.Study, cfg: DictConfig) -> None:
    """Print search results and best configuration."""
    search = cfg.search

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(
        f"\nTrials: {len(completed)} completed, {len(pruned)} pruned, {len(failed)} failed"
    )

    if not completed:
        print("\nNo completed trials. Check logs for errors.")
        return

    # Best trial
    best = study.best_trial
    print(f"\nBest trial #{best.number}: avg_metric = {best.value:.4f}")
    print("Parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # All trials ranked
    print("\nAll completed trials (ranked):")
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
    for i, t in enumerate(sorted_trials, 1):
        params = ", ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"  {i}. trial #{t.number}: {t.value:.4f} | {params}")

    # Ready-to-copy command
    print("\nTrain with best config:")
    override_parts = []
    for k, v in best.params.items():
        if k == "learning_rate":
            override_parts.append(f"spec.training.learning_rate={v}")
        elif k == "dropout_rate":
            override_parts.append(f"spec.model.dropout_rate={v}")
        elif k == "neg_candidates":
            override_parts.append(f"spec.training.negative_sampling.candidates={v}")
            override_parts.append(f"spec.inputs.impressions.max_length={v + 1}")
        else:
            override_parts.append(f"{k}={v}")

    dataset = search.get("dataset", "mind")
    framework = search.framework
    model = search.model
    overrides_str = " \\\n    ".join(override_parts)
    print(
        f"  python src/train.py \\\n"
        f"    experiment={dataset}/{model} \\\n"
        f"    framework={framework} \\\n"
        f"    {overrides_str}"
    )
