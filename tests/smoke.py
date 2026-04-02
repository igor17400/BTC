#!/usr/bin/env python3
"""Smoke test: train all models × frameworks on synthetic data and compare.

Usage:
    # All models, all frameworks
    uv run python tests/smoke.py

    # Specific models and frameworks
    uv run python tests/smoke.py --models nrms naml --frameworks jax pytorch

    # Single combo
    uv run python tests/smoke.py --models nrms --frameworks jax
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

ALL_MODELS = ["nrms", "naml", "lstur"]
ALL_FRAMEWORKS = ["keras", "jax", "pytorch"]

FRAMEWORK_MODULES = {
    "keras": "src.frameworks.keras.runner",
    "pytorch": "src.frameworks.pytorch.runner",
    "jax": "src.frameworks.jax.runner",
}


def load_smoke_config(model: str) -> dict:
    """Load and resolve a smoke experiment config."""
    config_dir = PROJECT_ROOT / "configs"

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"experiment=smoke/{model}"])

    return cfg


def run_one(model: str, framework: str) -> dict:
    """Run a single model×framework smoke test. Returns metrics dict."""
    result = {
        "model": model.upper(),
        "framework": framework,
        "status": "FAIL",
        "loss": "-",
        "auc": "-",
        "mrr": "-",
        "ndcg@5": "-",
        "ndcg@10": "-",
        "time": "-",
        "error": None,
    }

    try:
        cfg = load_smoke_config(model)

        # Override framework
        cfg.framework = framework

        # Force eval + test on, wandb off
        cfg.eval.fast_evaluation = True
        cfg.eval.run_test_after_training = True
        cfg.logging.enable_wandb = False

        start = time.time()

        runner = importlib.import_module(FRAMEWORK_MODULES[framework])
        metrics = runner.run(cfg)

        elapsed = time.time() - start
        result["time"] = f"{elapsed:.1f}s"

        # Extract metrics from runner return (if available)
        if isinstance(metrics, dict):
            for key in ["loss", "auc", "mrr", "ndcg@5", "ndcg@10"]:
                val = metrics.get(key) or metrics.get(f"val_{key}")
                if val is not None and val != 0:
                    result[key] = f"{float(val):.4f}"

        result["status"] = "PASS"

    except Exception as e:
        result["error"] = str(e)
        console.print(f"[red]  ERROR: {e}[/red]")

    # Clear Hydra global state for next run
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    return result


def print_results_table(results: list[dict]) -> None:
    """Print a rich comparison table."""
    table = Table(
        title="Smoke Test Results",
        show_lines=True,
    )

    table.add_column("Model", style="bold cyan")
    table.add_column("Framework", style="bold")
    table.add_column("Status")
    table.add_column("Loss", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("nDCG@5", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        status_style = "green" if r["status"] == "PASS" else "red"
        table.add_row(
            r["model"],
            r["framework"],
            f"[{status_style}]{r['status']}[/{status_style}]",
            str(r["loss"]),
            str(r["auc"]),
            str(r["mrr"]),
            str(r["ndcg@5"]),
            str(r["ndcg@10"]),
            str(r["time"]),
        )

    console.print()
    console.print(table)

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    console.print(f"\n[bold]{passed} passed[/bold], [bold red]{failed} failed[/bold red]")

    if failed > 0:
        console.print("\n[red]Failures:[/red]")
        for r in results:
            if r["status"] == "FAIL":
                console.print(f"  {r['model']}/{r['framework']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="NewsReX smoke tests")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help=f"Models to test (default: {ALL_MODELS})",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=ALL_FRAMEWORKS,
        choices=ALL_FRAMEWORKS,
        help=f"Frameworks to test (default: {ALL_FRAMEWORKS})",
    )
    args = parser.parse_args()

    console.print(f"[bold]Running smoke tests: {args.models} × {args.frameworks}[/bold]\n")

    results = []
    for model in args.models:
        for framework in args.frameworks:
            console.rule(f"{model.upper()} / {framework}")
            result = run_one(model, framework)
            results.append(result)

    print_results_table(results)

    # Exit with failure count
    sys.exit(sum(1 for r in results if r["status"] == "FAIL"))


if __name__ == "__main__":
    main()
