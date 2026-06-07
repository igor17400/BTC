"""Cross-framework benchmark runner.

Run the same experiment config across all 3 frameworks and collect
metrics + timing (training time, eval time, memory).
"""

import time
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

console = Console()

SUPPORTED_FRAMEWORKS = ["pytorch", "jax"]


def run_single_framework(
    cfg: DictConfig,
    framework: str,
) -> dict[str, Any]:
    """Run a single framework and collect results.

    Args:
        cfg: Hydra configuration
        framework: One of 'pytorch', 'jax'

    Returns:
        Dictionary with metrics, timing, and metadata
    """
    console.log(f"[bold blue]Running {framework} framework...[/bold blue]")
    result = {
        "framework": framework,
        "model": cfg.model_name,
        "dataset": cfg.dataset.name,
        "metrics": {},
        "timing": {},
        "error": None,
    }

    start_time = time.time()

    try:
        import importlib

        framework_modules = {
            "pytorch": "src.frameworks.pytorch.runner",
            "jax": "src.frameworks.jax.runner",
        }
        if framework not in framework_modules:
            raise ValueError(f"Unknown framework: {framework}")
        runner = importlib.import_module(framework_modules[framework])
        runner.run(cfg)

        result["timing"]["total_seconds"] = time.time() - start_time

    except Exception as e:
        result["error"] = str(e)
        result["timing"]["total_seconds"] = time.time() - start_time
        console.log(f"[red]Error running {framework}: {e}[/red]")

    return result


def run_benchmark(
    cfg: DictConfig,
    frameworks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run benchmark across specified frameworks.

    Args:
        cfg: Base Hydra configuration
        frameworks: List of frameworks to benchmark (default: all)

    Returns:
        List of result dictionaries, one per framework
    """
    if frameworks is None:
        frameworks = SUPPORTED_FRAMEWORKS

    results = []
    for fw in frameworks:
        if fw not in SUPPORTED_FRAMEWORKS:
            console.log(f"[yellow]Skipping unknown framework: {fw}[/yellow]")
            continue

        fw_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        fw_cfg.framework = fw

        result = run_single_framework(fw_cfg, fw)
        results.append(result)

    return results


def display_results_table(results: list[dict[str, Any]]) -> None:
    """Display benchmark results as a Rich table."""
    table = Table(title="Cross-Framework Benchmark Results")
    table.add_column("Framework", style="cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("nDCG@5", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("Status", style="green")

    for r in results:
        metrics = r.get("metrics", {})
        timing = r.get("timing", {})
        status = "OK" if r["error"] is None else f"ERROR: {r['error'][:30]}"

        table.add_row(
            r["framework"],
            f"{timing.get('total_seconds', 0):.1f}",
            f"{metrics.get('auc', 0):.4f}",
            f"{metrics.get('mrr', 0):.4f}",
            f"{metrics.get('ndcg@5', 0):.4f}",
            f"{metrics.get('ndcg@10', 0):.4f}",
            status,
        )

    console.print(table)
