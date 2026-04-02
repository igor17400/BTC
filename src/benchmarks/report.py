"""Generate comparison tables from benchmark results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def generate_comparison_table(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> str:
    """Generate a Markdown comparison table from benchmark results.

    Args:
        results: List of benchmark result dicts (from runner.py)
        output_path: Optional path to save the table as markdown

    Returns:
        Markdown string of the comparison table
    """
    if not results:
        return "No results to display."

    # Collect all metric keys
    metric_keys = set()
    for r in results:
        metric_keys.update(r.get("metrics", {}).keys())
    metric_keys = sorted(metric_keys)

    # Header
    headers = ["Framework", "Model", "Time (s)"] + metric_keys + ["Status"]
    separator = "|".join(["---"] * len(headers))
    header_row = "|".join(headers)

    rows = [f"|{header_row}|", f"|{separator}|"]

    for r in results:
        metrics = r.get("metrics", {})
        timing = r.get("timing", {})
        status = "OK" if r["error"] is None else "ERROR"

        values = [
            r["framework"],
            r.get("model", "N/A"),
            f"{timing.get('total_seconds', 0):.1f}",
        ]
        for key in metric_keys:
            val = metrics.get(key, 0)
            values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        values.append(status)

        rows.append("|" + "|".join(values) + "|")

    table_md = "\n".join(rows)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Cross-Framework Comparison\n\n{table_md}\n")
        console.log(f"Comparison table saved to {output_path}")

    return table_md


def generate_model_comparison(
    results: List[Dict[str, Any]],
) -> None:
    """Generate a Rich table comparing frameworks grouped by model.

    Args:
        results: List of benchmark result dicts
    """
    # Group by model
    models = {}
    for r in results:
        model = r.get("model", "unknown")
        if model not in models:
            models[model] = []
        models[model].append(r)

    for model_name, model_results in models.items():
        table = Table(title=f"Model: {model_name}")
        table.add_column("Framework", style="cyan")
        table.add_column("Training Time", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("nDCG@5", justify="right")
        table.add_column("nDCG@10", justify="right")

        for r in model_results:
            metrics = r.get("metrics", {})
            timing = r.get("timing", {})

            table.add_row(
                r["framework"],
                f"{timing.get('total_seconds', 0):.1f}s",
                f"{metrics.get('auc', 0):.4f}",
                f"{metrics.get('mrr', 0):.4f}",
                f"{metrics.get('ndcg@5', 0):.4f}",
                f"{metrics.get('ndcg@10', 0):.4f}",
            )

        console.print(table)
        console.print()


def save_results_json(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save raw benchmark results as JSON.

    Args:
        results: List of benchmark result dicts
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    console.log(f"Raw results saved to {output_path}")
