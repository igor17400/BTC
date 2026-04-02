#!/usr/bin/env python3
"""Smoke test: train all models × frameworks on synthetic data and compare.

Usage:
    # All models, all frameworks
    uv run python tests/smoke.py

    # Specific models and frameworks
    uv run python tests/smoke.py --models nrms naml --frameworks jax pytorch

    # With Keras backends
    uv run python tests/smoke.py --models nrms --frameworks jax pytorch keras+jax keras+torch
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
console = Console()

ALL_MODELS = ["nrms", "naml", "lstur"]
ALL_FRAMEWORKS = ["jax", "pytorch", "keras+jax", "keras+torch"]


def run_one(model: str, framework: str) -> dict:
    """Run a single model×framework smoke test in a subprocess."""
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

    # Build the train.py command with Hydra overrides
    cmd = [
        sys.executable,
        "src/train.py",
        f"experiment=smoke/{model}",
        "eval.run_test_after_training=true",
        "logging.enable_wandb=false",
    ]

    # Parse framework spec: "keras+jax" → framework=keras, backend=jax
    if "+" in framework:
        fw, backend = framework.split("+")
        cmd.append(f"framework={fw}")
        cmd.append(f"device.keras_backend={backend}")
    else:
        cmd.append(f"framework={framework}")

    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )

        elapsed = time.time() - start
        result["time"] = f"{elapsed:.1f}s"

        if proc.returncode != 0:
            # Extract last meaningful error line
            stderr_lines = proc.stderr.strip().split("\n")
            error_msg = stderr_lines[-1] if stderr_lines else "Unknown error"
            # Also check stdout for rich-formatted errors
            for line in proc.stdout.split("\n"):
                if "ERROR" in line or "Error" in line:
                    error_msg = line.strip()
            result["error"] = error_msg
            console.print(f"[red]  FAIL: {error_msg}[/red]")
            return result

        # Parse metrics from stdout (look for "Test results:" or "val:" lines)
        for line in proc.stdout.split("\n"):
            if "Test results:" in line or "val:" in line:
                # Extract key=value or key: value pairs
                for part in line.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                    elif ":" in part and part[0].isalpha():
                        continue  # skip "results:" etc
                    else:
                        continue
                    k = k.strip()
                    try:
                        v_float = float(v)
                        if k in ("loss", "auc", "mrr", "ndcg@5", "ndcg@10"):
                            result[k] = f"{v_float:.4f}"
                    except ValueError:
                        pass

        # Also try parsing "key: value" format from JAX output
        for line in proc.stdout.split("\n"):
            if "Test results:" in line:
                parts = line.split("Test results:")[-1].strip()
                for pair in parts.split("  "):
                    pair = pair.strip()
                    if ": " in pair:
                        k, v = pair.split(": ", 1)
                        k = k.strip()
                        try:
                            v_float = float(v)
                            if k in ("loss", "auc", "mrr", "ndcg@5", "ndcg@10"):
                                result[k] = f"{v_float:.4f}"
                        except ValueError:
                            pass

        result["status"] = "PASS"

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (120s)"
        console.print("[red]  TIMEOUT[/red]")
    except Exception as e:
        result["error"] = str(e)
        console.print(f"[red]  ERROR: {e}[/red]")

    return result


def print_results_table(results: list[dict]) -> None:
    """Print a rich comparison table."""
    table = Table(title="Smoke Test Results", show_lines=True)

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

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    console.print(
        f"\n[bold]{passed} passed[/bold], [bold red]{failed} failed[/bold red]"
    )

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

    console.print(
        f"[bold]Running smoke tests: {args.models} × {args.frameworks}[/bold]\n"
    )

    results = []
    for model in args.models:
        for framework in args.frameworks:
            console.rule(f"{model.upper()} / {framework}")
            result = run_one(model, framework)
            results.append(result)

    print_results_table(results)

    sys.exit(sum(1 for r in results if r["status"] == "FAIL"))


if __name__ == "__main__":
    main()
