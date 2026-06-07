#!/usr/bin/env python3
"""Smoke test: train all models × frameworks on synthetic data and compare.

Usage:
    # All models, all frameworks
    uv run python tests/smoke.py

    # Specific models and frameworks
    uv run python tests/smoke.py --models nrms naml --frameworks jax pytorch


Logs are saved to tests/.smoke_logs/<model>_<framework>.log for inspection.
"""

import argparse
import contextlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = Path(__file__).parent / ".smoke_logs"
console = Console()

ALL_MODELS = ["nrms", "naml", "lstur", "pprec", "crown"]
ALL_FRAMEWORKS = ["jax", "pytorch"]


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
        "log_file": None,
    }

    # Build the train.py command with Hydra overrides
    cmd = [
        sys.executable,
        "src/train.py",
        f"experiment=smoke/{model}",
        "eval.run_test_after_training=true",
        "logging.enable_wandb=false",
    ]

    cmd.append(f"framework={framework}")

    # Log file for this run
    log_name = f"{model}_{framework.replace('+', '_')}.log"
    log_path = LOGS_DIR / log_name
    result["log_file"] = str(log_path)

    start = time.time()
    console.print(f"  Running {model.upper()}/{framework}...", end=" ")

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

        # Save full output to log file
        full_output = proc.stdout + "\n" + proc.stderr
        log_path.write_text(
            f"# {model.upper()} / {framework}\n"
            f"# Command: {' '.join(cmd)}\n"
            f"# Date: {datetime.now().isoformat()}\n"
            f"# Exit code: {proc.returncode}\n"
            f"# Elapsed: {elapsed:.1f}s\n\n"
            f"--- STDOUT ---\n{proc.stdout}\n\n"
            f"--- STDERR ---\n{proc.stderr}\n"
        )

        if proc.returncode != 0:
            # Extract last meaningful error line
            stderr_lines = proc.stderr.strip().split("\n")
            error_msg = stderr_lines[-1] if stderr_lines else "Unknown error"
            # Also check stdout for rich-formatted errors
            for line in proc.stdout.split("\n"):
                if "ERROR" in line or "Error" in line:
                    error_msg = line.strip()
            result["error"] = error_msg
            console.print(f"[red]FAIL[/red] ({elapsed:.1f}s)")
            return result

        # Parse metrics from combined output
        metric_keys = {"loss", "auc", "mrr", "ndcg@5", "ndcg@10"}

        # Find the test results section and parse all key=value after it
        if "Test results:" in full_output:
            after_test = full_output.split("Test results:")[-1]
            for part in after_test.split():
                if part.startswith("---") or part.startswith("["):
                    break
                if "=" in part:
                    k, _, v = part.partition("=")
                    if k in metric_keys:
                        with contextlib.suppress(ValueError):
                            result[k] = f"{float(v):.4f}"

        result["status"] = "PASS"
        console.print(f"[green]PASS[/green] ({elapsed:.1f}s)")

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (120s)"
        log_path.write_text(f"# {model.upper()} / {framework}\n# TIMEOUT after 120s\n")
        console.print("[red]TIMEOUT[/red]")
    except Exception as e:
        result["error"] = str(e)
        log_path.write_text(f"# {model.upper()} / {framework}\n# ERROR: {e}\n")
        console.print("[red]ERROR[/red]")

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
                console.print(f"    Log: {r['log_file']}")

    console.print(f"\n[dim]Full logs: {LOGS_DIR}/[/dim]")


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

    # Create logs directory
    LOGS_DIR.mkdir(exist_ok=True)

    total = len(args.models) * len(args.frameworks)
    console.print(
        f"[bold]Running {total} smoke tests: {args.models} × {args.frameworks}[/bold]"
    )
    console.print(f"[dim]Logs will be saved to {LOGS_DIR}/[/dim]\n")

    results = []
    for model in args.models:
        for framework in args.frameworks:
            result = run_one(model, framework)
            results.append(result)

    print_results_table(results)

    sys.exit(sum(1 for r in results if r["status"] == "FAIL"))


if __name__ == "__main__":
    main()
