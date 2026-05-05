"""Scan NewsReX outputs/ directory for completed training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from vewsx.config import OUTPUTS_DIR


@dataclass
class RunInfo:
    """Metadata for a discovered training run."""

    name: str
    path: str
    model: str = "unknown"
    framework: str = "unknown"
    dataset: str = "unknown"
    metrics: dict = field(default_factory=dict)
    prediction_files: list[str] = field(default_factory=list)
    summary_path: str | None = None


@st.cache_data(ttl=60)
def scan_runs(outputs_dir: str | None = None) -> list[RunInfo]:
    """Walk the outputs directory and collect training run metadata.

    Looks for:
    - training_run_summary.json files
    - prediction TSV files (*predictions*.txt)
    """
    base = Path(outputs_dir) if outputs_dir else OUTPUTS_DIR
    runs = []

    if not base.exists():
        return runs

    # Look for summary files
    for summary_file in sorted(base.rglob("training_run_summary.json")):
        run_dir = summary_file.parent
        try:
            with open(summary_file) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            summary = {}

        # Find prediction files in the same run directory
        pred_files = sorted(str(p) for p in run_dir.rglob("*predictions*.txt"))

        run = RunInfo(
            name=run_dir.name,
            path=str(run_dir),
            model=summary.get("model", summary.get("model_name", "unknown")),
            framework=summary.get("framework", "unknown"),
            dataset=summary.get("dataset", "unknown"),
            metrics=summary.get("best_metrics", summary.get("test_metrics", {})),
            prediction_files=pred_files,
            summary_path=str(summary_file),
        )
        runs.append(run)

    # Also check for prediction files without summaries
    if not runs:
        for pred_file in sorted(base.rglob("*predictions*.txt")):
            run_dir = pred_file.parent
            if not any(r.path == str(run_dir) for r in runs):
                runs.append(
                    RunInfo(
                        name=run_dir.name,
                        path=str(run_dir),
                        prediction_files=[str(pred_file)],
                    )
                )

    return runs
