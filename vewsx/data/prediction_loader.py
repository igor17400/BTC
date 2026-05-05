"""Load model prediction files produced by NewsReX training runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_predictions(prediction_path: str) -> pd.DataFrame:
    """Parse a prediction TSV file.

    Expected format: ImpressionID \\t GroundTruth(JSON) \\t PredictionScores(JSON)

    Returns a DataFrame with columns: impression_id, ground_truth (list),
    prediction_scores (list).
    """
    path = Path(prediction_path)
    if not path.exists():
        return pd.DataFrame(
            columns=["impression_id", "ground_truth", "prediction_scores"]
        )

    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                rows.append(
                    {
                        "impression_id": parts[0],
                        "ground_truth": json.loads(parts[1]),
                        "prediction_scores": json.loads(parts[2]),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data
def load_run_summary(summary_path: str) -> dict:
    """Load a training_run_summary.json file."""
    path = Path(summary_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
