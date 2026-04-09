"""Global sidebar components shared across all pages."""

from __future__ import annotations

import streamlit as st

from vewsx.data.loaders import discover_datasets
from vewsx.data.run_scanner import scan_runs


def render_dataset_selector() -> dict | None:
    """Render dataset selector in the sidebar.

    Returns the selected dataset dict with keys: name, path, splits,
    and optionally processed_path.
    """
    datasets = discover_datasets()
    if not datasets:
        st.sidebar.warning("No datasets found in .data/ directory.")
        dataset_path = st.sidebar.text_input(
            "Dataset path", placeholder="/path/to/mind-small/small"
        )
        if dataset_path:
            return {"name": "custom", "path": dataset_path, "splits": []}
        return None

    names = [d["name"] for d in datasets]
    selected = st.sidebar.selectbox("Dataset", names, key="vewsx_dataset")
    return next(d for d in datasets if d["name"] == selected)


def render_split_selector(available_splits: list[str] | None = None) -> str:
    """Render train/val/test split selector in the sidebar."""
    splits = available_splits or ["train", "valid", "test"]
    return st.sidebar.radio("Split", splits, key="vewsx_split", horizontal=True)


def render_category_filter(categories: list[str]) -> list[str]:
    """Render category multi-select filter in the sidebar."""
    return st.sidebar.multiselect(
        "Filter categories",
        sorted(categories),
        key="vewsx_categories",
    )


def render_run_selector(multi: bool = False) -> list | None:
    """Render model run selector in the sidebar.

    Args:
        multi: Allow multiple run selection.

    Returns:
        List of selected RunInfo objects, or None.
    """
    runs = scan_runs()
    if not runs:
        st.sidebar.info("No training runs found in outputs/.")
        return None

    labels = [f"{r.model}/{r.framework} ({r.name})" for r in runs]
    if multi:
        selected = st.sidebar.multiselect("Select runs", labels, key="vewsx_runs")
        return [runs[labels.index(s)] for s in selected] if selected else None
    else:
        selected = st.sidebar.selectbox("Select run", labels, key="vewsx_run")
        return [runs[labels.index(selected)]] if selected else None
