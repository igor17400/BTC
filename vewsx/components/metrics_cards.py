"""Reusable KPI / metric card components."""

from __future__ import annotations

import streamlit as st


def render_metric_row(metrics: dict[str, float | int | str], columns: int = 4) -> None:
    """Render a row of st.metric cards.

    Args:
        metrics: Dict of {label: value}.
        columns: Number of columns in the row.
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if isinstance(value, float):
                st.metric(label, f"{value:.4f}")
            else:
                st.metric(label, f"{value:,}" if isinstance(value, int) else str(value))


def render_metric_comparison(
    current: dict[str, float],
    baseline: dict[str, float],
    columns: int = 4,
) -> None:
    """Render metric cards with delta vs. baseline.

    Args:
        current: Current metric values.
        baseline: Baseline metric values for delta computation.
        columns: Number of columns.
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(current.items()):
        with cols[i % columns]:
            delta = value - baseline.get(label, value)
            st.metric(
                label, f"{value:.4f}", delta=f"{delta:+.4f}" if delta != 0 else None
            )
