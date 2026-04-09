"""Shared Plotly theme and helpers for consistent styling."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from vewsx.config import CATEGORY_COLORS

# -- Register custom Plotly template --
# Uses transparent backgrounds so Plotly charts adapt to Streamlit's
# light/dark theme automatically.

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family="Inter, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=CATEGORY_COLORS,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

pio.templates["vewsx"] = _template
pio.templates.default = "streamlit+vewsx"


def build_category_color_map(categories: list[str]) -> dict[str, str]:
    """Build a deterministic color map for categories."""
    return {
        cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        for i, cat in enumerate(sorted(categories))
    }


def apply_defaults(fig: go.Figure, height: int = 450) -> go.Figure:
    """Apply consistent layout defaults to a figure."""
    fig.update_layout(height=height)
    return fig
