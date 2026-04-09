"""Page 7: Model Comparison — compare multiple training runs."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_run_selector
from vewsx.data.prediction_loader import load_run_summary

st.title("Model Comparison")

runs = render_run_selector(multi=True)
if not runs or len(runs) < 2:
    st.info("Select at least 2 runs from the sidebar to compare.")
    st.stop()

# -- Metric Comparison Table --
st.subheader("Metric Comparison")
rows = []
for run in runs:
    row = {"Run": run.name, "Model": run.model, "Framework": run.framework}
    if run.summary_path:
        summary = load_run_summary(run.summary_path)
        metrics = summary.get("test_metrics", summary.get("best_metrics", {}))
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    rows.append(row)

comparison_df = pd.DataFrame(rows)
st.dataframe(comparison_df, use_container_width=True)

# -- Grouped Bar Chart --
metric_cols = [
    c for c in comparison_df.columns if c not in {"Run", "Model", "Framework"}
]
if metric_cols:
    selected = st.multiselect(
        "Metrics to compare",
        metric_cols,
        default=[c for c in ["auc", "mrr", "ndcg@5", "ndcg@10"] if c in metric_cols],
    )
    if selected:
        melted = comparison_df.melt(
            id_vars=["Run", "Model"],
            value_vars=selected,
            var_name="Metric",
            value_name="Value",
        )
        fig = px.bar(
            melted,
            x="Metric",
            y="Value",
            color="Run",
            barmode="group",
            title="Metric Comparison",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Training Curves Overlay --
st.subheader("Training Curves")
curve_metric = st.selectbox(
    "Metric for curves",
    ["loss", "auc", "mrr", "ndcg@5", "ndcg@10"],
    index=1,
)

fig = go.Figure()
for run in runs:
    if not run.summary_path:
        continue
    summary = load_run_summary(run.summary_path)
    epoch_history = summary.get("epoch_history", [])
    if not epoch_history:
        continue
    hist_df = pd.DataFrame(epoch_history)
    if curve_metric in hist_df.columns:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(hist_df))),
                y=hist_df[curve_metric],
                name=f"{run.model}/{run.framework}",
                mode="lines+markers",
            )
        )

fig.update_layout(
    title=f"{curve_metric} Over Epochs",
    xaxis_title="Epoch",
    yaxis_title=curve_metric,
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Config Diff --
st.subheader("Configuration Differences")
configs = {}
for run in runs:
    if run.summary_path:
        summary = load_run_summary(run.summary_path)
        configs[run.name] = summary.get("config", {})

if len(configs) >= 2:
    # Flatten configs and find differences
    all_keys = set()
    flat_configs = {}
    for name, cfg in configs.items():
        flat = pd.json_normalize(cfg, sep=".").to_dict(orient="records")
        flat_configs[name] = flat[0] if flat else {}
        all_keys.update(flat_configs[name].keys())

    diff_rows = []
    for key in sorted(all_keys):
        values = {name: flat_configs[name].get(key, "—") for name in flat_configs}
        unique_vals = set(str(v) for v in values.values())
        if len(unique_vals) > 1:
            diff_rows.append({"Parameter": key, **values})

    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)
    else:
        st.info("No configuration differences found between selected runs.")
else:
    st.info("Config comparison requires summaries with config data.")
