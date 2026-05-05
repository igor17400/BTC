"""Page 6: Model Predictions — single model prediction analysis."""

import contextlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_run_selector
from vewsx.data.prediction_loader import load_predictions, load_run_summary

st.title("Model Predictions")

runs = render_run_selector(multi=False)
if not runs:
    st.stop()

run = runs[0]
st.caption(f"Run: {run.model} / {run.framework} — {run.name}")

# -- Run Summary Metrics --
if run.summary_path:
    summary = load_run_summary(run.summary_path)
    if summary.get("best_metrics") or summary.get("test_metrics"):
        st.subheader("Final Metrics")
        metrics = summary.get("test_metrics", summary.get("best_metrics", {}))
        render_metric_row(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            columns=5,
        )

    # -- Training Curve --
    epoch_history = summary.get("epoch_history", [])
    if epoch_history:
        st.subheader("Training Curve")
        hist_df = pd.DataFrame(epoch_history)
        metric_cols = [c for c in hist_df.columns if c != "epoch"]

        selected_metrics = st.multiselect(
            "Metrics to plot",
            metric_cols,
            default=[c for c in ["loss", "auc", "mrr"] if c in metric_cols],
        )
        if selected_metrics:
            fig = go.Figure()
            for col in selected_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=hist_df["epoch"]
                        if "epoch" in hist_df
                        else list(range(len(hist_df))),
                        y=hist_df[col],
                        name=col,
                        mode="lines+markers",
                    )
                )
            fig.update_layout(
                title="Metrics Over Epochs", xaxis_title="Epoch", yaxis_title="Value"
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Prediction Analysis --
if not run.prediction_files:
    st.info("No prediction files found for this run.")
    st.stop()

pred_file = st.selectbox("Prediction file", run.prediction_files)
pred_df = load_predictions(pred_file)

if pred_df.empty:
    st.warning("Prediction file is empty.")
    st.stop()

# Flatten scores and labels
all_scores = []
all_labels = []
per_impression_metrics = []

for _, row in pred_df.iterrows():
    gt = row["ground_truth"]
    scores = row["prediction_scores"]
    if len(gt) != len(scores) or len(gt) == 0:
        continue
    all_scores.extend(scores)
    all_labels.extend(gt)

    # Per-impression AUC
    gt_arr = np.array(gt)
    sc_arr = np.array(scores)
    if gt_arr.sum() > 0 and gt_arr.sum() < len(gt_arr):
        from sklearn.metrics import roc_auc_score

        with contextlib.suppress(ValueError):
            auc = roc_auc_score(gt_arr, sc_arr)
            per_impression_metrics.append({"auc": auc, "n_candidates": len(gt)})

# -- Score Distribution --
st.subheader("Score Distribution")
score_df = pd.DataFrame(
    {
        "score": all_scores,
        "label": ["Clicked" if lab == 1 else "Not Clicked" for lab in all_labels],
    }
)

fig = px.histogram(
    score_df,
    x="score",
    color="label",
    nbins=50,
    barmode="overlay",
    opacity=0.7,
    title="Prediction Score Distribution by Ground Truth",
    labels={"score": "Prediction Score", "count": "Count"},
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Score Separation Violin --
st.subheader("Score Separation")
fig = px.violin(
    score_df,
    y="score",
    x="label",
    box=True,
    title="Score Distribution: Clicked vs. Not Clicked",
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Calibration Plot --
st.subheader("Calibration")
n_bins = st.slider("Number of bins", 5, 20, 10)
scores_arr = np.array(all_scores)
labels_arr = np.array(all_labels)
bin_edges = np.linspace(scores_arr.min(), scores_arr.max(), n_bins + 1)
cal_data = []
for i in range(n_bins):
    mask = (scores_arr >= bin_edges[i]) & (scores_arr < bin_edges[i + 1])
    if mask.sum() > 0:
        cal_data.append(
            {
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "actual_ctr": labels_arr[mask].mean(),
                "count": int(mask.sum()),
            }
        )

if cal_data:
    cal_df = pd.DataFrame(cal_data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cal_df["bin_center"],
            y=cal_df["actual_ctr"],
            mode="lines+markers",
            name="Model",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title="Calibration Plot",
        xaxis_title="Predicted Score",
        yaxis_title="Actual CTR",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Per-Impression AUC Distribution --
if per_impression_metrics:
    st.subheader("Per-Impression AUC")
    pi_df = pd.DataFrame(per_impression_metrics)
    fig = px.histogram(
        pi_df,
        x="auc",
        nbins=40,
        title="Per-Impression AUC Distribution",
        labels={"auc": "AUC", "count": "Impressions"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)
    st.metric("Mean Per-Impression AUC", f"{pi_df['auc'].mean():.4f}")
