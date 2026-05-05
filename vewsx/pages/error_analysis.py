"""Page 8: Error Analysis — understand where and why models fail."""

import contextlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector, render_run_selector
from vewsx.data.loaders import load_raw_news
from vewsx.data.prediction_loader import load_predictions

st.title("Error Analysis")

dataset = render_dataset_selector()
runs = render_run_selector(multi=False)
if not runs or not dataset:
    st.info("Select a dataset and a run from the sidebar.")
    st.stop()

run = runs[0]
if not run.prediction_files:
    st.warning("No prediction files found for this run.")
    st.stop()

pred_file = st.selectbox("Prediction file", run.prediction_files)
pred_df = load_predictions(pred_file)
news_df = load_raw_news(dataset["path"])

if pred_df.empty:
    st.warning("Prediction file is empty.")
    st.stop()

# Compute per-impression analysis
impression_results = []
for _, row in pred_df.iterrows():
    gt = np.array(row["ground_truth"])
    scores = np.array(row["prediction_scores"])
    if len(gt) != len(scores) or len(gt) == 0:
        continue

    ranked_idx = np.argsort(-scores)
    top_label = gt[ranked_idx[0]]

    # Per-impression AUC
    auc = np.nan
    if gt.sum() > 0 and gt.sum() < len(gt):
        from sklearn.metrics import roc_auc_score

        with contextlib.suppress(ValueError):
            auc = roc_auc_score(gt, scores)

    # Find rank of first positive
    positive_ranks = np.where(gt[ranked_idx] == 1)[0]
    first_pos_rank = int(positive_ranks[0] + 1) if len(positive_ranks) > 0 else -1

    impression_results.append(
        {
            "impression_id": row["impression_id"],
            "n_candidates": len(gt),
            "n_positive": int(gt.sum()),
            "top_correct": int(top_label == 1),
            "auc": auc,
            "first_pos_rank": first_pos_rank,
            "max_score": float(scores.max()),
            "score_gap": float(scores.max() - scores.min()),
        }
    )

ir = pd.DataFrame(impression_results)

# -- Error Type Breakdown --
st.subheader("Error Type Breakdown")
n_correct = ir["top_correct"].sum()
n_wrong = len(ir) - n_correct
fig = px.pie(
    values=[n_correct, n_wrong],
    names=["Top-1 Correct", "Top-1 Wrong"],
    title="Was the Highest-Scored Item Actually Clicked?",
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- AUC Distribution --
st.subheader("Per-Impression AUC")
valid_auc = ir.dropna(subset=["auc"])

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        valid_auc,
        x="auc",
        nbins=40,
        title="Per-Impression AUC Distribution",
        labels={"auc": "AUC", "count": "Impressions"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

with col2:
    fig = px.histogram(
        ir[ir["first_pos_rank"] > 0],
        x="first_pos_rank",
        nbins=20,
        title="Rank of First Clicked Item",
        labels={"first_pos_rank": "Rank Position", "count": "Impressions"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Errors by Impression Size --
st.subheader("Error Rate by Impression Size")
size_stats = (
    ir.groupby("n_candidates")
    .agg(
        error_rate=("top_correct", lambda x: 1 - x.mean()),
        count=("top_correct", "count"),
    )
    .reset_index()
)

fig = px.bar(
    size_stats[size_stats["count"] >= 10],
    x="n_candidates",
    y="error_rate",
    title="Error Rate by Number of Candidates",
    labels={"n_candidates": "Candidates", "error_rate": "Top-1 Error Rate"},
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Confidence vs Correctness --
st.subheader("Confidence vs. Correctness")
fig = px.scatter(
    ir.sample(min(3000, len(ir))),
    x="max_score",
    y="top_correct",
    opacity=0.2,
    title="Model Confidence vs. Top-1 Correctness",
    labels={"max_score": "Max Prediction Score", "top_correct": "Top-1 Correct"},
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Hardest Impressions --
st.subheader("Hardest Impressions")
st.caption("Impressions with lowest AUC")
hardest = valid_auc.nsmallest(20, "auc")[
    ["impression_id", "auc", "n_candidates", "first_pos_rank"]
]
st.dataframe(hardest.reset_index(drop=True), use_container_width=True)
