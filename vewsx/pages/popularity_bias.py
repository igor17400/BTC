"""Page 9: Popularity Bias & Fairness — analyze model popularity bias."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector, render_run_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news
from vewsx.data.prediction_loader import load_predictions

st.title("Popularity & Bias Analysis")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

news_df = load_raw_news(dataset["path"])
if news_df.empty:
    st.warning("No news data found.")
    st.stop()

# -- Compute Article Popularity from Behaviors --
st.caption("Computing article popularity from behavior data...")
article_stats = {}
for split in ["train", "valid", "test"]:
    bdf = load_raw_behaviors(dataset["path"], split)
    if bdf.empty:
        continue
    for _, row in bdf.dropna(subset=["impressions"]).iterrows():
        for pair in str(row["impressions"]).split():
            parts = pair.rsplit("-", 1)
            if len(parts) == 2:
                nid, label = parts[0], int(parts[1])
                if nid not in article_stats:
                    article_stats[nid] = {"shown": 0, "clicked": 0}
                article_stats[nid]["shown"] += 1
                article_stats[nid]["clicked"] += label

pop_df = pd.DataFrame(
    [
        {
            "news_id": nid,
            "shown": s["shown"],
            "clicked": s["clicked"],
            "ctr": s["clicked"] / max(s["shown"], 1),
        }
        for nid, s in article_stats.items()
    ]
)

if pop_df.empty:
    st.warning("No impression data to compute popularity.")
    st.stop()

pop_df = pop_df.merge(
    news_df[["id", "category"]].rename(columns={"id": "news_id"}),
    on="news_id",
    how="left",
)

# -- Popularity Distribution --
st.subheader("Article Popularity Distribution")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        pop_df,
        x="shown",
        nbins=50,
        log_y=True,
        title="Impressions per Article (log scale)",
        labels={"shown": "Times Shown", "count": "Articles"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

with col2:
    fig = px.histogram(
        pop_df,
        x="ctr",
        nbins=50,
        title="CTR Distribution",
        labels={"ctr": "Click-Through Rate", "count": "Articles"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Long-Tail Curve --
st.subheader("Long-Tail Exposure")
sorted_shown = np.sort(pop_df["shown"].values)[::-1]
cum_shown = np.cumsum(sorted_shown) / sorted_shown.sum()
article_pct = np.arange(1, len(cum_shown) + 1) / len(cum_shown)

# Gini coefficient
n = len(sorted_shown)
gini = (
    (
        2 * np.sum(np.arange(1, n + 1) * np.sort(sorted_shown))
        - (n + 1) * np.sum(sorted_shown)
    )
    / (n * np.sum(sorted_shown))
    if sorted_shown.sum() > 0
    else 0
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=article_pct * 100, y=cum_shown * 100, name="Actual", mode="lines")
)
fig.add_trace(
    go.Scatter(
        x=[0, 100],
        y=[0, 100],
        name="Perfect Equality",
        mode="lines",
        line=dict(dash="dash", color="gray"),
    )
)
fig.update_layout(
    title=f"Lorenz Curve (Gini = {gini:.3f})",
    xaxis_title="% of Articles (ranked by popularity)",
    yaxis_title="% of Total Impressions",
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

render_metric_row(
    {
        "Total Articles": len(pop_df),
        "Gini Coefficient": round(gini, 4),
        "Top 10% Gets": f"{cum_shown[int(len(cum_shown) * 0.1)] * 100:.1f}%",
        "Bottom 50% Gets": f"{(cum_shown[-1] - cum_shown[int(len(cum_shown) * 0.5)]) * 100:.1f}%",
    }
)

# -- Category Fairness --
st.subheader("Category Exposure Fairness")
cat_article_dist = pop_df["category"].value_counts(normalize=True)
cat_impression_dist = pop_df.groupby("category")["shown"].sum()
cat_impression_dist = cat_impression_dist / cat_impression_dist.sum()

fairness_df = pd.DataFrame(
    {
        "category": cat_article_dist.index,
        "article_share": cat_article_dist.values,
        "impression_share": [
            cat_impression_dist.get(c, 0) for c in cat_article_dist.index
        ],
    }
)
fairness_df["bias"] = fairness_df["impression_share"] - fairness_df["article_share"]

fig = px.bar(
    fairness_df.sort_values("bias"),
    x="bias",
    y="category",
    orientation="h",
    title="Category Exposure Bias (impression share - article share)",
    labels={"bias": "Bias", "category": "Category"},
    color="bias",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
)
st.plotly_chart(apply_defaults(fig, height=500), use_container_width=True)

# -- Model Prediction Bias (if run selected) --
runs = render_run_selector(multi=False)
if runs and runs[0].prediction_files:
    st.subheader("Model Recommendation Bias")
    run = runs[0]
    pred_file = st.selectbox("Prediction file", run.prediction_files, key="pop_pred")
    pred_df = load_predictions(pred_file)

    if not pred_df.empty:
        # Check which articles get top-ranked by the model
        top_ranked = []
        for _, row in pred_df.iterrows():
            scores = np.array(row["prediction_scores"])
            gt = np.array(row["ground_truth"])
            if len(scores) > 0:
                top_idx = np.argmax(scores)
                top_ranked.append({"top_correct": int(gt[top_idx] == 1)})

        if top_ranked:
            tr_df = pd.DataFrame(top_ranked)
            st.metric(
                "Top-1 Accuracy",
                f"{tr_df['top_correct'].mean():.4f}",
            )
