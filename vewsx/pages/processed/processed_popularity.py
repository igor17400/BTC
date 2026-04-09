"""Page: Popularity Features — CTR, recency, and bucketed popularity data."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector
from vewsx.data.loaders import load_popularity_data, load_processed_news

st.title("Popularity Features")

dataset = render_dataset_selector()
if not dataset or not dataset.get("processed_path"):
    st.warning("Select a dataset with processed data.")
    st.stop()

processed_path = dataset["processed_path"]
pn = load_processed_news(processed_path)
pop = load_popularity_data(processed_path)

if not pop:
    st.warning("No popularity data found. PP-Rec features may not have been computed.")
    st.stop()

# -- Aggregate CTR --
if "news_ctr" in pop:
    st.subheader("Aggregate CTR")
    ctr = pop["news_ctr"]
    nonzero = ctr[ctr > 0]

    render_metric_row(
        {
            "Total Articles": len(ctr),
            "With CTR > 0": len(nonzero),
            "Mean CTR": f"{nonzero.mean():.4f}" if len(nonzero) > 0 else "—",
            "Median CTR": f"{np.median(nonzero):.4f}" if len(nonzero) > 0 else "—",
            "Max CTR": f"{ctr.max():.4f}",
        },
        columns=5,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            nonzero,
            nbins=50,
            labels={"value": "CTR", "count": "Articles"},
            title="CTR Distribution (articles with CTR > 0)",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        fig = px.histogram(
            nonzero,
            nbins=50,
            log_y=True,
            labels={"value": "CTR", "count": "Articles (log)"},
            title="CTR Distribution (log scale)",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # CTR percentiles
    if len(nonzero) > 0:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(nonzero, percentiles)
        st.caption("CTR Percentiles")
        st.dataframe(
            pd.DataFrame(
                {"Percentile": [f"P{p}" for p in percentiles], "CTR": pct_values}
            ),
            use_container_width=True,
        )

# -- Bucketed CTR --
if "news_ctr_bucketed" in pop:
    st.subheader("Time-Bucketed CTR")
    bucketed = pop["news_ctr_bucketed"]

    render_metric_row(
        {
            "Shape": str(bucketed.shape),
            "Num Articles": bucketed.shape[0],
            "Num Buckets": bucketed.shape[1],
            "Non-zero Entries": f"{(bucketed > 0).sum():,}",
            "Sparsity": f"{(1 - (bucketed > 0).sum() / bucketed.size) * 100:.1f}%",
        },
        columns=5,
    )

    # Average CTR per bucket (across all articles)
    avg_per_bucket = bucketed.mean(axis=0)
    fig = px.line(
        x=list(range(len(avg_per_bucket))),
        y=avg_per_bucket,
        labels={"x": "Time Bucket", "y": "Mean CTR"},
        title="Average CTR per Time Bucket (across all articles)",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # Heatmap of a sample of articles
    st.caption("CTR Heatmap (sample of articles with highest CTR)")
    article_total_ctr = bucketed.sum(axis=1)
    top_articles = np.argsort(-article_total_ctr)[:50]
    sample = bucketed[top_articles, :200]  # first 200 buckets

    fig = px.imshow(
        sample,
        labels=dict(x="Time Bucket", y="Article (sorted by total CTR)", color="CTR"),
        title="Bucketed CTR Heatmap (top 50 articles, first 200 buckets)",
        aspect="auto",
    )
    st.plotly_chart(apply_defaults(fig, height=500), use_container_width=True)

    # CTR decay: average CTR by bucket age for articles that exist in each bucket
    st.subheader("CTR Decay Over Time")
    active_per_bucket = (bucketed > 0).sum(axis=0)
    sum_per_bucket = bucketed.sum(axis=0)
    avg_active_ctr = np.divide(
        sum_per_bucket,
        active_per_bucket,
        out=np.zeros_like(sum_per_bucket),
        where=active_per_bucket > 0,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(min(300, len(avg_active_ctr)))),
            y=avg_active_ctr[:300],
            mode="lines",
            name="Avg CTR (active articles)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(min(300, len(active_per_bucket)))),
            y=active_per_bucket[:300],
            mode="lines",
            name="Active articles",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="CTR Decay and Article Activity Over Buckets",
        xaxis_title="Time Bucket",
        yaxis_title="Avg CTR",
        yaxis2=dict(title="Active Articles", overlaying="y", side="right"),
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Publish Times --
if "news_publish_time" in pop:
    st.subheader("Publish Times")
    pub_times = pop["news_publish_time"]

    if isinstance(pub_times, dict):
        # JSON format: {news_id: ISO timestamp}
        times = pd.to_datetime(list(pub_times.values()), errors="coerce")
        st.caption(f"{len(pub_times)} articles with publish times")
    elif isinstance(pub_times, np.ndarray):
        times = pd.to_datetime(pub_times, errors="coerce")
        st.caption(f"{len(pub_times)} articles with publish times")
    else:
        times = pd.Series(dtype="datetime64[ns]")

    valid_times = times.dropna()
    if not valid_times.empty:
        hourly = valid_times.dt.floor("h").value_counts().sort_index()
        fig = px.line(
            x=hourly.index,
            y=hourly.values,
            labels={"x": "Time", "y": "Articles Published"},
            title="Article Publication Timeline",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Earliest", str(valid_times.min().date()))
        col2.metric("Latest", str(valid_times.max().date()))
        col3.metric("Span", f"{(valid_times.max() - valid_times.min()).days} days")
