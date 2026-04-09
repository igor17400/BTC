"""Page: Dataset Overview — high-level summary of raw MIND data."""

import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector, render_split_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news

st.title("Dataset Overview")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

news_df = load_raw_news(dataset["path"])
if news_df.empty:
    st.warning("No news data found.")
    st.stop()

# -- KPI Cards --
st.subheader("Raw Dataset Summary")

splits_data = {}
total_impressions = 0
total_users = set()
for split in ["train", "valid", "test"]:
    bdf = load_raw_behaviors(dataset["path"], split)
    if not bdf.empty:
        splits_data[split] = bdf
        total_impressions += len(bdf)
        total_users.update(bdf["user_id"].dropna().unique())

render_metric_row(
    {
        "News Articles": len(news_df),
        "Categories": news_df["category"].nunique(),
        "Subcategories": news_df["subcategory"].nunique(),
        "Unique Users": len(total_users),
        "Total Impressions": total_impressions,
    },
    columns=5,
)

# -- Split Size Comparison --
if splits_data:
    st.subheader("Split Sizes")
    split_sizes = {s: len(df) for s, df in splits_data.items()}
    split_users = {s: df["user_id"].nunique() for s, df in splits_data.items()}

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=list(split_sizes.keys()),
            y=list(split_sizes.values()),
            labels={"x": "Split", "y": "Impressions"},
            title="Impressions per Split",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        fig = px.bar(
            x=list(split_users.keys()),
            y=list(split_users.values()),
            labels={"x": "Split", "y": "Users"},
            title="Unique Users per Split",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Label Distribution --
if splits_data:
    st.subheader("Click Distribution")
    label_stats = []
    for split, bdf in splits_data.items():
        imp_col = bdf["impressions"].dropna()
        total_pos = 0
        total_neg = 0
        for imp_str in imp_col:
            for pair in str(imp_str).split():
                if pair.endswith("-1"):
                    total_pos += 1
                elif pair.endswith("-0"):
                    total_neg += 1
        label_stats.append({"split": split, "label": "clicked", "count": total_pos})
        label_stats.append({"split": split, "label": "not clicked", "count": total_neg})

    fig = px.bar(
        pd.DataFrame(label_stats),
        x="split",
        y="count",
        color="label",
        barmode="group",
        title="Clicked vs. Not Clicked per Split",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- History Length Distribution --
if splits_data:
    st.subheader("User History Length")
    split = render_split_selector(list(splits_data.keys()))
    bdf = splits_data.get(split)
    if bdf is not None:
        hist_lengths = bdf["history"].dropna().apply(lambda x: len(str(x).split()))
        fig = px.histogram(
            hist_lengths,
            nbins=50,
            labels={"value": "History Length", "count": "Users"},
            title=f"History Length Distribution ({split})",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{hist_lengths.mean():.1f}")
        col2.metric("Median", f"{hist_lengths.median():.0f}")
        col3.metric("Max", f"{hist_lengths.max()}")
