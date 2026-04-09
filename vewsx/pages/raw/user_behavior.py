"""Page 4: User Behavior Analysis — reading patterns and engagement."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector, render_split_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news

st.title("User Behavior Analysis")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

split = render_split_selector(dataset.get("splits"))
behaviors_df = load_raw_behaviors(dataset["path"], split)
news_df = load_raw_news(dataset["path"])

if behaviors_df.empty:
    st.warning(f"No behavior data for split '{split}'.")
    st.stop()

# -- History Length Distribution --
st.subheader("History Length Distribution")
hist_lengths = behaviors_df["history"].dropna().apply(lambda x: len(str(x).split()))

fig = px.histogram(
    hist_lengths,
    nbins=50,
    labels={"value": "History Length", "count": "Impressions"},
    title="Number of Previously Read Articles per Impression",
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Reading Diversity (Category Entropy per User) --
st.subheader("Reading Diversity")
st.caption("Shannon entropy of category distribution per user (higher = more diverse)")

if not news_df.empty:
    news_cat_map = dict(zip(news_df["id"], news_df["category"]))

    user_cats = {}
    for _, row in behaviors_df.dropna(subset=["history"]).iterrows():
        uid = row["user_id"]
        if uid not in user_cats:
            user_cats[uid] = []
        for nid in str(row["history"]).split():
            cat = news_cat_map.get(nid)
            if cat:
                user_cats[uid].append(cat)

    entropies = []
    for uid, cats in user_cats.items():
        if len(cats) < 2:
            continue
        counts = pd.Series(cats).value_counts(normalize=True)
        entropy = -np.sum(counts * np.log2(counts + 1e-10))
        entropies.append(
            {"user_id": uid, "entropy": entropy, "history_size": len(cats)}
        )

    if entropies:
        entropy_df = pd.DataFrame(entropies)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                entropy_df,
                x="entropy",
                nbins=40,
                title="Category Entropy Distribution",
                labels={"entropy": "Shannon Entropy", "count": "Users"},
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        with col2:
            fig = px.scatter(
                entropy_df.sample(min(2000, len(entropy_df))),
                x="history_size",
                y="entropy",
                opacity=0.3,
                title="Diversity vs. History Size",
                labels={"history_size": "History Size", "entropy": "Category Entropy"},
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Impression-to-Click Funnel --
st.subheader("Click-Through Analysis")
click_data = []
for _, row in behaviors_df.dropna(subset=["impressions"]).iterrows():
    pairs = str(row["impressions"]).split()
    n_total = len(pairs)
    n_clicked = sum(1 for p in pairs if p.endswith("-1"))
    click_data.append({"impression_size": n_total, "clicks": n_clicked})

if click_data:
    click_df = pd.DataFrame(click_data)

    col1, col2 = st.columns(2)
    with col1:
        avg_ctr = click_df.groupby("impression_size")["clicks"].mean().reset_index()
        avg_ctr.columns = ["impression_size", "avg_clicks"]
        fig = px.bar(
            avg_ctr.head(30),
            x="impression_size",
            y="avg_clicks",
            title="Average Clicks by Impression Size",
            labels={"impression_size": "Candidates Shown", "avg_clicks": "Avg Clicks"},
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        fig = px.histogram(
            click_df,
            x="clicks",
            nbins=20,
            title="Clicks per Impression Distribution",
            labels={"clicks": "Number of Clicks", "count": "Impressions"},
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Individual User Explorer --
st.subheader("User Explorer")
user_id = st.text_input("Enter a user ID", placeholder="U12345")
if user_id and not news_df.empty:
    user_rows = behaviors_df[behaviors_df["user_id"] == user_id]
    if user_rows.empty:
        st.warning(f"User '{user_id}' not found in {split} split.")
    else:
        st.caption(f"{len(user_rows)} impressions for user {user_id}")

        # Show history categories
        history_str = user_rows.iloc[0].get("history", "")
        if pd.notna(history_str):
            history_ids = str(history_str).split()
            history_cats = [news_cat_map.get(nid, "unknown") for nid in history_ids]
            cat_counts = pd.Series(history_cats).value_counts()

            fig = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                title=f"Reading History Categories ({len(history_ids)} articles)",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)
