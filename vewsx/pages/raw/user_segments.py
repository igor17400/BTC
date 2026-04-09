"""Page 5: User Segments — clustering and segmentation analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news

st.title("User Segments")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

news_df = load_raw_news(dataset["path"])
if news_df.empty:
    st.warning("No news data found.")
    st.stop()

news_cat_map = dict(zip(news_df["id"], news_df["category"]))

# Build user profiles across all splits
st.caption("Building user profiles across all splits...")
user_profiles = {}

for split in ["train", "valid", "test"]:
    bdf = load_raw_behaviors(dataset["path"], split)
    if bdf.empty:
        continue
    for _, row in bdf.iterrows():
        uid = row["user_id"]
        if uid not in user_profiles:
            user_profiles[uid] = {
                "impressions": 0,
                "clicks": 0,
                "history_cats": [],
                "splits": set(),
            }
        user_profiles[uid]["impressions"] += 1
        user_profiles[uid]["splits"].add(split)
        if pd.notna(row.get("impressions")):
            for p in str(row["impressions"]).split():
                if p.endswith("-1"):
                    user_profiles[uid]["clicks"] += 1
        if pd.notna(row.get("history")):
            for nid in str(row["history"]).split():
                cat = news_cat_map.get(nid)
                if cat:
                    user_profiles[uid]["history_cats"].append(cat)

if not user_profiles:
    st.warning("No user data found.")
    st.stop()

# Compute features per user
user_features = []
for uid, p in user_profiles.items():
    cats = p["history_cats"]
    n_cats = len(set(cats)) if cats else 0
    history_len = len(cats)
    cat_counts = (
        pd.Series(cats).value_counts(normalize=True) if cats else pd.Series(dtype=float)
    )
    entropy = (
        -np.sum(cat_counts * np.log2(cat_counts + 1e-10)) if len(cat_counts) > 1 else 0
    )
    max_cat_frac = cat_counts.max() if len(cat_counts) > 0 else 0
    ctr = p["clicks"] / max(p["impressions"], 1)

    user_features.append(
        {
            "user_id": uid,
            "history_len": history_len,
            "n_categories": n_cats,
            "entropy": entropy,
            "max_cat_fraction": max_cat_frac,
            "impressions": p["impressions"],
            "ctr": ctr,
            "splits": len(p["splits"]),
        }
    )

uf = pd.DataFrame(user_features)

# -- Activity Distribution --
st.subheader("Activity Distribution")
render_metric_row(
    {
        "Total Users": len(uf),
        "Median History": int(uf["history_len"].median()),
        "Median Impressions": int(uf["impressions"].median()),
        "Mean CTR": round(uf["ctr"].mean(), 4),
    }
)

fig = px.histogram(
    uf,
    x="impressions",
    nbins=50,
    log_y=True,
    title="Impressions per User (log scale)",
    labels={"impressions": "Number of Impressions", "count": "Users"},
)
st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Cold / Warm / Active Users --
st.subheader("User Activity Segments")


def segment(h):
    if h <= 5:
        return "Cold (0-5)"
    elif h <= 20:
        return "Warm (6-20)"
    else:
        return "Active (20+)"


uf["segment"] = uf["history_len"].apply(segment)
seg_stats = (
    uf.groupby("segment")
    .agg(
        count=("user_id", "count"),
        avg_ctr=("ctr", "mean"),
        avg_entropy=("entropy", "mean"),
        avg_impressions=("impressions", "mean"),
    )
    .reset_index()
)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        seg_stats,
        x="segment",
        y="count",
        title="Users per Segment",
        labels={"segment": "Segment", "count": "Users"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

with col2:
    fig = px.bar(
        seg_stats,
        x="segment",
        y="avg_ctr",
        title="Average CTR per Segment",
        labels={"segment": "Segment", "avg_ctr": "CTR"},
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Category Loyalty Scatter --
st.subheader("Category Loyalty")
st.caption("X = categories read, Y = concentration in top category")

sample = uf[uf["history_len"] > 0].sample(min(3000, len(uf[uf["history_len"] > 0])))
fig = px.scatter(
    sample,
    x="n_categories",
    y="max_cat_fraction",
    color="segment",
    opacity=0.4,
    title="Specialist vs. Generalist Users",
    labels={
        "n_categories": "Distinct Categories Read",
        "max_cat_fraction": "Top Category Fraction",
    },
)
st.plotly_chart(apply_defaults(fig, height=500), use_container_width=True)

# -- User Overlap Across Splits --
st.subheader("User Overlap Across Splits")
split_users = {}
for split in ["train", "valid", "test"]:
    bdf = load_raw_behaviors(dataset["path"], split)
    if not bdf.empty:
        split_users[split] = set(bdf["user_id"].dropna().unique())

if len(split_users) >= 2:
    overlap_data = []
    splits = list(split_users.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            common = len(split_users[s1] & split_users[s2])
            overlap_data.append(
                {
                    "pair": f"{s1} ∩ {s2}",
                    "overlap": common,
                    "pct_of_smaller": common
                    / min(len(split_users[s1]), len(split_users[s2]))
                    * 100,
                }
            )
    overlap_df = pd.DataFrame(overlap_data)
    st.dataframe(overlap_df, use_container_width=True)
