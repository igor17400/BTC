"""Page 3: Temporal Analysis — time dynamics of the dataset."""

import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.plotly_theme import apply_defaults, build_category_color_map
from vewsx.components.sidebar import render_dataset_selector, render_split_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news

st.title("Temporal Analysis")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

split = render_split_selector(dataset.get("splits"))
behaviors_df = load_raw_behaviors(dataset["path"], split)
news_df = load_raw_news(dataset["path"])

if behaviors_df.empty:
    st.warning(f"No behavior data for split '{split}'.")
    st.stop()

# -- Impression Volume Over Time --
st.subheader("Impression Volume Over Time")
time_col = behaviors_df["time"].dropna()
if not time_col.empty:
    hourly = time_col.dt.floor("h").value_counts().sort_index()
    fig = px.line(
        x=hourly.index,
        y=hourly.values,
        labels={"x": "Time", "y": "Impressions"},
        title="Impressions per Hour",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)
else:
    st.info("No timestamp data available.")

# -- Day-of-Week / Hour-of-Day Patterns --
if not time_col.empty:
    st.subheader("Activity Patterns")
    col1, col2 = st.columns(2)

    with col1:
        dow = time_col.dt.day_name().value_counts()
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        dow = dow.reindex(day_order).fillna(0)
        fig = px.bar(
            x=dow.index,
            y=dow.values,
            labels={"x": "Day", "y": "Impressions"},
            title="Impressions by Day of Week",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        hod = time_col.dt.hour.value_counts().sort_index()
        fig = px.bar(
            x=hod.index,
            y=hod.values,
            labels={"x": "Hour", "y": "Impressions"},
            title="Impressions by Hour of Day",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Category Trends Over Time --
if not time_col.empty and not news_df.empty:
    st.subheader("Category Trends Over Time")

    # Parse impressions to extract news IDs and join with category
    impression_records = []
    for _, row in behaviors_df.dropna(subset=["impressions", "time"]).iterrows():
        ts = row["time"]
        for pair in str(row["impressions"]).split():
            nid = pair.rsplit("-", 1)[0]
            impression_records.append({"time": ts, "news_id": nid})

    if impression_records:
        imp_df = pd.DataFrame(impression_records)
        imp_df = imp_df.merge(
            news_df[["id", "category"]].rename(columns={"id": "news_id"}),
            on="news_id",
            how="left",
        )
        imp_df["date"] = imp_df["time"].dt.date

        daily_cats = (
            imp_df.groupby(["date", "category"]).size().reset_index(name="count")
        )

        categories = daily_cats["category"].dropna().unique().tolist()
        color_map = build_category_color_map(categories)

        fig = px.area(
            daily_cats,
            x="date",
            y="count",
            color="category",
            color_discrete_map=color_map,
            title="Category Volume Over Time",
            labels={"date": "Date", "count": "Impressions"},
        )
        st.plotly_chart(apply_defaults(fig, height=500), use_container_width=True)

# -- News Freshness at Impression Time --
if not time_col.empty and not news_df.empty:
    st.subheader("Article Freshness")
    st.caption("How old are articles when users see them?")

    # Estimate publish time as earliest appearance in any behavior
    earliest_seen = {}
    for _, row in behaviors_df.dropna(subset=["time"]).iterrows():
        ts = row["time"]
        # From history
        if pd.notna(row.get("history")):
            for nid in str(row["history"]).split():
                if nid not in earliest_seen or ts < earliest_seen[nid]:
                    earliest_seen[nid] = ts
        # From impressions
        if pd.notna(row.get("impressions")):
            for pair in str(row["impressions"]).split():
                nid = pair.rsplit("-", 1)[0]
                if nid not in earliest_seen or ts < earliest_seen[nid]:
                    earliest_seen[nid] = ts

    # Compute age at impression time for a sample
    ages_hours = []
    sample = behaviors_df.dropna(subset=["impressions", "time"]).head(5000)
    for _, row in sample.iterrows():
        ts = row["time"]
        for pair in str(row["impressions"]).split():
            nid = pair.rsplit("-", 1)[0]
            pub = earliest_seen.get(nid)
            if pub and ts > pub:
                age_h = (ts - pub).total_seconds() / 3600
                if age_h < 500:  # cap outliers
                    ages_hours.append(age_h)

    if ages_hours:
        fig = px.histogram(
            ages_hours,
            nbins=50,
            labels={"value": "Age (hours)", "count": "Impressions"},
            title="Article Age When Shown to Users",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)
