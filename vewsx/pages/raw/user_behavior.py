"""User Behavior Analysis — organized in three tabs.

- General Statistics: corpus-level user behavior distributions.
- Single User: select a user and inspect their full behavior profile.
- User Search: search users by activity level or category preference.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.dataframe_display import render_paginated_table
from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults, build_category_color_map
from vewsx.components.sidebar import render_dataset_selector, render_split_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news, load_user_stats

st.title("User Behavior Analysis")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

news_df = load_raw_news(dataset["path"])
news_cat_map = (
    dict(zip(news_df["id"], news_df["category"])) if not news_df.empty else {}
)
news_subcat_map = (
    dict(zip(news_df["id"], news_df["subcategory"])) if not news_df.empty else {}
)
news_title_map = dict(zip(news_df["id"], news_df["title"])) if not news_df.empty else {}

categories = (
    sorted(news_df["category"].dropna().unique().tolist()) if not news_df.empty else []
)
color_map = build_category_color_map(categories) if categories else {}


user_stats = load_user_stats(dataset["path"])
if user_stats is None:
    st.warning(
        "User stats not found. Run NewsReX training first to generate them "
        "(`uv run python src/train.py experiment=...`)."
    )
    st.stop()
if user_stats.empty:
    st.warning("No user data found.")
    st.stop()

# Corpus-wide averages (fixed, never filtered)
corpus_avg_ctr = user_stats["ctr"].mean()
corpus_avg_impressions = user_stats["impressions"].mean()
corpus_avg_history = user_stats["history_size"].mean()
corpus_avg_entropy = user_stats["entropy"].mean()

# =====================================================================
tab_general, tab_user, tab_search = st.tabs(
    ["General Statistics", "Single User", "User Search"]
)

# =====================================================================
# TAB 1: General Statistics
# =====================================================================
with tab_general:
    split = render_split_selector(dataset.get("splits"))
    behaviors_df = load_raw_behaviors(dataset["path"], split)

    render_metric_row(
        {
            "Total Users": len(user_stats),
            "Mean Impressions": f"{corpus_avg_impressions:.1f}",
            "Mean History Size": f"{corpus_avg_history:.1f}",
            "Mean CTR": f"{corpus_avg_ctr:.4f}",
            "Mean Entropy": f"{corpus_avg_entropy:.2f}",
        },
        columns=5,
    )

    # History Length Distribution
    st.subheader("History Length Distribution")
    if not behaviors_df.empty:
        hist_lengths = (
            behaviors_df["history"].dropna().apply(lambda x: len(str(x).split()))
        )
        fig = px.histogram(
            hist_lengths,
            nbins=50,
            labels={"value": "History Length", "count": "Impressions"},
            title=f"History Length Distribution ({split})",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # Reading Diversity
    st.subheader("Reading Diversity")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            user_stats[user_stats["entropy"] > 0],
            x="entropy",
            nbins=40,
            title="Category Entropy Distribution",
            labels={"entropy": "Shannon Entropy", "count": "Users"},
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        sample = user_stats[user_stats["history_size"] > 0].sample(
            min(2000, len(user_stats))
        )
        fig = px.scatter(
            sample,
            x="history_size",
            y="entropy",
            opacity=0.3,
            title="Diversity vs. History Size",
            labels={"history_size": "History Size", "entropy": "Category Entropy"},
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # Click-Through Analysis
    st.subheader("Click-Through Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            user_stats,
            x="ctr",
            nbins=50,
            title="CTR Distribution Across Users",
            labels={"ctr": "CTR", "count": "Users"},
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        fig = px.scatter(
            user_stats.sample(min(2000, len(user_stats))),
            x="history_size",
            y="ctr",
            opacity=0.3,
            title="CTR vs. History Size",
            labels={"history_size": "History Size", "ctr": "CTR"},
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)


# =====================================================================
# TAB 2: Single User
# =====================================================================
with tab_user:
    # -- Top X Users --
    st.subheader("Top Users")

    col_slider, col_metric, col_min = st.columns(3)
    with col_slider:
        top_x = st.slider("Number of users", 5, 50, 20, key="top_users_slider")
    with col_metric:
        user_metric = st.radio(
            "Metric",
            ["CTR", "Total Clicks", "History Size", "Impressions"],
            horizontal=True,
            key="top_user_metric",
        )
    with col_min:
        min_imp = st.number_input(
            "Min impressions", min_value=1, value=3, key="min_user_imp"
        )

    metric_col_map = {
        "CTR": "ctr",
        "Total Clicks": "total_clicked",
        "History Size": "history_size",
        "Impressions": "impressions",
    }
    sort_col = metric_col_map[user_metric]

    qualified = user_stats[user_stats["impressions"] >= min_imp]
    top_users = qualified.nlargest(top_x, sort_col)

    if not top_users.empty:
        fig = px.bar(
            top_users,
            x=sort_col,
            y="user_id",
            orientation="h",
            color="top_category",
            color_discrete_map=color_map,
            hover_data=[
                "impressions",
                "total_clicked",
                "ctr",
                "history_size",
                "entropy",
            ],
            title=f"Top {top_x} Users by {user_metric} (min {min_imp} impressions)",
            labels={sort_col: user_metric, "user_id": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(
            apply_defaults(fig, height=max(400, top_x * 22)), use_container_width=True
        )

    # -- Single User Detail --
    st.markdown("---")
    st.subheader("User Detail")

    user_labels = user_stats["user_id"].tolist()
    selected_uid = st.selectbox(
        "Select a user",
        user_labels,
        index=None,
        placeholder="Start typing a user ID...",
        key="user_selector",
    )

    if selected_uid:
        r = user_stats[user_stats["user_id"] == selected_uid].iloc[0]

        # Row 1: Core metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Impressions", int(r["impressions"]))
        col2.metric("Total Clicked", int(r["total_clicked"]))
        col3.metric("CTR", f"{r['ctr']:.4f}")
        col4.metric("History Size", int(r["history_size"]))

        # Row 2: Diversity & reach
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Categories Read", int(r["n_categories"]))
        col2.metric("Category Entropy", f"{r['entropy']:.2f}")
        col3.metric("Top Category", r["top_category"] or "—")
        col4.metric("Splits", r["splits"])

        # Row 3: Temporal
        if pd.notna(r.get("first_seen")) and pd.notna(r.get("last_seen")):
            first = pd.Timestamp(r["first_seen"])
            last = pd.Timestamp(r["last_seen"])
            col1, col2, col3 = st.columns(3)
            col1.metric("First Seen", first.strftime("%Y-%m-%d %H:%M"))
            col2.metric("Last Seen", last.strftime("%Y-%m-%d %H:%M"))
            col3.metric("Active Span", str(last - first))

        # Reading history analysis
        st.markdown("---")
        st.subheader("Reading History")

        # Get this user's history from behaviors
        all_history_ids = set()
        all_clicked_ids = set()
        user_impressions = []
        for split_name in ["train", "valid", "test"]:
            bdf = load_raw_behaviors(dataset["path"], split_name)
            if bdf.empty:
                continue
            user_rows = bdf[bdf["user_id"] == selected_uid]
            for _, brow in user_rows.iterrows():
                if pd.notna(brow.get("history")):
                    for nid in str(brow["history"]).split():
                        all_history_ids.add(nid)
                if pd.notna(brow.get("impressions")):
                    imp_time = brow.get("time")
                    for pair in str(brow["impressions"]).split():
                        parts = pair.rsplit("-", 1)
                        if len(parts) == 2:
                            nid, label = parts[0], parts[1]
                            user_impressions.append(
                                {
                                    "news_id": nid,
                                    "clicked": label == "1",
                                    "time": imp_time,
                                    "split": split_name,
                                    "category": news_cat_map.get(nid, "unknown"),
                                    "title": news_title_map.get(nid, nid),
                                }
                            )
                            if label == "1":
                                all_clicked_ids.add(nid)

        # Category pie chart of history
        if all_history_ids and news_cat_map:
            col1, col2 = st.columns(2)
            with col1:
                hist_cats = [
                    news_cat_map.get(nid, "unknown") for nid in all_history_ids
                ]
                cat_counts = pd.Series(hist_cats).value_counts()
                fig = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    title=f"History Categories ({len(all_history_ids)} articles)",
                    color=cat_counts.index,
                    color_discrete_map=color_map,
                )
                st.plotly_chart(
                    apply_defaults(fig, height=350), use_container_width=True
                )

            with col2:
                if all_clicked_ids:
                    click_cats = [
                        news_cat_map.get(nid, "unknown") for nid in all_clicked_ids
                    ]
                    click_counts = pd.Series(click_cats).value_counts()
                    fig = px.pie(
                        values=click_counts.values,
                        names=click_counts.index,
                        title=f"Clicked Categories ({len(all_clicked_ids)} articles)",
                        color=click_counts.index,
                        color_discrete_map=color_map,
                    )
                    st.plotly_chart(
                        apply_defaults(fig, height=350), use_container_width=True
                    )

        # Impression timeline
        if user_impressions:
            imp_df = pd.DataFrame(user_impressions)
            st.subheader("Impression Timeline")
            st.caption(
                f"{len(imp_df)} candidate articles across {imp_df['split'].nunique()} splits"
            )

            # Show clicked articles
            clicked_df = imp_df[imp_df["clicked"]]
            if not clicked_df.empty:
                st.markdown(f"**Clicked articles ({len(clicked_df)}):**")
                render_paginated_table(
                    clicked_df[["title", "category", "split"]].reset_index(drop=True),
                    page_size=15,
                    key="user_clicked",
                )

        # Comparison vs corpus
        st.markdown("---")
        st.caption("Compared to corpus average")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "CTR vs Avg", f"{r['ctr']:.4f}", delta=f"{r['ctr'] - corpus_avg_ctr:+.4f}"
        )
        col2.metric(
            "Impressions vs Avg",
            int(r["impressions"]),
            delta=f"{int(r['impressions'] - corpus_avg_impressions):+d}",
        )
        col3.metric(
            "History vs Avg",
            int(r["history_size"]),
            delta=f"{int(r['history_size'] - corpus_avg_history):+d}",
        )
        col4.metric(
            "Entropy vs Avg",
            f"{r['entropy']:.2f}",
            delta=f"{r['entropy'] - corpus_avg_entropy:+.2f}",
        )

    else:
        st.info("Select a user from the dropdown above.")


# =====================================================================
# TAB 3: User Search
# =====================================================================
with tab_search:
    st.subheader("User Search")
    st.caption("Filter users by activity level, CTR, or category preference.")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_hist = st.slider("Min history size", 0, 200, 0, key="search_min_hist")
    with col2:
        min_ctr_filter = st.slider(
            "Min CTR", 0.0, 1.0, 0.0, step=0.01, key="search_min_ctr"
        )
    with col3:
        cat_filter = st.selectbox(
            "Top category", ["Any"] + categories, key="search_cat_filter"
        )

    filtered = user_stats[
        (user_stats["history_size"] >= min_hist) & (user_stats["ctr"] >= min_ctr_filter)
    ]
    if cat_filter != "Any":
        filtered = filtered[filtered["top_category"] == cat_filter]

    st.markdown(f"**{len(filtered)}** users matching filters")

    if not filtered.empty:
        # Distribution of filtered users
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                filtered,
                x="ctr",
                nbins=40,
                title="CTR Distribution (filtered)",
                labels={"ctr": "CTR", "count": "Users"},
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        with col2:
            top_cats = filtered["top_category"].value_counts().head(15)
            fig = px.bar(
                x=top_cats.values,
                y=top_cats.index,
                orientation="h",
                labels={"x": "Users", "y": "Top Category"},
                title="Dominant Category Distribution (filtered)",
                color=top_cats.index,
                color_discrete_map=color_map,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        # Results table
        st.subheader("Matching Users")
        display_cols = [
            "user_id",
            "impressions",
            "total_clicked",
            "ctr",
            "history_size",
            "n_categories",
            "entropy",
            "top_category",
        ]
        render_paginated_table(
            filtered[display_cols]
            .sort_values("ctr", ascending=False)
            .reset_index(drop=True),
            page_size=20,
            key="user_search_results",
        )
