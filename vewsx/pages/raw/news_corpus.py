"""News Corpus Explorer — organized in three tabs.

- General Statistics: corpus-level distributions and charts.
- Single Article: select an article and inspect its behavior data.
- Topic Search: search by keyword and explore matching articles.
"""

from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.dataframe_display import (
    render_news_article,
    render_paginated_table,
)
from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults, build_category_color_map
from vewsx.components.sidebar import render_category_filter, render_dataset_selector
from vewsx.data.loaders import load_raw_behaviors, load_raw_news

st.title("News Corpus Explorer")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

news_df = load_raw_news(dataset["path"])
if news_df.empty:
    st.warning("No news data found.")
    st.stop()

categories = news_df["category"].dropna().unique().tolist()
color_map = build_category_color_map(categories)

# =====================================================================
tab_general, tab_article, tab_topic = st.tabs(
    ["General Statistics", "Single Article", "Topic Search"]
)

# =====================================================================
# TAB 1: General Statistics
# =====================================================================
with tab_general:
    selected_cats = render_category_filter(categories)
    filtered = (
        news_df[news_df["category"].isin(selected_cats)] if selected_cats else news_df
    )

    # KPIs
    render_metric_row(
        {
            "Total Articles": len(filtered),
            "Categories": filtered["category"].nunique(),
            "Subcategories": filtered["subcategory"].nunique(),
            "Has Abstract": int(filtered["abstract"].notna().sum()),
        },
    )

    # Category Sunburst
    st.subheader("Category Distribution")
    cat_counts = (
        filtered.groupby(["category", "subcategory"]).size().reset_index(name="count")
    )
    if not cat_counts.empty:
        fig = px.sunburst(
            cat_counts,
            path=["category", "subcategory"],
            values="count",
            title="Category / Subcategory Hierarchy",
        )
        st.plotly_chart(apply_defaults(fig, height=550), use_container_width=True)

    # Title / Abstract Length
    st.subheader("Text Length Analysis")
    col1, col2 = st.columns(2)

    with col1:
        title_lengths = filtered["title"].dropna().apply(lambda x: len(str(x).split()))
        fig = px.histogram(
            title_lengths,
            nbins=40,
            labels={"value": "Word Count", "count": "Articles"},
            title="Title Length Distribution",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        abstract_lengths = (
            filtered["abstract"].dropna().apply(lambda x: len(str(x).split()))
        )
        if not abstract_lengths.empty:
            fig = px.histogram(
                abstract_lengths,
                nbins=40,
                labels={"value": "Word Count", "count": "Articles"},
                title="Abstract Length Distribution",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)
        else:
            st.info("No abstracts available.")

    # Top Categories
    st.subheader("Top Categories")
    cat_series = filtered["category"].value_counts().head(20)
    fig = px.bar(
        x=cat_series.values,
        y=cat_series.index,
        orientation="h",
        labels={"x": "Article Count", "y": "Category"},
        title="Top 20 Categories by Article Count",
        color=cat_series.index,
        color_discrete_map=color_map,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_defaults(fig, height=500), use_container_width=True)

    # Zipf's Law
    st.subheader("Word Frequency (Zipf's Law)")
    all_words = " ".join(filtered["title"].dropna()).lower().split()
    if all_words:
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(500)
        ranks = list(range(1, len(top_words) + 1))
        freqs = [c for _, c in top_words]

        fig = px.scatter(
            x=ranks,
            y=freqs,
            log_x=True,
            log_y=True,
            labels={"x": "Rank", "y": "Frequency"},
            title="Word Rank-Frequency (log-log)",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)


# =====================================================================
# TAB 2: Single Article
# =====================================================================
with tab_article:
    # -- Precompute article-level stats across all splits (cached) --
    @st.cache_data(show_spinner="Computing article statistics...")
    def _compute_article_stats(dataset_path: str) -> pd.DataFrame:
        stats: dict[str, dict] = {}
        for split in ["train", "valid", "test"]:
            bdf = load_raw_behaviors(dataset_path, split)
            if bdf.empty:
                continue
            for _, row in bdf.dropna(subset=["impressions"]).iterrows():
                uid = row.get("user_id", "")
                ts = row.get("time")
                pairs = str(row["impressions"]).split()
                for pos, pair in enumerate(pairs):
                    parts = pair.rsplit("-", 1)
                    if len(parts) != 2:
                        continue
                    nid, label = parts[0], parts[1]
                    if nid not in stats:
                        stats[nid] = {
                            "shown": 0,
                            "clicked": 0,
                            "in_history": 0,
                            "unique_users_shown": set(),
                            "unique_users_clicked": set(),
                            "splits": set(),
                            "positions": [],
                            "timestamps": [],
                        }
                    s = stats[nid]
                    s["shown"] += 1
                    s["splits"].add(split)
                    s["positions"].append(pos)
                    if pd.notna(ts):
                        s["timestamps"].append(ts)
                    if uid:
                        s["unique_users_shown"].add(uid)
                    if label == "1":
                        s["clicked"] += 1
                        if uid:
                            s["unique_users_clicked"].add(uid)

            for _, row in bdf.dropna(subset=["history"]).iterrows():
                for nid in str(row["history"]).split():
                    if nid not in stats:
                        stats[nid] = {
                            "shown": 0,
                            "clicked": 0,
                            "in_history": 0,
                            "unique_users_shown": set(),
                            "unique_users_clicked": set(),
                            "splits": set(),
                            "positions": [],
                            "timestamps": [],
                        }
                    stats[nid]["in_history"] += 1

        rows = []
        for nid, s in stats.items():
            positions = s["positions"]
            timestamps = s["timestamps"]
            rows.append(
                {
                    "id": nid,
                    "shown": s["shown"],
                    "clicked": s["clicked"],
                    "ctr": s["clicked"] / max(s["shown"], 1),
                    "in_history": s["in_history"],
                    "unique_users_shown": len(s["unique_users_shown"]),
                    "unique_users_clicked": len(s["unique_users_clicked"]),
                    "splits": ",".join(sorted(s["splits"])),
                    "avg_position": sum(positions) / len(positions) if positions else 0,
                    "impression_size_avg": 0,  # filled below if needed
                    "first_seen": min(timestamps) if timestamps else pd.NaT,
                    "last_seen": max(timestamps) if timestamps else pd.NaT,
                }
            )
        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "id",
                    "shown",
                    "clicked",
                    "ctr",
                    "in_history",
                    "unique_users_shown",
                    "unique_users_clicked",
                    "splits",
                    "avg_position",
                    "first_seen",
                    "last_seen",
                ]
            )
        )

    article_stats = _compute_article_stats(dataset["path"])

    # Corpus-wide averages (computed once on full unfiltered data)
    corpus_avg_ctr = article_stats["ctr"].mean() if not article_stats.empty else 0
    corpus_avg_shown = article_stats["shown"].mean() if not article_stats.empty else 0
    corpus_avg_history = (
        article_stats["in_history"].mean() if not article_stats.empty else 0
    )

    # Merge with news metadata
    if not article_stats.empty:
        article_stats = article_stats.merge(
            news_df[["id", "title", "category", "subcategory"]], on="id", how="left"
        )

    # -- Top X Articles --
    st.subheader("Top Articles")

    col_slider, col_metric, col_min = st.columns(3)
    with col_slider:
        top_x = st.slider("Number of articles", 5, 50, 20, key="top_x_slider")
    with col_metric:
        chart_metric = st.radio(
            "Metric", ["CTR", "Total Clicks"], horizontal=True, key="top_metric"
        )
    with col_min:
        min_shown = st.number_input(
            "Min impressions", min_value=1, value=5, key="min_shown"
        )

    sort_col = "ctr" if chart_metric == "CTR" else "clicked"

    if not article_stats.empty:
        qualified = article_stats[article_stats["shown"] >= min_shown]
        top_articles = qualified.nlargest(top_x, sort_col)

        if not top_articles.empty:
            fig = px.bar(
                top_articles,
                x=sort_col,
                y="title",
                orientation="h",
                color="category",
                color_discrete_map=color_map,
                hover_data=["id", "shown", "clicked", "ctr", "in_history"],
                title=f"Top {top_x} Articles by {chart_metric} (min {min_shown} impressions)",
                labels={"ctr": "CTR", "clicked": "Total Clicks", "title": ""},
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(
                apply_defaults(fig, height=max(400, top_x * 22)),
                use_container_width=True,
            )

            # Summary stats for top articles
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean CTR", f"{top_articles['ctr'].mean():.4f}")
            col2.metric("Mean Shown", f"{top_articles['shown'].mean():.0f}")
            col3.metric(
                "Top Category",
                top_articles["category"].mode().iloc[0]
                if not top_articles["category"].mode().empty
                else "—",
            )
    else:
        st.info("No behavior data available to compute article stats.")

    # -- Single Article Detail --
    st.markdown("---")
    st.subheader("Article Detail")

    news_options = news_df[["id", "title"]].dropna(subset=["title"])
    article_labels = (
        news_options["title"].str[:80] + "  (" + news_options["id"] + ")"
    ).tolist()

    selected_label = st.selectbox(
        "Select an article",
        article_labels,
        index=None,
        placeholder="Start typing to search...",
        key="article_selector",
    )

    if selected_label:
        selected_id = selected_label.rsplit("(", 1)[-1].rstrip(")")
        article = news_df[news_df["id"] == selected_id].iloc[0]

        # Article metadata
        render_news_article(article)

        col1, col2, col3 = st.columns(3)
        col1.metric("Category", article.get("category", "—"))
        col2.metric("Subcategory", article.get("subcategory", "—"))
        col3.metric("Title Words", len(str(article.get("title", "")).split()))

        if pd.notna(article.get("abstract")):
            st.markdown("**Abstract:**")
            st.caption(str(article["abstract"]))

        if pd.notna(article.get("title_entities")):
            st.markdown("**Title Entities:**")
            st.code(str(article["title_entities"])[:500])

        # Behavior metrics from precomputed stats
        st.markdown("---")
        st.subheader("Behavior Analysis")

        if not article_stats.empty:
            row = article_stats[article_stats["id"] == selected_id]
            if not row.empty:
                r = row.iloc[0]

                # Row 1: Core engagement metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Times Shown", int(r["shown"]))
                col2.metric("Times Clicked", int(r["clicked"]))
                col3.metric("CTR", f"{r['ctr']:.4f}")
                col4.metric("In User Histories", int(r["in_history"]))

                # Row 2: Reach & positioning
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Unique Users (shown)", int(r["unique_users_shown"]))
                col2.metric("Unique Users (clicked)", int(r["unique_users_clicked"]))
                col3.metric("Avg Position in List", f"{r['avg_position']:.1f}")
                col4.metric("Splits", r["splits"])

                # Row 3: Temporal info
                if pd.notna(r.get("first_seen")) and pd.notna(r.get("last_seen")):
                    first = pd.Timestamp(r["first_seen"])
                    last = pd.Timestamp(r["last_seen"])
                    lifespan = last - first
                    col1, col2, col3 = st.columns(3)
                    col1.metric("First Seen", first.strftime("%Y-%m-%d %H:%M"))
                    col2.metric("Last Seen", last.strftime("%Y-%m-%d %H:%M"))
                    col3.metric("Lifespan", str(lifespan))

                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    if r["shown"] > 0:
                        fig = px.pie(
                            values=[int(r["clicked"]), int(r["shown"] - r["clicked"])],
                            names=["Clicked", "Not Clicked"],
                            title="Click Distribution",
                        )
                        st.plotly_chart(
                            apply_defaults(fig, height=300), use_container_width=True
                        )

                with col2:
                    if r["unique_users_shown"] > 0:
                        repeat_rate = r["shown"] / r["unique_users_shown"]
                        fig = px.bar(
                            x=["Unique Users", "Total Impressions"],
                            y=[int(r["unique_users_shown"]), int(r["shown"])],
                            labels={"x": "", "y": "Count"},
                            title=f"Repeat Exposure (avg {repeat_rate:.1f}x per user)",
                        )
                        st.plotly_chart(
                            apply_defaults(fig, height=300), use_container_width=True
                        )

                # Comparison vs corpus
                st.markdown("---")
                st.caption("Compared to corpus average")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "CTR vs Avg",
                    f"{r['ctr']:.4f}",
                    delta=f"{r['ctr'] - corpus_avg_ctr:+.4f}",
                )
                col2.metric(
                    "Shown vs Avg",
                    int(r["shown"]),
                    delta=f"{int(r['shown'] - corpus_avg_shown):+d}",
                )
                col3.metric(
                    "History vs Avg",
                    int(r["in_history"]),
                    delta=f"{int(r['in_history'] - corpus_avg_history):+d}",
                )
            else:
                st.info("This article has no recorded impressions.")
        else:
            st.info("No behavior data available.")
    else:
        st.info("Select an article from the dropdown above.")


# =====================================================================
# TAB 3: Topic Search
# =====================================================================
with tab_topic:
    st.subheader("Topic Search")
    st.caption("Search articles by keyword and explore topic clusters.")

    query = st.text_input(
        "Search keyword",
        placeholder="e.g. election, climate, football...",
        key="topic_search",
    )

    if query:
        # Search in both title and abstract
        title_match = news_df["title"].str.contains(query, case=False, na=False)
        abstract_match = news_df["abstract"].str.contains(query, case=False, na=False)
        matches = news_df[title_match | abstract_match].copy()

        st.markdown(f"**{len(matches)}** articles matching `{query}`")

        if not matches.empty:
            # Category breakdown of matches
            col1, col2 = st.columns(2)
            with col1:
                match_cats = matches["category"].value_counts().head(15)
                fig = px.bar(
                    x=match_cats.values,
                    y=match_cats.index,
                    orientation="h",
                    labels={"x": "Articles", "y": "Category"},
                    title=f"Categories containing '{query}'",
                    color=match_cats.index,
                    color_discrete_map=color_map,
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(apply_defaults(fig), use_container_width=True)

            with col2:
                match_subcats = matches["subcategory"].value_counts().head(15)
                fig = px.bar(
                    x=match_subcats.values,
                    y=match_subcats.index,
                    orientation="h",
                    labels={"x": "Articles", "y": "Subcategory"},
                    title=f"Subcategories containing '{query}'",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(apply_defaults(fig), use_container_width=True)

            # Co-occurring words in matching titles
            st.subheader("Co-occurring Terms")
            match_words = " ".join(matches["title"].dropna()).lower().split()
            # Remove the query term itself and common stopwords
            stopwords = {
                "the",
                "a",
                "an",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "and",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "has",
                "have",
                "had",
                "it",
                "its",
                "this",
                "that",
                "with",
                "from",
                "by",
                "as",
                "or",
                "not",
                "but",
                "he",
                "she",
                "his",
                "her",
                "they",
                "their",
                "we",
                "you",
                "your",
                "my",
                "i",
                "me",
                "us",
                "him",
                "who",
                "what",
                "how",
                "when",
                "where",
                "why",
                "will",
                "can",
                "do",
                "did",
                "about",
                "after",
                "all",
                "also",
                "just",
                "than",
                "more",
                "new",
                "out",
                "up",
                "over",
                "no",
                "so",
                "if",
                "-",
                "—",
                "'s",
            }
            query_lower = query.lower()
            filtered_words = [
                w
                for w in match_words
                if w not in stopwords and w != query_lower and len(w) > 2
            ]
            if filtered_words:
                co_counts = Counter(filtered_words).most_common(30)
                fig = px.bar(
                    x=[c for _, c in co_counts],
                    y=[w for w, _ in co_counts],
                    orientation="h",
                    labels={"x": "Frequency", "y": "Term"},
                    title=f"Top co-occurring terms with '{query}'",
                )
                fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(
                    apply_defaults(fig, height=500), use_container_width=True
                )

            # Results table
            st.subheader("Matching Articles")
            render_paginated_table(
                matches[["id", "category", "subcategory", "title"]].reset_index(
                    drop=True
                ),
                page_size=20,
                key="topic_results",
            )
    else:
        st.info("Enter a keyword above to search the corpus.")
