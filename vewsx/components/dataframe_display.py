"""Styled DataFrame display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_paginated_table(
    df: pd.DataFrame, page_size: int = 50, key: str = "table"
) -> None:
    """Render a paginated dataframe with page controls.

    Args:
        df: DataFrame to display.
        page_size: Rows per page.
        key: Unique key for session state.
    """
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    page_key = f"{key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button(
            "Previous", key=f"{key}_prev", disabled=st.session_state[page_key] == 0
        ):
            st.session_state[page_key] -= 1
    with col2:
        st.caption(
            f"Page {st.session_state[page_key] + 1} of {total_pages} ({len(df)} rows)"
        )
    with col3:
        if st.button(
            "Next",
            key=f"{key}_next",
            disabled=st.session_state[page_key] >= total_pages - 1,
        ):
            st.session_state[page_key] += 1

    start = st.session_state[page_key] * page_size
    end = start + page_size
    st.dataframe(df.iloc[start:end], use_container_width=True)


def render_news_article(row: pd.Series) -> None:
    """Render a single news article with formatted display."""
    st.markdown(f"**{row.get('title', 'No title')}**")
    cats = []
    if row.get("category"):
        cats.append(f"`{row['category']}`")
    if row.get("subcategory"):
        cats.append(f"`{row['subcategory']}`")
    if cats:
        st.markdown(" / ".join(cats))
    if row.get("abstract"):
        st.caption(str(row["abstract"])[:300])
