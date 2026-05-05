"""Page: Tensors & Embeddings — inspect processed arrays in detail."""

import numpy as np
import plotly.express as px
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector
from vewsx.data.loaders import load_processed_news

st.title("Tensors & Embeddings")

dataset = render_dataset_selector()
if not dataset or not dataset.get("processed_path"):
    st.warning("Select a dataset with processed data.")
    st.stop()

pn = load_processed_news(dataset["processed_path"])
if pn is None:
    st.warning("Could not load processed_news pickle.")
    st.stop()

# -- Embedding Analysis --
if "embeddings" in pn:
    st.subheader("Word Embeddings")
    emb = pn["embeddings"]
    zero_rows = np.all(emb == 0, axis=1).sum()
    norms = np.linalg.norm(emb, axis=1)

    render_metric_row(
        {
            "Shape": str(emb.shape),
            "Zero Vectors": f"{zero_rows} ({zero_rows / emb.shape[0] * 100:.1f}%)",
            "Coverage": f"{(1 - zero_rows / emb.shape[0]) * 100:.1f}%",
            "Mean Norm": f"{norms[norms > 0].mean():.3f}",
        },
        columns=4,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            norms[norms > 0],
            nbins=50,
            labels={"value": "L2 Norm", "count": "Vectors"},
            title="Embedding Norm Distribution (non-zero)",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    with col2:
        # Variance per dimension
        dim_var = np.var(emb[norms > 0], axis=0)
        fig = px.line(
            x=list(range(len(dim_var))),
            y=dim_var,
            labels={"x": "Dimension", "y": "Variance"},
            title="Per-Dimension Variance",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Title Token Analysis --
if "tokens" in pn:
    st.subheader("Title Tokens")
    tokens = pn["tokens"]
    lengths = np.count_nonzero(tokens, axis=1)

    render_metric_row(
        {
            "Shape": str(tokens.shape),
            "Max Seq Length": tokens.shape[1],
            "Mean Non-Pad": f"{lengths.mean():.1f}",
            "Padding Rate": f"{(1 - lengths.mean() / tokens.shape[1]) * 100:.1f}%",
        },
        columns=4,
    )

    fig = px.histogram(
        lengths,
        nbins=40,
        labels={"value": "Non-padding Tokens", "count": "Articles"},
        title="Title Token Length Distribution",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # Token frequency (which token IDs are most common?)
    st.caption("Most frequent token IDs (excluding padding 0)")
    flat_tokens = tokens.flatten()
    flat_tokens = flat_tokens[flat_tokens > 0]
    unique_ids, counts = np.unique(flat_tokens, return_counts=True)
    top_k = 30
    top_idx = np.argsort(-counts)[:top_k]
    fig = px.bar(
        x=[str(unique_ids[i]) for i in top_idx],
        y=[counts[i] for i in top_idx],
        labels={"x": "Token ID", "y": "Frequency"},
        title=f"Top {top_k} Most Frequent Token IDs",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Abstract Token Analysis --
if "abstract_tokens" in pn:
    st.subheader("Abstract Tokens")
    abs_tokens = pn["abstract_tokens"]
    abs_lengths = np.count_nonzero(abs_tokens, axis=1)

    render_metric_row(
        {
            "Shape": str(abs_tokens.shape),
            "Max Seq Length": abs_tokens.shape[1],
            "Mean Non-Pad": f"{abs_lengths.mean():.1f}",
            "Padding Rate": f"{(1 - abs_lengths.mean() / abs_tokens.shape[1]) * 100:.1f}%",
        },
        columns=4,
    )

    fig = px.histogram(
        abs_lengths,
        nbins=40,
        labels={"value": "Non-padding Tokens", "count": "Articles"},
        title="Abstract Token Length Distribution",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Category / Subcategory Index Distribution --
col1, col2 = st.columns(2)

if "category_indices" in pn:
    with col1:
        st.subheader("Category Indices")
        cat_idx = pn["category_indices"]
        unique, counts = np.unique(cat_idx, return_counts=True)
        fig = px.bar(
            x=[str(u) for u in unique],
            y=counts,
            labels={"x": "Category Index", "y": "Articles"},
            title=f"Articles per Category Index ({len(unique)} categories)",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

if "subcategory_indices" in pn:
    with col2:
        st.subheader("Subcategory Indices")
        subcat_idx = pn["subcategory_indices"]
        unique, counts = np.unique(subcat_idx, return_counts=True)
        # Show top 30 for readability
        top_idx = np.argsort(-counts)[:30]
        fig = px.bar(
            x=[str(unique[i]) for i in top_idx],
            y=[counts[i] for i in top_idx],
            labels={"x": "Subcategory Index", "y": "Articles"},
            title=f"Top 30 Subcategory Indices ({len(unique)} total)",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

# -- Entity Indices --
if "entity_indices" in pn:
    st.subheader("Entity Indices")
    ent = pn["entity_indices"]
    ent_counts = np.count_nonzero(ent, axis=1)

    render_metric_row(
        {
            "Shape": str(ent.shape),
            "Max Entities": ent.shape[1],
            "Mean Entities/Article": f"{ent_counts.mean():.1f}",
            "Articles With Entities": f"{(ent_counts > 0).sum()} ({(ent_counts > 0).mean() * 100:.1f}%)",
        },
        columns=4,
    )

    fig = px.histogram(
        ent_counts,
        nbins=ent.shape[1] + 1,
        labels={"value": "Entity Count", "count": "Articles"},
        title="Entities per Article",
    )
    st.plotly_chart(apply_defaults(fig), use_container_width=True)
