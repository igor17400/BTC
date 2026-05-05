"""Page: Processed Behaviors — inspect behavior tensors and negative sampling."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.plotly_theme import apply_defaults
from vewsx.components.sidebar import render_dataset_selector
from vewsx.data.loaders import load_processed_behaviors

st.title("Processed Behaviors & Sampling")

dataset = render_dataset_selector()
if not dataset or not dataset.get("processed_path"):
    st.warning("Select a dataset with processed data.")
    st.stop()

split = st.radio(
    "Split", ["train", "val", "test"], horizontal=True, key="behaviors_split"
)
data = load_processed_behaviors(dataset["processed_path"], split)

if data is None:
    st.warning(f"No processed behavior data found for split '{split}'.")
    st.stop()

# =====================================================================
tab_overview, tab_sampling, tab_features = st.tabs(
    ["Tensor Overview", "Negative Sampling", "Feature Analysis"]
)

# =====================================================================
# TAB 1: Tensor Overview
# =====================================================================
with tab_overview:
    st.subheader("Behavior Tensor Shapes")

    shapes = {}
    total_bytes = 0
    for key in sorted(data.keys()):
        val = data[key]
        if hasattr(val, "shape"):
            shapes[key] = {
                "shape": str(val.shape),
                "dtype": str(val.dtype),
                "size_mb": f"{val.nbytes / 1e6:.1f}",
            }
            total_bytes += val.nbytes
        elif hasattr(val, "__len__"):
            shapes[key] = {
                "shape": f"len={len(val)}",
                "dtype": type(val).__name__,
                "size_mb": "—",
            }

    render_metric_row(
        {
            "Samples": data["labels"].shape[0] if "labels" in data else "—",
            "Candidates/Sample": data["labels"].shape[1]
            if "labels" in data and data["labels"].ndim == 2
            else "—",
            "History Length": data["history_news_tokens"].shape[1]
            if "history_news_tokens" in data
            else "—",
            "Total Size": f"{total_bytes / 1e6:.1f} MB",
        }
    )

    st.dataframe(
        pd.DataFrame.from_dict(shapes, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "Field",
                "shape": "Shape",
                "dtype": "DType",
                "size_mb": "Size (MB)",
            }
        ),
        use_container_width=True,
    )

    # Label distribution
    if "labels" in data:
        st.subheader("Label Distribution")
        labels = data["labels"]
        pos_per_sample = labels.sum(axis=1)
        neg_per_sample = labels.shape[1] - pos_per_sample

        col1, col2 = st.columns(2)
        with col1:
            total_pos = int(labels.sum())
            total_neg = int(labels.size - total_pos)
            fig = px.pie(
                values=[total_pos, total_neg],
                names=["Positive (clicked)", "Negative (not clicked)"],
                title="Overall Label Balance",
            )
            st.plotly_chart(apply_defaults(fig, height=350), use_container_width=True)

        with col2:
            fig = px.histogram(
                pos_per_sample,
                nbins=max(int(labels.shape[1]), 5),
                labels={"value": "Positives per Sample", "count": "Samples"},
                title="Positives per Training Sample",
            )
            st.plotly_chart(apply_defaults(fig, height=350), use_container_width=True)

        render_metric_row(
            {
                "Total Positive": total_pos,
                "Total Negative": total_neg,
                "Pos:Neg Ratio": f"1:{total_neg / max(total_pos, 1):.1f}",
                "Avg Positives/Sample": f"{pos_per_sample.mean():.2f}",
            }
        )

# =====================================================================
# TAB 2: Negative Sampling
# =====================================================================
with tab_sampling:
    st.subheader("Negative Sampling Analysis")

    if "labels" not in data or "candidate_news_ids" not in data:
        st.info("No label or candidate data available.")
    else:
        labels = data["labels"]
        cand_ids = data["candidate_news_ids"]
        n_samples, n_candidates = labels.shape

        # Sampling ratio
        pos_counts = labels.sum(axis=1)
        neg_counts = n_candidates - pos_counts

        render_metric_row(
            {
                "Samples": n_samples,
                "Candidates/Sample": n_candidates,
                "Avg Positives": f"{pos_counts.mean():.2f}",
                "Avg Negatives": f"{neg_counts.mean():.2f}",
            }
        )

        # Position of positive label(s)
        st.subheader("Positive Label Position")
        st.caption("Where does the clicked article appear in the candidate list?")
        pos_positions = []
        for i in range(min(n_samples, 50000)):
            for j in range(n_candidates):
                if labels[i, j] == 1:
                    pos_positions.append(j)

        if pos_positions:
            fig = px.histogram(
                pos_positions,
                nbins=n_candidates,
                labels={"value": "Position Index", "count": "Count"},
                title="Position of Positive Label in Candidate List",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        # Candidate news ID overlap
        st.subheader("Candidate Diversity")
        unique_cands = len(np.unique(cand_ids))
        total_cand_slots = cand_ids.size
        st.caption(
            f"{unique_cands:,} unique candidate articles across {total_cand_slots:,} total slots"
        )

        # How often does the same article appear as candidate?
        flat_ids = cand_ids.flatten()
        flat_ids = flat_ids[flat_ids > 0]
        unique_ids, id_counts = np.unique(flat_ids, return_counts=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                id_counts,
                nbins=50,
                log_y=True,
                labels={"value": "Times as Candidate", "count": "Articles (log)"},
                title="How Often Articles Appear as Candidates",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        with col2:
            render_metric_row(
                {
                    "Unique Candidates": unique_cands,
                    "Mean Appearances": f"{id_counts.mean():.1f}",
                    "Max Appearances": int(id_counts.max()),
                    "Median Appearances": int(np.median(id_counts)),
                },
                columns=2,
            )

        # History overlap with candidates
        if "histories_news_ids" in data:
            st.subheader("History-Candidate Overlap")
            st.caption("Do candidates ever appear in the user's own history?")
            overlap_count = 0
            check_n = min(n_samples, 20000)
            hist_ids = data["histories_news_ids"]
            for i in range(check_n):
                hist_set = set(hist_ids[i][hist_ids[i] > 0])
                cand_set = set(cand_ids[i][cand_ids[i] > 0])
                if hist_set & cand_set:
                    overlap_count += 1
            overlap_pct = overlap_count / check_n * 100
            st.metric(
                f"Samples with history-candidate overlap (checked {check_n:,})",
                f"{overlap_count} ({overlap_pct:.1f}%)",
            )

# =====================================================================
# TAB 3: Feature Analysis
# =====================================================================
with tab_features:
    st.subheader("Feature Distributions")

    # CTR features
    if "history_news_ctr" in data:
        st.subheader("History CTR")
        hist_ctr = data["history_news_ctr"]
        nonzero = hist_ctr[hist_ctr > 0]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                nonzero.flatten(),
                nbins=50,
                labels={"value": "CTR", "count": "Count"},
                title="History News CTR Distribution (non-zero)",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        with col2:
            # Average CTR per position in history
            avg_per_pos = hist_ctr.mean(axis=0)
            fig = px.line(
                x=list(range(len(avg_per_pos))),
                y=avg_per_pos,
                labels={"x": "History Position", "y": "Mean CTR"},
                title="Average CTR by History Position",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

    if "candidate_news_ctr" in data:
        st.subheader("Candidate CTR")
        cand_ctr = data["candidate_news_ctr"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                cand_ctr.flatten(),
                nbins=50,
                labels={"value": "CTR", "count": "Count"},
                title="Candidate News CTR Distribution",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        with col2:
            # CTR of positive vs negative candidates
            if "labels" in data:
                pos_ctr = cand_ctr[data["labels"] == 1]
                neg_ctr = cand_ctr[data["labels"] == 0]
                ctr_df = pd.DataFrame(
                    {
                        "CTR": np.concatenate([pos_ctr, neg_ctr]),
                        "Label": ["Clicked"] * len(pos_ctr)
                        + ["Not Clicked"] * len(neg_ctr),
                    }
                )
                fig = px.violin(
                    ctr_df,
                    y="CTR",
                    x="Label",
                    box=True,
                    title="Candidate CTR: Clicked vs Not Clicked",
                )
                st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # Recency features
    if "candidate_news_recency" in data:
        st.subheader("Candidate Recency")
        recency = data["candidate_news_recency"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                recency.flatten(),
                nbins=50,
                labels={"value": "Recency Bucket", "count": "Count"},
                title="Candidate Recency Distribution",
            )
            st.plotly_chart(apply_defaults(fig), use_container_width=True)

        with col2:
            if "labels" in data:
                pos_rec = recency[data["labels"] == 1]
                neg_rec = recency[data["labels"] == 0]
                rec_df = pd.DataFrame(
                    {
                        "Recency": np.concatenate([pos_rec, neg_rec]),
                        "Label": ["Clicked"] * len(pos_rec)
                        + ["Not Clicked"] * len(neg_rec),
                    }
                )
                fig = px.violin(
                    rec_df,
                    y="Recency",
                    x="Label",
                    box=True,
                    title="Candidate Recency: Clicked vs Not Clicked",
                )
                st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # Category distribution in candidates
    if "candidate_news_categories" in data:
        st.subheader("Candidate Categories")
        cand_cats = data["candidate_news_categories"].flatten()
        cand_cats = cand_cats[cand_cats > 0]
        unique, counts = np.unique(cand_cats, return_counts=True)

        fig = px.bar(
            x=[str(u) for u in unique],
            y=counts,
            labels={"x": "Category Index", "y": "Occurrences"},
            title="Category Distribution in Candidate Sets",
        )
        st.plotly_chart(apply_defaults(fig), use_container_width=True)

    # User ID distribution
    if "user_ids" in data:
        st.subheader("User ID Distribution")
        uids = data["user_ids"]
        unique_users = len(np.unique(uids))
        render_metric_row(
            {
                "Total Samples": len(uids),
                "Unique Users": unique_users,
                "Avg Samples/User": f"{len(uids) / max(unique_users, 1):.1f}",
                "Max User ID": int(uids.max()),
            }
        )
