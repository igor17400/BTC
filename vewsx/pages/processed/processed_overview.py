"""Page: Processed Overview — summary of preprocessed tensors and arrays."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from vewsx.components.metrics_cards import render_metric_row
from vewsx.components.sidebar import render_dataset_selector
from vewsx.data.loaders import load_processed_news

st.title("Processed Data Overview")

dataset = render_dataset_selector()
if not dataset:
    st.stop()

processed_path = dataset.get("processed_path")
if not processed_path:
    st.warning(
        "No `processed/` directory found for this dataset. "
        "Run NewsReX training first to generate processed data."
    )
    st.stop()

pn = load_processed_news(processed_path)
if pn is None:
    st.warning("Could not load processed_news pickle.")
    st.stop()

# -- KPI Cards --
st.subheader("Processed Dataset Summary")
num_news = len(pn.get("news_ids_original_strings", []))

render_metric_row(
    {
        "News Articles": num_news,
        "Vocab Size": pn.get("vocab_size", "N/A"),
        "Categories": pn.get("num_categories", "N/A"),
        "Subcategories": pn.get("num_subcategories", "N/A"),
        "Num Users": pn.get("num_users", "N/A"),
    },
    columns=5,
)

# -- Tensor Shapes Table --
st.subheader("Tensor Shapes")
shapes = {}
for key in sorted(pn.keys()):
    val = pn[key]
    if hasattr(val, "shape"):
        shapes[key] = {
            "shape": str(val.shape),
            "dtype": str(val.dtype),
            "size_mb": f"{val.nbytes / 1e6:.1f}",
        }
    elif isinstance(val, list) and len(val) > 0:
        shapes[key] = {
            "shape": f"list[{len(val)}]",
            "dtype": type(val[0]).__name__,
            "size_mb": "—",
        }
    elif isinstance(val, dict):
        shapes[key] = {"shape": f"dict[{len(val)}]", "dtype": "dict", "size_mb": "—"}

if shapes:
    shapes_df = pd.DataFrame.from_dict(shapes, orient="index").reset_index()
    shapes_df.columns = ["Field", "Shape", "DType", "Size (MB)"]
    st.dataframe(shapes_df, use_container_width=True)

# -- Processing Config (from meta JSON) --
st.subheader("Processing Configuration")
meta_files = sorted(Path(processed_path).glob("processed_meta_*.json"))
if meta_files:
    with open(meta_files[0]) as f:
        meta = json.load(f)
    col1, col2 = st.columns(2)
    with col1:
        st.json(meta)
    with col2:
        render_metric_row(
            {
                "Max Title Length": meta.get("max_title_length", "—"),
                "Max History Length": meta.get("max_history_length", "—"),
                "Max Impressions": meta.get("max_impressions_length", "—"),
                "Max Abstract Length": meta.get("max_abstract_length", "—"),
            },
            columns=2,
        )
        st.markdown("**Features enabled:**")
        for feat in [
            "process_title",
            "process_abstract",
            "process_category",
            "process_subcategory",
            "process_user_id",
            "process_entities",
        ]:
            val = meta.get(feat, False)
            icon = "✅" if val else "❌"
            st.markdown(f"- {icon} `{feat}`")
else:
    st.info("No processed_meta JSON found.")

# -- Cached Behavior Files --
st.subheader("Cached Behavior Files")
pkl_files = sorted(Path(processed_path).glob("processed_*.pkl"))
if pkl_files:
    file_info = []
    for f in pkl_files:
        size_mb = f.stat().st_size / 1e6
        file_info.append({"file": f.name, "size_mb": f"{size_mb:.1f}"})
    st.dataframe(pd.DataFrame(file_info), use_container_width=True)
else:
    st.info("No cached behavior pickles found.")
