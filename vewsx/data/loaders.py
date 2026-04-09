"""Data loaders for MIND dataset (raw TSV + processed data)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from vewsx.config import BEHAVIORS_COLUMNS, CACHE_DIR, DATA_DIR, NEWS_COLUMNS

# ---------------------------------------------------------------------------
# Raw TSV loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_raw_news(dataset_path: str | None = None) -> pd.DataFrame:
    """Load and deduplicate news.tsv from all splits (train/valid/test).

    Returns a DataFrame with columns: id, category, subcategory, title,
    abstract, url, title_entities, abstract_entities.
    """
    base = Path(dataset_path) if dataset_path else DATA_DIR
    frames = []
    for split_dir in ["train", "valid", "test"]:
        tsv_path = base / split_dir / "news.tsv"
        if tsv_path.exists():
            df = pd.read_csv(
                tsv_path,
                sep="\t",
                header=None,
                names=NEWS_COLUMNS,
                na_values="",
            )
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset="id").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_raw_behaviors(
    dataset_path: str | None = None, split: str = "train"
) -> pd.DataFrame:
    """Load behaviors.tsv for a given split.

    Returns a DataFrame with columns: impression_id, user_id, time,
    history, impressions. The time column is parsed to datetime.
    """
    base = Path(dataset_path) if dataset_path else DATA_DIR
    split_map = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    tsv_path = base / split_map.get(split, split) / "behaviors.tsv"
    if not tsv_path.exists():
        return pd.DataFrame(columns=BEHAVIORS_COLUMNS)
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=BEHAVIORS_COLUMNS,
        na_values="",
    )
    df["time"] = pd.to_datetime(
        df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    return df


# ---------------------------------------------------------------------------
# Processed data loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_processed_news(processed_path: str | None = None) -> dict | None:
    """Load the processed_news pickle.

    Args:
        processed_path: Path to the ``processed/`` directory inside the
            dataset (e.g. ``.data/mind-small/small/processed/``), or
            the global cache directory.
    """
    base = Path(processed_path) if processed_path else CACHE_DIR
    pkl_files = sorted(base.glob("processed_news_thresh*.pkl"))
    if not pkl_files:
        # Fall back to recursive search
        pkl_files = sorted(base.glob("**/processed_news_thresh*.pkl"))
    if not pkl_files:
        return None
    with open(pkl_files[0], "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_processed_behaviors(
    processed_path: str | None = None, split: str = "train"
) -> dict | None:
    """Load processed behavior data pickle for a given split."""
    base = Path(processed_path) if processed_path else CACHE_DIR
    pattern = f"{split}_behaviors_*.pkl"
    pkl_files = sorted(base.glob(pattern))
    if not pkl_files:
        pkl_files = sorted(base.glob(f"**/{pattern}"))
    if not pkl_files:
        return None
    with open(pkl_files[0], "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_popularity_data(processed_path: str | None = None) -> dict:
    """Load popularity/CTR numpy arrays."""
    base = Path(processed_path) if processed_path else CACHE_DIR
    result = {}

    # Aggregate CTR
    ctr_files = sorted(base.glob("**/news_ctr_*.npy"))
    for f in ctr_files:
        if "bucketed" not in f.name:
            result["news_ctr"] = np.load(f)
            break

    # Bucketed CTR
    bucketed_files = sorted(base.glob("**/news_ctr_bucketed_*.npy"))
    if bucketed_files:
        result["news_ctr_bucketed"] = np.load(bucketed_files[0])

    # Publish times
    time_files = sorted(base.glob("**/news_publish_time_*.pkl"))
    if time_files:
        with open(time_files[0], "rb") as f:
            result["news_publish_time"] = pickle.load(f)

    return result


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_datasets(data_dir: str | None = None) -> list[dict]:
    """Scan the data directory for available datasets.

    Handles nested structures like ``.data/mind-small/small/train/``
    by finding directories that contain ``train/behaviors.tsv``.

    Returns a list of dicts with keys: name, path, splits, processed_path.
    """
    base = Path(data_dir) if data_dir else DATA_DIR
    datasets = []
    if not base.exists():
        return datasets

    # Find all behaviors.tsv files and work backwards to find dataset roots
    seen_roots = set()
    for behaviors_tsv in sorted(base.rglob("behaviors.tsv")):
        # The dataset root is the parent of the split dir (train/valid/test)
        split_dir = behaviors_tsv.parent  # e.g. .data/mind-small/small/train
        dataset_root = split_dir.parent  # e.g. .data/mind-small/small

        if dataset_root in seen_roots:
            continue
        seen_roots.add(dataset_root)

        splits = []
        for split_name in ["train", "valid", "test"]:
            if (dataset_root / split_name / "behaviors.tsv").exists():
                splits.append(split_name)

        # Build a readable name from the path relative to base
        rel = dataset_root.relative_to(base)
        name = str(rel).replace("/", "-")

        # Check for processed data
        processed_path = dataset_root / "processed"
        entry = {
            "name": name,
            "path": str(dataset_root),
            "splits": splits,
        }
        if processed_path.exists():
            entry["processed_path"] = str(processed_path)

        datasets.append(entry)

    return datasets
