"""Precompute article-level and user-level statistics for VewsX.

Pure computation — no Streamlit dependency. Called by the dataset
preprocessing pipeline and loaded by VewsX at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
ARTICLE_STATS_FILE = "vewsx_article_stats.parquet"
USER_STATS_FILE = "vewsx_user_stats.parquet"


def _iter_behaviors(dataset_path: Path):
    """Yield (split, DataFrame) for each available split."""
    for split, dirname in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        tsv_path = dataset_path / dirname / "behaviors.tsv"
        if tsv_path.exists():
            df = pd.read_csv(
                tsv_path, sep="\t", header=None, names=BEHAVIORS_COLUMNS, na_values=""
            )
            df["time"] = pd.to_datetime(
                df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
            )
            yield split, df


def compute_article_stats(dataset_path: Path) -> pd.DataFrame:
    """Compute per-article behavior statistics across all splits."""
    stats: dict[str, dict] = {}

    for split, bdf in _iter_behaviors(dataset_path):
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
                "first_seen": min(timestamps) if timestamps else pd.NaT,
                "last_seen": max(timestamps) if timestamps else pd.NaT,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_user_stats(
    dataset_path: Path, news_cat_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Compute per-user behavior statistics across all splits."""
    if news_cat_map is None:
        news_cat_map = {}

    stats: dict[str, dict] = {}

    for split, bdf in _iter_behaviors(dataset_path):
        for _, row in bdf.iterrows():
            uid = str(row.get("user_id", ""))
            if not uid:
                continue
            if uid not in stats:
                stats[uid] = {
                    "impressions": 0,
                    "total_shown": 0,
                    "total_clicked": 0,
                    "history_ids": set(),
                    "clicked_ids": set(),
                    "splits": set(),
                    "timestamps": [],
                    "history_cats": [],
                }
            s = stats[uid]
            s["impressions"] += 1
            s["splits"].add(split)

            ts = row.get("time")
            if pd.notna(ts):
                s["timestamps"].append(ts)

            if pd.notna(row.get("history")):
                for nid in str(row["history"]).split():
                    s["history_ids"].add(nid)
                    cat = news_cat_map.get(nid)
                    if cat:
                        s["history_cats"].append(cat)

            if pd.notna(row.get("impressions")):
                for pair in str(row["impressions"]).split():
                    parts = pair.rsplit("-", 1)
                    if len(parts) != 2:
                        continue
                    nid, label = parts[0], parts[1]
                    s["total_shown"] += 1
                    if label == "1":
                        s["total_clicked"] += 1
                        s["clicked_ids"].add(nid)

    rows = []
    for uid, s in stats.items():
        cats = s["history_cats"]
        cat_counts = (
            pd.Series(cats).value_counts(normalize=True)
            if cats
            else pd.Series(dtype=float)
        )
        entropy = (
            float(-np.sum(cat_counts * np.log2(cat_counts + 1e-10)))
            if len(cat_counts) > 1
            else 0.0
        )
        n_categories = len(set(cats)) if cats else 0
        max_cat_frac = float(cat_counts.max()) if len(cat_counts) > 0 else 0.0
        top_category = cat_counts.index[0] if len(cat_counts) > 0 else ""
        timestamps = s["timestamps"]

        rows.append(
            {
                "user_id": uid,
                "impressions": s["impressions"],
                "history_size": len(s["history_ids"]),
                "total_shown": s["total_shown"],
                "total_clicked": s["total_clicked"],
                "ctr": s["total_clicked"] / max(s["total_shown"], 1),
                "unique_clicked": len(s["clicked_ids"]),
                "n_categories": n_categories,
                "entropy": entropy,
                "max_cat_fraction": max_cat_frac,
                "top_category": top_category,
                "splits": ",".join(sorted(s["splits"])),
                "first_seen": min(timestamps) if timestamps else pd.NaT,
                "last_seen": max(timestamps) if timestamps else pd.NaT,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_vewsx_stats(
    dataset_path: Path, news_cat_map: dict[str, str] | None = None
) -> None:
    """Compute and save article + user stats to the processed/ directory.

    Called from ``NewsDatasetBase._process_data()`` during training.
    """
    processed_dir = dataset_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    article_path = processed_dir / ARTICLE_STATS_FILE
    user_path = processed_dir / USER_STATS_FILE

    logger.info("Computing VewsX article statistics...")
    article_df = compute_article_stats(dataset_path)
    if not article_df.empty:
        article_df.to_parquet(article_path, index=False)
        logger.info(f"Saved {len(article_df)} article stats to {article_path}")

    logger.info("Computing VewsX user statistics...")
    user_df = compute_user_stats(dataset_path, news_cat_map)
    if not user_df.empty:
        user_df.to_parquet(user_path, index=False)
        logger.info(f"Saved {len(user_df)} user stats to {user_path}")
