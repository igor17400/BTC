"""Dataset preparation utilities for MIND-format news datasets.

Provides three composable functions:
    - ``combine_news_splits`` — merge train/valid news TSVs, deduplicate
    - ``infer_publication_dates`` — extract earliest appearance from behaviors
    - ``enrich_news_with_dates`` — left-join news with inferred dates

And a convenience pipeline:
    - ``prepare_mind_data`` — runs all three in sequence
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NEWS_COLUMNS = [
    "id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


# ---------------------------------------------------------------------------
# 1. Combine news splits
# ---------------------------------------------------------------------------


def combine_news_splits(
    train_path: str | Path,
    valid_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Merge train and valid news TSVs, keeping only unique articles.

    Articles that appear in both splits are kept from the training set;
    validation-only articles are appended.

    Returns:
        The combined DataFrame.
    """
    df_train = pd.read_table(train_path, header=None, names=NEWS_COLUMNS)
    df_valid = pd.read_table(valid_path, header=None, names=NEWS_COLUMNS)

    valid_only_ids = set(df_valid["id"]) - set(df_train["id"])
    df_valid_unique = df_valid[df_valid["id"].isin(valid_only_ids)]
    df_combined = pd.concat([df_train, df_valid_unique], ignore_index=True)

    logger.info(
        "Combined news: %d train + %d valid-only = %d total (%d overlaps removed)",
        len(df_train),
        len(df_valid_unique),
        len(df_combined),
        len(df_valid) - len(df_valid_unique),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(out, sep="\t", index=False, header=False)
    logger.info("Saved combined news to %s", out)
    return df_combined


# ---------------------------------------------------------------------------
# 2. Infer publication dates
# ---------------------------------------------------------------------------


def _parse_timestamp(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%m/%d/%Y %I:%M:%S %p")
    except (ValueError, TypeError):
        return None


def _process_behaviors_file(path: str | Path) -> dict[str, list]:
    """Return {news_id: [timestamps]} from a behaviors TSV."""
    timestamps: dict[str, list] = defaultdict(list)
    total = processed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            total += 1
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            _, _, ts_str, history, impressions = parts[:5]
            ts = _parse_timestamp(ts_str)
            if ts is None:
                continue

            news_ids: set[str] = set()
            if history and history.strip():
                news_ids.update(history.split())
            if impressions and impressions.strip():
                for item in impressions.split():
                    if "-" in item:
                        news_ids.add(item.split("-")[0])

            for nid in news_ids:
                timestamps[nid].append(ts)
            processed += 1

    logger.info("Processed %d/%d rows from %s", processed, total, path)
    return timestamps


def infer_publication_dates(
    train_behaviors_path: str | Path,
    valid_behaviors_path: str | Path,
    output_path: str | Path,
) -> dict[str, datetime]:
    """Infer news publication dates as earliest appearance in behaviors.

    Returns:
        Dict mapping news_id to its earliest timestamp.
    """
    merged: dict[str, list] = defaultdict(list)
    for path in (train_behaviors_path, valid_behaviors_path):
        for nid, tss in _process_behaviors_file(path).items():
            merged[nid].extend(tss)

    pub_dates = {nid: min(tss) for nid, tss in merged.items()}
    logger.info("Inferred dates for %d news articles", len(pub_dates))

    df = pd.DataFrame(
        [{"news_id": k, "publication_date": v.isoformat()} for k, v in pub_dates.items()]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved publication dates to %s", out)
    return pub_dates


# ---------------------------------------------------------------------------
# 3. Enrich news with dates
# ---------------------------------------------------------------------------


def enrich_news_with_dates(
    news_path: str | Path,
    dates_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Left-join news articles with inferred publication dates.

    Returns:
        The enriched DataFrame.
    """
    df_news = pd.read_table(news_path, header=None, names=NEWS_COLUMNS)
    df_dates = pd.read_csv(dates_path)

    df = pd.merge(df_news, df_dates, left_on="id", right_on="news_id", how="left")
    df = df.drop(columns=["news_id"])

    with_dates = df["publication_date"].notna().sum()
    logger.info(
        "Enriched %d articles: %d with dates, %d without",
        len(df),
        with_dates,
        len(df) - with_dates,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, header=False, na_rep="")
    logger.info("Saved enriched news to %s", out)
    return df


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------


def prepare_mind_data(
    train_news: str | Path,
    valid_news: str | Path,
    train_behaviors: str | Path,
    valid_behaviors: str | Path,
    output_dir: str | Path,
) -> None:
    """Run the full data preparation pipeline.

    Steps:
        1. Combine train/valid news (deduplicate)
        2. Infer publication dates from behaviors
        3. Enrich combined news with dates

    Args:
        train_news: Path to train/news.tsv
        valid_news: Path to valid/news.tsv
        train_behaviors: Path to train/behaviors.tsv
        valid_behaviors: Path to valid/behaviors.tsv
        output_dir: Directory for all output files
    """
    out = Path(output_dir)
    combined_path = out / "non_overlapping_news.tsv"
    dates_path = out / "news_publication_dates.csv"
    enriched_path = out / "news_with_dates.tsv"

    combine_news_splits(train_news, valid_news, combined_path)
    infer_publication_dates(train_behaviors, valid_behaviors, dates_path)
    enrich_news_with_dates(combined_path, dates_path, enriched_path)

    logger.info("Data preparation pipeline complete. Output: %s", out)
