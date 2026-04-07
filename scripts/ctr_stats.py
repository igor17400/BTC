"""Quick statistics on CTR tables used by PP-Rec."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CACHE = Path("/home/igor/NewsReX/.data/mind-small/small/processed")


def main() -> None:
    method = sys.argv[1] if len(sys.argv) > 1 else "age_bucketed"
    suffix = "" if method == "age_bucketed" else f"_{method}"
    print(f"\n>>> Method: {method}\n")

    ctr_path = CACHE / f"news_ctr{suffix}.npy"
    bucketed_path = CACHE / f"news_ctr_bucketed{suffix}.npy"
    publish_path = CACHE / f"news_publish_time{suffix}.pkl"

    if not ctr_path.exists():
        # Fallback for legacy unsuffixed files
        ctr_path = CACHE / "news_ctr.npy"
        bucketed_path = CACHE / "news_ctr_bucketed.npy"
        publish_path = CACHE / "news_publish_time.pkl"

    ctr = np.load(ctr_path)
    bucketed = np.load(bucketed_path)
    with open(publish_path, "rb") as f:
        publish = pickle.load(f)
    dataset_start_path = CACHE / f"news_dataset_start{suffix}.pkl"
    if dataset_start_path.exists():
        with open(dataset_start_path, "rb") as f:
            dataset_start = pickle.load(f)
        print(f"dataset_start: {dataset_start}")
    print()

    num_news, num_buckets = bucketed.shape
    print(f"Num news:    {num_news:,}")
    print(f"Num buckets: {num_buckets}")
    print()

    print("=" * 70)
    print("Aggregate CTR (news_ctr.npy)")
    print("=" * 70)
    nz = ctr[ctr > 0]
    print(f"  Nonzero news: {len(nz):,} / {num_news:,} ({100 * len(nz) / num_news:.1f}%)")
    print(f"  Min:   {nz.min():.6f}")
    print(f"  Max:   {nz.max():.6f}")
    print(f"  Mean:  {nz.mean():.6f}")
    print(f"  Median:{np.median(nz):.6f}")
    print(f"  Std:   {nz.std():.6f}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        print(f"  P{int(q * 100):>2}:   {np.quantile(nz, q):.6f}")
    print()

    print("=" * 70)
    print("Time-bucketed CTR (news_ctr_bucketed.npy)")
    print("=" * 70)
    nz_cells = bucketed[bucketed > 0]
    total_cells = bucketed.size
    print(f"  Nonzero cells: {len(nz_cells):,} / {total_cells:,} "
          f"({100 * len(nz_cells) / total_cells:.2f}%)")
    print(f"  Bucketed values (nonzero only):")
    print(f"    Min:    {nz_cells.min():.6f}")
    print(f"    Max:    {nz_cells.max():.6f}")
    print(f"    Mean:   {nz_cells.mean():.6f}")
    print(f"    Median: {np.median(nz_cells):.6f}")
    print(f"    Std:    {nz_cells.std():.6f}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        print(f"    P{int(q * 100):>2}:    {np.quantile(nz_cells, q):.6f}")
    print()

    # Bucket usage
    print("Bucket usage across news (which buckets have any data):")
    bucket_use = (bucketed > 0).sum(axis=0)  # per bucket
    active = np.nonzero(bucket_use)[0]
    if len(active) > 0:
        print(f"  First active bucket: {active.min()}  "
              f"(≈ {active.min() * 2} hours = {active.min() * 2 / 24:.1f} days)")
        print(f"  Last active bucket:  {active.max()}  "
              f"(≈ {active.max() * 2} hours = {active.max() * 2 / 24:.1f} days)")
        print(f"  Active buckets:      {len(active)} of {num_buckets}")
    top5 = np.argsort(bucket_use)[-5:][::-1]
    print(f"  Top 5 most-used buckets: {top5.tolist()}  "
          f"(counts: {bucket_use[top5].tolist()})")
    print()

    # Publish time coverage
    print("=" * 70)
    print("Publish times")
    print("=" * 70)
    has_publish = ~pd.isna(publish)
    print(f"  News with publish time: {has_publish.sum():,} / {num_news:,} "
          f"({100 * has_publish.sum() / num_news:.1f}%)")
    if has_publish.sum() > 0:
        pub_ts = pd.to_datetime(publish[has_publish])
        print(f"  Earliest: {pub_ts.min()}")
        print(f"  Latest:   {pub_ts.max()}")
        print(f"  Span:     {(pub_ts.max() - pub_ts.min())}")

    # Compare aggregate vs averaged-over-active-buckets per news
    print()
    print("=" * 70)
    print("Aggregate CTR vs mean-of-active-buckets")
    print("=" * 70)
    active_mask = bucketed > 0
    active_count = active_mask.sum(axis=1)
    bucketed_sum = bucketed.sum(axis=1)
    mean_active = np.where(active_count > 0, bucketed_sum / np.maximum(active_count, 1), 0.0)

    # Correlation on news that have BOTH aggregate and bucketed data
    both = (ctr > 0) & (active_count > 0)
    if both.sum() > 0:
        corr = np.corrcoef(ctr[both], mean_active[both])[0, 1]
        print(f"  News with both: {both.sum():,}")
        print(f"  Correlation (agg vs mean-active): {corr:.4f}")
        diff = ctr[both] - mean_active[both]
        print(f"  Diff (agg − mean_active): "
              f"mean={diff.mean():+.6f}, std={diff.std():.6f}, "
              f"max_abs={np.abs(diff).max():.6f}")
    else:
        print("  No news with both aggregate and bucketed data.")

    # Top news by aggregate CTR, show their bucket distribution
    print()
    print("=" * 70)
    print("Top 5 news by aggregate CTR — bucket distribution")
    print("=" * 70)
    top_idx = np.argsort(ctr)[-5:][::-1]
    for i, idx in enumerate(top_idx):
        row = bucketed[idx]
        nz_buckets = np.nonzero(row)[0]
        print(f"  #{i + 1} news_idx={idx}  agg_ctr={ctr[idx]:.4f}")
        if len(nz_buckets) > 0:
            print(f"      active_buckets: {len(nz_buckets)}  "
                  f"range: {nz_buckets.min()}..{nz_buckets.max()}  "
                  f"min/max/mean bucketed_ctr: {row[nz_buckets].min():.4f}/"
                  f"{row[nz_buckets].max():.4f}/{row[nz_buckets].mean():.4f}")


if __name__ == "__main__":
    main()
