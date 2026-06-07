"""Audit processed/ dataset caches: report duplicates and orphans.

Walks every ``.data/<dataset>/<version>/processed/`` directory, reads the
``processed_meta_*.json`` files, groups them by their *effective* cache
key (the one ``dataset.py`` would compute today — i.e. ignoring the
model-input flags that don't change pkl contents), and prints which
caches are redundant under the new key.

Read-only by default. Pass ``--print-rm`` to emit ``rm`` commands you
can paste into a shell; ``--apply`` to actually delete the redundant
files (asks for confirmation).

Usage::

    python scripts/audit_preprocess_cache.py
    python scripts/audit_preprocess_cache.py --print-rm
    python scripts/audit_preprocess_cache.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# Mirror dataset.py:_cache_key_fields. Keeping this in sync with the
# class is a known maintenance cost — but the script is meant to be
# usable without importing the project (no torch/jax needed).
KEYED_FIELDS = (
    "max_history_length",
    "max_impressions_length",
    "max_title_length",
    "max_abstract_length",
    "process_entities",
    "max_entities",
    "popularity_ctr_method",
    "popularity_bucket_hours",
    "popularity_max_buckets",
    "popularity_ctr_smoothing",
    "random_train_samples",
    "validation_split_strategy",
    "validation_split_percentage",
    "validation_split_seed",
    "sampling_strategy",
)


def compute_new_key(meta: dict) -> str | None:
    """Return the cache key this meta WOULD have under the new logic.

    The ``meta`` shape is one of:
    - legacy flat dict — every field at the top level.
    - new shape — ``{"keyed": {...}, "informational": {...}, ...}``.

    Returns ``None`` if a required field is missing (treat the cache as
    incompatible / pre-fix).
    """
    src = meta.get("keyed", meta)
    key_data: dict = {}
    for f in KEYED_FIELDS:
        if f not in src:
            # ``sampling_strategy`` may be absent in very old metas — let
            # it default to empty so we still produce a key.
            if f == "sampling_strategy":
                key_data[f] = ""
                continue
            return None
        key_data[f] = src[f]
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()[:10]


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"


def audit_dir(processed_dir: Path) -> tuple[list[Path], list[Path], int]:
    """Return (redundant_files, orphan_metas, total_reclaimable_bytes)."""
    metas = sorted(processed_dir.glob("processed_meta_*.json"))
    if not metas:
        return [], [], 0

    print(f"\n=== {processed_dir} ({len(metas)} meta files) ===")

    # Group: new-key -> list of (old_key, meta_path, has_pkls, size).
    groups: dict[str | None, list[tuple]] = defaultdict(list)
    for meta_path in metas:
        old_key = meta_path.stem.removeprefix("processed_meta_")
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        new_key = compute_new_key(meta)
        train_pkl = processed_dir / f"processed_train_{old_key}.pkl"
        val_pkl = processed_dir / f"processed_val_{old_key}.pkl"
        test_pkl = processed_dir / f"processed_test_{old_key}.pkl"
        size = sum(file_size(p) for p in (train_pkl, val_pkl, test_pkl, meta_path))
        has_pkls = train_pkl.exists()
        groups[new_key].append((old_key, meta_path, has_pkls, size, meta))

    redundant: list[Path] = []
    orphans: list[Path] = []
    reclaim = 0

    for new_key, entries in groups.items():
        # Orphan metas: meta exists but no train pkl. Always safe to drop.
        for _old_key, meta_path, has_pkls, size, _meta in entries:
            if not has_pkls:
                orphans.append(meta_path)
                reclaim += size

        with_pkls = [e for e in entries if e[2]]
        if len(with_pkls) <= 1:
            continue

        # Pick a survivor: prefer one whose old_key already equals new_key
        # (idempotent — already on the new scheme). Otherwise the most
        # recently modified train pkl, so we keep the freshest copy.
        def sort_key(entry, _new_key=new_key):
            old_key, _, _, _, _ = entry
            train = processed_dir / f"processed_train_{old_key}.pkl"
            return (old_key == _new_key, train.stat().st_mtime)

        with_pkls.sort(key=sort_key)
        survivor = with_pkls[-1]
        losers = with_pkls[:-1]
        keep_key, _, _, _, keep_meta = survivor
        print(
            f"  new_key={new_key}: keep '{keep_key}' "
            f"(split={keep_meta.get('keyed', keep_meta).get('validation_split_strategy', '?')}, "
            f"title={keep_meta.get('keyed', keep_meta).get('max_title_length', '?')}, "
            f"abstr={keep_meta.get('keyed', keep_meta).get('max_abstract_length', '?')}, "
            f"entities={keep_meta.get('keyed', keep_meta).get('process_entities', '?')})"
        )
        for old_key, meta_path, _, size, _ in losers:
            print(f"    redundant: '{old_key}' ({human(size)})")
            for suffix in ("train", "val", "test"):
                p = processed_dir / f"processed_{suffix}_{old_key}.pkl"
                if p.exists():
                    redundant.append(p)
            redundant.append(meta_path)
            reclaim += size

    if orphans:
        print(f"  {len(orphans)} orphan meta(s) (no matching pkl)")

    return redundant, orphans, reclaim


def find_data_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    # Project sits at NewsReX/, .data is one level above.
    candidate = here.parent / ".data"
    if candidate.exists():
        return candidate
    candidate2 = here / ".data"
    if candidate2.exists():
        return candidate2
    print(f"No .data/ dir found near {here}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the .data/ root (defaults to auto-detect).",
    )
    parser.add_argument(
        "--print-rm",
        action="store_true",
        help="Print rm commands for files identified as redundant/orphan.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete redundant/orphan files (asks for confirmation).",
    )
    args = parser.parse_args()

    root = args.data_root or find_data_root()
    print(f"Scanning {root} ...")

    total_redundant: list[Path] = []
    total_orphans: list[Path] = []
    total_reclaim = 0

    for processed in sorted(root.glob("*/*/processed")):
        redundant, orphans, reclaim = audit_dir(processed)
        total_redundant.extend(redundant)
        total_orphans.extend(orphans)
        total_reclaim += reclaim

    print()
    print(f"Total redundant pkl/meta files: {len(total_redundant)}")
    print(f"Total orphan metas:             {len(total_orphans)}")
    print(f"Total reclaimable disk:         {human(total_reclaim)}")

    to_remove = total_redundant + total_orphans
    if not to_remove:
        return 0

    if args.print_rm:
        print()
        for p in to_remove:
            print(f"rm {p}")

    if args.apply:
        print()
        answer = input(f"Delete {len(to_remove)} files? [y/N] ").strip().lower()
        if answer == "y":
            for p in to_remove:
                try:
                    p.unlink()
                except OSError as exc:
                    print(f"  failed: {p} ({exc})", file=sys.stderr)
            print(f"Deleted {len(to_remove)} files.")
        else:
            print("Aborted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
