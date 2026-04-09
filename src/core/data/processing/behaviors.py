"""Behavior processing utilities.

Standalone functions extracted from NewsDatasetBase for filtering, sampling,
and converting raw user-behavior TSV data into numerical arrays ready for
model consumption.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_at_least_one_pos_neg_samples(
    impressions_item,
    available_news_ids: set[int],
    parse_news_id: Callable[[str], int],
    progress_callback: Callable | None = None,
) -> bool:
    """Check whether an impressions string contains at least one positive and one negative sample.

    Args:
        impressions_item: Space-separated impressions string (e.g. ``"N123-1 N456-0"``).
        available_news_ids: Set of known integer news IDs.
        parse_news_id: Function to convert a string news ID to an integer.
        progress_callback: Optional callable invoked once before returning.

    Returns:
        ``True`` if both a positive (label ``1``) and a negative (label ``0``)
        sample exist among valid news IDs.
    """
    try:
        impressions_list = str(impressions_item).split()

        if len(impressions_list) == 0:
            if progress_callback:
                progress_callback()
            return False

        has_positive = False
        has_negative = False

        for item in impressions_list:
            if isinstance(item, str) and "-" in item:
                parts = item.split("-")
                news_id, label = parts[0], parts[1]
                news_id_int = parse_news_id(news_id)
                if news_id_int in available_news_ids and len(parts) >= 2:
                    if label == "1":
                        has_positive = True
                    elif label == "0":
                        has_negative = True

            if has_positive and has_negative:
                if progress_callback:
                    progress_callback()
                return True

        result = has_positive and has_negative
        if progress_callback:
            progress_callback()
        return result

    except Exception as e:
        logger.warning(
            f"Failed to check positive/negative samples for {impressions_item}: {e}"
        )
        if progress_callback:
            progress_callback()
        return False


# ---------------------------------------------------------------------------
# Core behaviour processing
# ---------------------------------------------------------------------------


def process_behaviors(
    behaviors_df: pd.DataFrame,
    stage: str,
    processed_news: dict[str, np.ndarray],
    parse_news_id: Callable[[str], int],
    parse_user_id: Callable[[str], int],
    sampler,
    max_history_length: int,
    max_title_length: int,
    max_abstract_length: int,
    random_train_samples: bool = False,
    float_dtype: str = "float32",
    console: Console | None = None,
) -> dict[str, np.ndarray | list]:
    """Process a behaviours DataFrame into arrays of token sequences and labels.

    Filters out rows without at least one positive and one negative impression,
    then iterates over remaining rows to build history and candidate arrays.

    Args:
        behaviors_df: Raw behaviours DataFrame with columns
            ``impression_id``, ``user_id``, ``time``, ``history``, ``impressions``.
        stage: One of ``"train"``, ``"val"``, ``"test"``.
        processed_news: Dict containing ``news_ids_original_strings``, ``tokens``,
            ``abstract_tokens``, ``category_indices``, ``subcategory_indices``.
        parse_news_id: Converts string news ID to integer.
        parse_user_id: Converts string user ID to integer.
        sampler: An ``ImpressionSampler`` instance with a
            ``sample_candidates_news`` method.
        max_history_length: Maximum number of history items to keep.
        max_title_length: Token length for titles.
        max_abstract_length: Token length for abstracts.
        random_train_samples: Whether to use random sampling for train candidates.
        float_dtype: NumPy float dtype string for label arrays.
        console: Rich Console for progress display.

    Returns:
        Dictionary of arrays keyed by ``histories_news_ids``,
        ``history_news_tokens``, ``candidate_news_tokens``, ``labels``, etc.
    """
    if console is None:
        console = Console()

    # Build fast lookup dicts from processed news
    news_ids_str = processed_news["news_ids_original_strings"]
    news_int_ids = [parse_news_id(nid) for nid in news_ids_str]
    news_tokens: dict[int, np.ndarray] = dict(
        zip(news_int_ids, processed_news["tokens"])
    )
    news_abstract_tokens: dict[int, np.ndarray] = dict(
        zip(news_int_ids, processed_news["abstract_tokens"])
    )
    news_categories: dict[int, int] = dict(
        zip(news_int_ids, processed_news["category_indices"])
    )
    news_subcategories: dict[int, int] = dict(
        zip(news_int_ids, processed_news["subcategory_indices"])
    )

    # Entity indices (optional, for PP-Rec)
    has_entities = "entity_indices" in processed_news
    news_entity_indices: dict[int, np.ndarray] = {}
    if has_entities:
        news_entity_indices = dict(zip(news_int_ids, processed_news["entity_indices"]))

    # CTR data (optional, for PP-Rec)
    has_ctr = "news_ctr" in processed_news
    news_ctr_values: dict[int, float] = {}
    if has_ctr:
        news_ctr_values = dict(zip(news_int_ids, processed_news["news_ctr"]))

    # Time-aware CTR (bucketed by recency since publish time, for PP-Rec candidates)
    has_time_ctr = (
        "news_ctr_bucketed" in processed_news and "news_publish_time" in processed_news
    )
    news_ctr_bucketed = processed_news.get("news_ctr_bucketed")
    news_publish_time = processed_news.get("news_publish_time")
    popularity_method = processed_news.get("popularity_ctr_method", "age_bucketed")
    popularity_dataset_start = processed_news.get("popularity_dataset_start")
    bucket_hours = int(processed_news.get("popularity_bucket_hours", 2))
    max_recency = int(processed_news.get("popularity_max_buckets", 1500)) - 1

    def _compute_recency_and_ctr(news_int_idx: int, impression_time):
        """Return (recency_bucket, ctr_val) for one news article.

        - ``recency_bucket`` is ALWAYS computed as age in ``bucket_hours``
          units (used by the model's recency embedding).
        - ``ctr_val`` is computed via the configured popularity method:
            * ``age_bucketed``: bucketed_ctr[news, age_bucket]
            * ``wall_clock``  : bucketed_ctr[news, wall_bucket - 1] (causal)
            * ``aggregate``   : aggregate news_ctr[news]
        """
        if not has_time_ctr or news_publish_time is None:
            return 0, news_ctr_values.get(news_int_idx, 0.0)
        news_row = news_int_to_row.get(news_int_idx)
        if news_row is None:
            return 0, 0.0
        pub = news_publish_time[news_row]
        if pd.isna(pub) or pd.isna(impression_time):
            return 0, news_ctr_values.get(news_int_idx, 0.0)

        # Recency = age (always)
        age_delta = pd.Timestamp(impression_time) - pd.Timestamp(pub)
        age_hours = age_delta.total_seconds() / 3600.0
        recency_bucket = max(0, min(int(age_hours / bucket_hours), max_recency))

        # CTR lookup depends on the popularity method
        if popularity_method == "wall_clock" and popularity_dataset_start is not None:
            wall_delta = pd.Timestamp(impression_time) - pd.Timestamp(
                popularity_dataset_start
            )
            wall_hours = wall_delta.total_seconds() / 3600.0
            wall_bucket = int(wall_hours / bucket_hours)
            # Causal: use the previous bucket (strictly before current time)
            ctr_bucket = max(0, min(wall_bucket - 1, max_recency))
            ctr_val = float(news_ctr_bucketed[news_row, ctr_bucket])
        elif popularity_method == "aggregate":
            ctr_val = float(news_ctr_values.get(news_int_idx, 0.0))
        else:  # age_bucketed (default)
            ctr_val = float(news_ctr_bucketed[news_row, recency_bucket])

        return recency_bucket, ctr_val

    # Map news_int_id -> row index in processed_news arrays
    news_int_to_row: dict[int, int] = {nid: i for i, nid in enumerate(news_int_ids)}

    # Parse impression timestamps once if we have time-aware CTR
    if has_time_ctr and "time" in behaviors_df.columns:
        behaviors_df = behaviors_df.copy()
        behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])

    # Accumulator lists
    histories_news_ids: list[list] = []
    history_news_tokens: list[list] = []
    history_news_abstract_tokens: list[list] = []
    history_news_categories: list[list] = []
    history_news_subcategories: list[list] = []
    history_news_entities: list[list] = []
    history_news_ctr: list[list] = []
    candidate_news_ids: list[list] = []
    candidate_news_tokens: list[list] = []
    candidate_news_abstract_tokens: list[list] = []
    candidate_news_categories: list[list] = []
    candidate_news_subcategories: list[list] = []
    candidate_news_entities: list[list] = []
    candidate_news_ctr: list[list] = []
    candidate_news_recency: list[list] = []
    labels: list[list] = []
    impression_ids: list[int] = []
    user_ids: list[str] = []

    total_original_rows = len(behaviors_df)
    total_positives = 0
    rows_with_multiple_positives = 0
    max_positives_in_row = 0

    # ------------------------------------------------------------------
    # Filter behaviours without at least one pos and one neg sample
    # ------------------------------------------------------------------
    initial_count = len(behaviors_df)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        filter_task = progress.add_task(
            "Filtering behaviors without at least one negative sample...",
            total=len(behaviors_df),
        )

        def progress_callback():
            progress.advance(filter_task)

        behaviors_mask = behaviors_df["impressions"].apply(
            lambda impressions_item: has_at_least_one_pos_neg_samples(
                impressions_item,
                parse_news_id=parse_news_id,
                available_news_ids=set(news_tokens.keys()),
                progress_callback=progress_callback,
            )
        )

    behaviors_df = behaviors_df[behaviors_mask].copy()

    n_removed = initial_count - len(behaviors_df)
    logger.info(
        f"Removed {n_removed} training behaviors without both positive and negative samples"
    )
    logger.info(f"Remaining training behaviors: {len(behaviors_df)}")

    if n_removed > 0:
        removal_percentage = (n_removed / initial_count) * 100
        logger.info(f"Filtered out {removal_percentage:.1f}% of training behaviors")

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {stage} behaviors...", total=len(behaviors_df)
        )

        for _, row in behaviors_df.iterrows():
            impressions = str(row["impressions"]).split()
            user_id = parse_user_id(str(row["user_id"]))

            positives_count = sum(imp.split("-")[1] == "1" for imp in impressions)
            total_positives += positives_count
            if positives_count > 1:
                rows_with_multiple_positives += 1
            max_positives_in_row = max(max_positives_in_row, positives_count)

            history = str(row["history"]).split() if pd.notna(row["history"]) else []
            history = history[-max_history_length:]

            # Parse history news IDs and filter missing
            history_nid_list: list[int] = []
            for h in history:
                h_idx = parse_news_id(h)
                if h_idx in news_tokens:
                    history_nid_list.append(h_idx)

            # Impression timestamp — used for both candidate and history CTR
            # (history CTR uses impression_time as a proxy for the unknown click time)
            impression_time = row["time"] if has_time_ctr and "time" in row else None

            curr_history_tokens = [news_tokens[h_idx] for h_idx in history_nid_list]
            curr_history_abstract_tokens = [
                news_abstract_tokens[h_idx] for h_idx in history_nid_list
            ]
            curr_history_categories = [
                news_categories[h_idx] for h_idx in history_nid_list
            ]
            curr_history_subcategories = [
                news_subcategories[h_idx] for h_idx in history_nid_list
            ]
            curr_history_entities = (
                [news_entity_indices[h_idx].tolist() for h_idx in history_nid_list]
                if has_entities
                else []
            )
            if has_ctr:
                # Time-aware CTR lookup for history items: use impression_time
                # as a proxy for the (unknown) click time, same recency formula
                # as candidates. Falls back to aggregate CTR if time-bucketed
                # data isn't available.
                if has_time_ctr and impression_time is not None:
                    curr_history_ctr = [
                        _compute_recency_and_ctr(h_idx, impression_time)[1]
                        for h_idx in history_nid_list
                    ]
                else:
                    curr_history_ctr = [
                        news_ctr_values.get(h_idx, 0.0) for h_idx in history_nid_list
                    ]
            else:
                curr_history_ctr = []

            # Pad history to max_history_length
            history_pad_length = max_history_length - len(history_nid_list)
            history_nid_list = [0] * history_pad_length + history_nid_list
            curr_history_tokens = [
                [0] * max_title_length
            ] * history_pad_length + curr_history_tokens
            curr_history_abstract_tokens = [
                [0] * max_abstract_length
            ] * history_pad_length + curr_history_abstract_tokens
            curr_history_categories = [0] * history_pad_length + curr_history_categories
            curr_history_subcategories = [
                0
            ] * history_pad_length + curr_history_subcategories
            if has_entities:
                max_ent = len(curr_history_entities[0]) if curr_history_entities else 5
                curr_history_entities = [
                    [0] * max_ent
                ] * history_pad_length + curr_history_entities
            if has_ctr:
                curr_history_ctr = [0.0] * history_pad_length + curr_history_ctr

            # Sample candidate news
            cand_nid_group_list, label_group_list = sampler.sample_candidates_news(
                stage=stage,
                candidates=impressions,
                random_train_samples=random_train_samples,
                parse_news_id=parse_news_id,
                available_news_ids=set(news_tokens.keys()),
            )

            if stage == "train":
                for cand_nid_group, label_group in zip(
                    cand_nid_group_list, label_group_list
                ):
                    histories_news_ids.append(history_nid_list)
                    history_news_tokens.append(curr_history_tokens)
                    history_news_abstract_tokens.append(curr_history_abstract_tokens)
                    history_news_categories.append(curr_history_categories)
                    history_news_subcategories.append(curr_history_subcategories)
                    if has_entities:
                        history_news_entities.append(curr_history_entities)
                        candidate_news_entities.append(
                            [
                                news_entity_indices[nid].tolist()
                                for nid in cand_nid_group
                            ]
                        )
                    if has_ctr:
                        history_news_ctr.append(curr_history_ctr)
                        # Time-aware CTR + recency for candidates
                        cand_ctrs, cand_recs = [], []
                        for nid in cand_nid_group:
                            rec, ctr_val = _compute_recency_and_ctr(
                                nid, impression_time
                            )
                            cand_ctrs.append(ctr_val)
                            cand_recs.append(rec)
                        candidate_news_ctr.append(cand_ctrs)
                        candidate_news_recency.append(cand_recs)
                    candidate_news_ids.append(cand_nid_group)
                    candidate_news_tokens.append(
                        [news_tokens[nid] for nid in cand_nid_group]
                    )
                    candidate_news_abstract_tokens.append(
                        [news_abstract_tokens[nid] for nid in cand_nid_group]
                    )
                    candidate_news_categories.append(
                        [news_categories[nid] for nid in cand_nid_group]
                    )
                    candidate_news_subcategories.append(
                        [news_subcategories[nid] for nid in cand_nid_group]
                    )
                    labels.append(label_group)
                    impression_ids.append(row["impression_id"])
                    user_ids.append(user_id)
            else:
                cand_nid_group, label_group = cand_nid_group_list, label_group_list

                histories_news_ids.append(history_nid_list)
                history_news_tokens.append(curr_history_tokens)
                history_news_abstract_tokens.append(curr_history_abstract_tokens)
                history_news_categories.append(curr_history_categories)
                history_news_subcategories.append(curr_history_subcategories)
                if has_entities:
                    history_news_entities.append(curr_history_entities)
                    candidate_news_entities.append(
                        [
                            news_entity_indices.get(
                                nid, np.zeros(5, dtype=np.int32)
                            ).tolist()
                            for nid in cand_nid_group
                        ]
                    )
                if has_ctr:
                    history_news_ctr.append(curr_history_ctr)
                    cand_ctrs, cand_recs = [], []
                    for nid in cand_nid_group:
                        rec, ctr_val = _compute_recency_and_ctr(nid, impression_time)
                        cand_ctrs.append(ctr_val)
                        cand_recs.append(rec)
                    candidate_news_ctr.append(cand_ctrs)
                    candidate_news_recency.append(cand_recs)
                candidate_news_ids.append(cand_nid_group)
                candidate_news_tokens.append(
                    [news_tokens[nid] for nid in cand_nid_group]
                )
                candidate_news_abstract_tokens.append(
                    [news_abstract_tokens[nid] for nid in cand_nid_group]
                )
                candidate_news_categories.append(
                    [news_categories[nid] for nid in cand_nid_group]
                )
                candidate_news_subcategories.append(
                    [news_subcategories[nid] for nid in cand_nid_group]
                )
                labels.append(label_group)
                impression_ids.append(row["impression_id"])
                user_ids.append(user_id)

            progress.advance(task)

        # Build result
        if stage == "train":
            np_float = (
                np.float32
                if float_dtype in ("float32", "mixed_float16")
                else np.float16
            )
            result: dict[str, np.ndarray | list] = {
                "histories_news_ids": np.array(histories_news_ids, dtype=np.int32),
                "history_news_tokens": np.array(history_news_tokens, dtype=np.int32),
                "history_news_abstract_tokens": np.array(
                    history_news_abstract_tokens, dtype=np.int32
                ),
                "history_news_categories": np.array(
                    history_news_categories, dtype=np.int32
                ),
                "history_news_subcategories": np.array(
                    history_news_subcategories, dtype=np.int32
                ),
                "candidate_news_ids": np.array(candidate_news_ids, dtype=np.int32),
                "candidate_news_tokens": np.array(
                    candidate_news_tokens, dtype=np.int32
                ),
                "candidate_news_abstract_tokens": np.array(
                    candidate_news_abstract_tokens, dtype=np.int32
                ),
                "candidate_news_categories": np.array(
                    candidate_news_categories, dtype=np.int32
                ),
                "candidate_news_subcategories": np.array(
                    candidate_news_subcategories, dtype=np.int32
                ),
                "labels": np.array(labels, dtype=np_float),
                "impression_ids": np.array(impression_ids, dtype=np.int32),
                "user_ids": np.array(user_ids, dtype=np.int32),
            }
            if has_entities and history_news_entities:
                result["history_news_entities"] = np.array(
                    history_news_entities, dtype=np.int32
                )
                result["candidate_news_entities"] = np.array(
                    candidate_news_entities, dtype=np.int32
                )
            if has_ctr and history_news_ctr:
                result["history_news_ctr"] = np.array(
                    history_news_ctr, dtype=np.float32
                )
                result["candidate_news_ctr"] = np.array(
                    candidate_news_ctr, dtype=np.float32
                )
                if candidate_news_recency:
                    result["candidate_news_recency"] = np.array(
                        candidate_news_recency, dtype=np.int32
                    )
        else:
            result = {
                "histories_news_ids": histories_news_ids,
                "history_news_tokens": history_news_tokens,
                "history_news_abstract_tokens": history_news_abstract_tokens,
                "history_news_categories": history_news_categories,
                "history_news_subcategories": history_news_subcategories,
                "candidate_news_ids": candidate_news_ids,
                "candidate_news_tokens": candidate_news_tokens,
                "candidate_news_abstract_tokens": candidate_news_abstract_tokens,
                "candidate_news_categories": candidate_news_categories,
                "candidate_news_subcategories": candidate_news_subcategories,
                "labels": labels,
                "impression_ids": impression_ids,
                "user_ids": user_ids,
            }
            if has_entities and history_news_entities:
                result["history_news_entities"] = history_news_entities
                result["candidate_news_entities"] = candidate_news_entities
            if has_ctr and history_news_ctr:
                result["history_news_ctr"] = history_news_ctr
                result["candidate_news_ctr"] = candidate_news_ctr
                if candidate_news_recency:
                    result["candidate_news_recency"] = candidate_news_recency

        total_processed_rows = len(histories_news_ids)
        expansion_factor = (
            total_processed_rows / total_original_rows
            if total_original_rows > 0
            else 0.0
        )

        logger.info(f"\nBehavior Processing Statistics ({stage}):")
        logger.info(f"Original number of rows: {total_original_rows:,}")
        logger.info(f"Processed number of rows: {total_processed_rows:,}")
        logger.info(f"Expansion factor: {expansion_factor:.2f}x")
        logger.info(f"Total positive samples: {total_positives:,}")
        logger.info(
            f"Rows with multiple positives: {rows_with_multiple_positives:,} "
            f"({rows_with_multiple_positives / total_original_rows * 100:.1f}%)"
            if total_original_rows > 0
            else "Rows with multiple positives: 0"
        )
        logger.info(f"Maximum positives in a single row: {max_positives_in_row}")
        logger.info(
            f"Average positives per row: {total_positives / total_original_rows:.2f}"
            if total_original_rows > 0
            else "Average positives per row: 0"
        )
        logger.info(f"Total users: {len(set(user_ids))}")

        return result


# ---------------------------------------------------------------------------
# Train / Val / Test data loading
# ---------------------------------------------------------------------------


def get_train_val_data(
    dataset_path: Path,
    validation_split_strategy: str,
    validation_split_percentage: float,
    validation_split_seed: int,
    processed_news: dict[str, np.ndarray],
    parse_news_id: Callable[[str], int],
    parse_user_id: Callable[[str], int],
    sampler,
    max_history_length: int,
    max_title_length: int,
    max_abstract_length: int,
    random_train_samples: bool = False,
    float_dtype: str = "float32",
    sampled_user_set: set[str] | None = None,
    console: Console | None = None,
) -> tuple[dict[str, np.ndarray | list], dict[str, np.ndarray | list]]:
    """Load and process training data, splitting into train and validation sets.

    Args:
        dataset_path: Root dataset path (expects ``train/behaviors.tsv``).
        validation_split_strategy: ``"random"`` or ``"chronological"``.
        validation_split_percentage: Fraction of data for validation (random only).
        validation_split_seed: Random seed for reproducible splits.
        processed_news: Processed news dict (from :func:`process_news`).
        parse_news_id: Converts string news ID to integer.
        parse_user_id: Converts string user ID to integer.
        sampler: ``ImpressionSampler`` instance.
        max_history_length: Maximum history items.
        max_title_length: Title token length.
        max_abstract_length: Abstract token length.
        random_train_samples: Random sampling flag for training candidates.
        float_dtype: NumPy float dtype for labels.
        sampled_user_set: Optional subset of user IDs to keep.
        console: Rich Console for progress display.

    Returns:
        Tuple of (train_behaviors_data, val_behaviors_data) dicts.
    """
    train_behaviors_path = dataset_path / "train" / "behaviors.tsv"

    if not train_behaviors_path.exists():
        raise FileNotFoundError(
            f"Training behaviors file not found at {train_behaviors_path}. "
            f"If using auto_split_behaviors=True, ensure the auto-split has been performed first."
        )

    behaviors_df = pd.read_csv(
        train_behaviors_path,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )

    if sampled_user_set is not None:
        behaviors_df = behaviors_df[
            behaviors_df["user_id"].isin(list(sampled_user_set))
        ]

    if validation_split_strategy == "random":
        logger.info(
            f"Using random split for validation: {validation_split_percentage * 100}% "
            f"of training behaviors data, seed: {validation_split_seed}"
        )
        shuffled_df = behaviors_df.sample(
            frac=1, random_state=validation_split_seed
        ).reset_index(drop=True)

        val_size = int(len(shuffled_df) * validation_split_percentage)
        val_behaviors = shuffled_df.iloc[:val_size]
        train_behaviors = shuffled_df.iloc[val_size:]
        logger.info(
            f"Random split: Train size: {len(train_behaviors)}, "
            f"Validation size: {len(val_behaviors)}"
        )

    elif validation_split_strategy == "chronological":
        logger.info(
            "Using chronological split for validation "
            "(last day of training behaviors data)."
        )
        behaviors_df["time"] = pd.to_datetime(behaviors_df["time"])
        last_day = behaviors_df["time"].max().date()
        train_behaviors = behaviors_df[behaviors_df["time"].dt.date < last_day]
        val_behaviors = behaviors_df[behaviors_df["time"].dt.date == last_day]
    else:
        raise ValueError(
            f"Unknown validation_split_strategy: {validation_split_strategy}"
        )

    _common_kwargs = dict(
        processed_news=processed_news,
        parse_news_id=parse_news_id,
        parse_user_id=parse_user_id,
        sampler=sampler,
        max_history_length=max_history_length,
        max_title_length=max_title_length,
        max_abstract_length=max_abstract_length,
        random_train_samples=random_train_samples,
        float_dtype=float_dtype,
        console=console,
    )

    logger.info(f"Train behaviors: {len(train_behaviors):,}")
    train_behaviors_data = process_behaviors(
        train_behaviors, stage="train", **_common_kwargs
    )

    logger.info(f"Validation behaviors: {len(val_behaviors):,}")
    val_behaviors_data = process_behaviors(val_behaviors, stage="val", **_common_kwargs)

    return train_behaviors_data, val_behaviors_data


def get_test_data(
    dataset_path: Path,
    processed_news: dict[str, np.ndarray],
    parse_news_id: Callable[[str], int],
    parse_user_id: Callable[[str], int],
    sampler,
    max_history_length: int,
    max_title_length: int,
    max_abstract_length: int,
    random_train_samples: bool = False,
    float_dtype: str = "float32",
    sampled_user_set: set[str] | None = None,
    console: Console | None = None,
) -> dict[str, np.ndarray | list]:
    """Load and process test data from ``valid/behaviors.tsv``.

    Args:
        dataset_path: Root dataset path (expects ``valid/behaviors.tsv``).
        processed_news: Processed news dict.
        parse_news_id: Converts string news ID to integer.
        parse_user_id: Converts string user ID to integer.
        sampler: ``ImpressionSampler`` instance.
        max_history_length: Maximum history items.
        max_title_length: Title token length.
        max_abstract_length: Abstract token length.
        random_train_samples: Random sampling flag.
        float_dtype: NumPy float dtype for labels.
        sampled_user_set: Optional subset of user IDs to keep.
        console: Rich Console for progress display.

    Returns:
        Dictionary of processed test behaviour arrays.
    """
    test_behaviors_path = dataset_path / "valid" / "behaviors.tsv"

    if not test_behaviors_path.exists():
        raise FileNotFoundError(
            f"Test behaviors file not found at {test_behaviors_path}. "
            f"If using auto_split_behaviors=True, ensure the auto-split has been performed first."
        )

    test_behaviors = pd.read_csv(
        test_behaviors_path,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )

    if sampled_user_set is not None:
        test_behaviors = test_behaviors[
            test_behaviors["user_id"].isin(list(sampled_user_set))
        ]

    logger.info(f"Test behaviors: {len(test_behaviors):,}")

    return process_behaviors(
        test_behaviors,
        stage="test",
        processed_news=processed_news,
        parse_news_id=parse_news_id,
        parse_user_id=parse_user_id,
        sampler=sampler,
        max_history_length=max_history_length,
        max_title_length=max_title_length,
        max_abstract_length=max_abstract_length,
        random_train_samples=random_train_samples,
        float_dtype=float_dtype,
        console=console,
    )
