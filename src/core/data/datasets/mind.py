"""
Simplified MIND Dataset class that inherits from NewsDatasetBase.
This class only defines MIND-specific configurations.
"""

import logging

import pandas as pd
from omegaconf import DictConfig

from src.core.data.datasets.dataset import NewsDatasetBase
from src.core.data.processing.popularity import (
    compute_news_ctr_and_publish_times,
    load_popularity_cache,
    save_popularity_cache,
)

logger = logging.getLogger(__name__)


class MINDDataset(NewsDatasetBase):
    """MIND (Microsoft News Dataset) implementation."""

    def _compute_extra_features(self) -> None:
        """Compute MIND-specific popularity features (CTR + publish times).

        Computes aggregate CTR, per-news publish times, and time-bucketed CTR
        according to ``self.popularity_ctr_method``:

        - ``"age_bucketed"`` — bucket = age since publish (current default).
          Uses train behaviors only.
        - ``"wall_clock"`` — bucket = wall-clock time since dataset start.
          Uses train + valid behaviors (test labels are unavailable in MIND
          so we can't use them; train+val is the maximum we can include).
          Causal lookups at prediction time use the previous bucket.
        - ``"aggregate"`` — single lifetime CTR per news, broadcast across
          all buckets. Baseline.
        """
        method = self.popularity_ctr_method
        cache_dir = self.dataset_path / "processed"
        cached = load_popularity_cache(cache_dir, method=method)
        if cached is not None:
            logger.info(f"Loading cached popularity features (method={method})...")
            news_ctr, publish_time_arr, news_ctr_bucketed, dataset_start = cached
            self.processed_news["news_ctr"] = news_ctr
            self.processed_news["news_ctr_bucketed"] = news_ctr_bucketed
            self.processed_news["news_publish_time"] = publish_time_arr
            self.processed_news["popularity_ctr_method"] = method
            self.processed_news["popularity_bucket_hours"] = (
                self.popularity_bucket_hours
            )
            self.processed_news["popularity_max_buckets"] = self.popularity_max_buckets
            if dataset_start is not None:
                self.processed_news["popularity_dataset_start"] = dataset_start
            return

        train_path = self.dataset_path / "train" / "behaviors.tsv"
        valid_path = self.dataset_path / "valid" / "behaviors.tsv"
        if not train_path.exists():
            logger.warning("No training behaviors found — popularity features skipped.")
            return

        logger.info(
            f"Computing MIND popularity features "
            f"(method={method}, CTR/publish/bucketed)..."
        )
        # For wall_clock we can include validation behaviors safely (the
        # causal lookup at prediction time uses the previous bucket only).
        # For age_bucketed and aggregate we keep using train only to match
        # the previous behavior.
        col_names = ["impression_id", "user_id", "time", "history", "impressions"]
        df = pd.read_csv(train_path, sep="\t", header=None, names=col_names)
        if method == "wall_clock" and valid_path.exists():
            df_val = pd.read_csv(valid_path, sep="\t", header=None, names=col_names)
            df = pd.concat([df, df_val], ignore_index=True)
            logger.info(
                f"wall_clock: combined train + valid behaviors ({len(df):,} rows total)"
            )

        news_ids_str = self.processed_news["news_ids_original_strings"]
        news_str_to_idx = {nid: i for i, nid in enumerate(news_ids_str)}

        news_ctr, publish_time_arr, news_ctr_bucketed, dataset_start = (
            compute_news_ctr_and_publish_times(
                behaviors_df=df,
                news_str_to_idx=news_str_to_idx,
                num_news=len(news_ids_str),
                bucket_hours=self.popularity_bucket_hours,
                max_buckets=self.popularity_max_buckets,
                method=method,
                ctr_smoothing=self.popularity_ctr_smoothing,
            )
        )

        self.processed_news["news_ctr"] = news_ctr
        self.processed_news["news_ctr_bucketed"] = news_ctr_bucketed
        self.processed_news["news_publish_time"] = publish_time_arr
        self.processed_news["popularity_ctr_method"] = method
        self.processed_news["popularity_bucket_hours"] = self.popularity_bucket_hours
        self.processed_news["popularity_max_buckets"] = self.popularity_max_buckets
        if method == "wall_clock":
            self.processed_news["popularity_dataset_start"] = dataset_start

        save_popularity_cache(
            cache_dir=cache_dir,
            news_ctr=news_ctr,
            publish_time_arr=publish_time_arr,
            news_ctr_bucketed=news_ctr_bucketed,
            news_str_to_idx=news_str_to_idx,
            method=method,
            dataset_start=dataset_start,
        )

    def __init__(
        self,
        name: str,
        version: str,
        urls: dict,
        max_title_length: int,
        max_abstract_length: int,
        max_history_length: int,
        max_impressions_length: int,
        seed: int,
        embedding_type: str = "glove",
        embedding_size: int = 300,
        sampling: DictConfig | None = None,
        data_fraction_train: float = 1.0,
        data_fraction_val: float = 1.0,
        data_fraction_test: float = 1.0,
        mode: str = "train",
        use_knowledge_graph: bool = False,
        random_train_samples: bool = False,
        validation_split_strategy: str = "chronological",
        validation_split_percentage: float = 0.05,
        validation_split_seed: int | None = None,
        word_threshold: int = 3,
        process_title: bool = True,
        process_abstract: bool = True,
        process_category: bool = True,
        process_subcategory: bool = True,
        process_user_id: bool = False,
        process_entities: bool = False,
        max_entities: int = 5,
        max_relations: int = 500,
        popularity: DictConfig | dict | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            version=version,
            data_path=None,  # Use default cache path
            urls=urls[version],  # Get URLs for specific version from config
            language="english",  # MIND is in English
            max_title_length=max_title_length,
            max_abstract_length=max_abstract_length,
            max_history_length=max_history_length,
            max_impressions_length=max_impressions_length,
            seed=seed,
            embedding_type=embedding_type,
            embedding_size=embedding_size,
            sampling=sampling,
            data_fraction_train=data_fraction_train,
            data_fraction_val=data_fraction_val,
            data_fraction_test=data_fraction_test,
            mode=mode,
            use_knowledge_graph=use_knowledge_graph,
            random_train_samples=random_train_samples,
            validation_split_strategy=validation_split_strategy,
            validation_split_percentage=validation_split_percentage,
            validation_split_seed=validation_split_seed,
            word_threshold=word_threshold,
            process_title=process_title,
            process_abstract=process_abstract,
            process_category=process_category,
            process_subcategory=process_subcategory,
            process_user_id=process_user_id,
            process_entities=process_entities,
            max_entities=max_entities,
            popularity=popularity,
            max_relations=max_relations,
            download_if_missing=True,
            id_prefix="N",  # MIND uses "N" prefix for news IDs
            user_id_prefix="U",  # MIND uses "U" prefix for user IDs
        )
