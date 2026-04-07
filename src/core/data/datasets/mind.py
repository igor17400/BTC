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

        Reads ``train/behaviors.tsv`` and computes aggregate CTR, per-news
        publish times (proxied as the earliest impression timestamp), and
        time-bucketed CTR. Used by PP-Rec.
        """
        cache_dir = self.dataset_path / "processed"
        cached = load_popularity_cache(cache_dir)
        if cached is not None:
            logger.info("Loading cached popularity features...")
            news_ctr, publish_time_arr, news_ctr_bucketed = cached
            self.processed_news["news_ctr"] = news_ctr
            self.processed_news["news_ctr_bucketed"] = news_ctr_bucketed
            self.processed_news["news_publish_time"] = publish_time_arr
            return

        train_path = self.dataset_path / "train" / "behaviors.tsv"
        if not train_path.exists():
            logger.warning("No training behaviors found — popularity features skipped.")
            return

        logger.info(
            "Computing MIND popularity features (CTR, publish times, bucketed CTR)..."
        )
        df = pd.read_csv(
            train_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        news_ids_str = self.processed_news["news_ids_original_strings"]
        news_str_to_idx = {nid: i for i, nid in enumerate(news_ids_str)}

        news_ctr, publish_time_arr, news_ctr_bucketed = (
            compute_news_ctr_and_publish_times(
                behaviors_df=df,
                news_str_to_idx=news_str_to_idx,
                num_news=len(news_ids_str),
                bucket_hours=2,
                max_buckets=1500,
            )
        )

        self.processed_news["news_ctr"] = news_ctr
        self.processed_news["news_ctr_bucketed"] = news_ctr_bucketed
        self.processed_news["news_publish_time"] = publish_time_arr

        save_popularity_cache(
            cache_dir=cache_dir,
            news_ctr=news_ctr,
            publish_time_arr=publish_time_arr,
            news_ctr_bucketed=news_ctr_bucketed,
            news_str_to_idx=news_str_to_idx,
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
        max_entities: int = 1000,
        max_relations: int = 500,
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
            max_relations=max_relations,
            download_if_missing=True,
            id_prefix="N",  # MIND uses "N" prefix for news IDs
            user_id_prefix="U",  # MIND uses "U" prefix for user IDs
        )
