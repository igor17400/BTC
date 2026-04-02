"""
Simplified MIND Dataset class that inherits from NewsDatasetBase.
This class only defines MIND-specific configurations.
"""
from typing import Dict, Optional
from omegaconf import DictConfig

from src.core.data.datasets.dataset import NewsDatasetBase


class MINDDataset(NewsDatasetBase):
    """MIND (Microsoft News Dataset) implementation."""

    def __init__(
            self,
            name: str,
            version: str,
            urls: Dict,
            max_title_length: int,
            max_abstract_length: int,
            max_history_length: int,
            max_impressions_length: int,
            seed: int,
            embedding_type: str = "glove",
            embedding_size: int = 300,
            sampling: Optional[DictConfig] = None,
            data_fraction_train: float = 1.0,
            data_fraction_val: float = 1.0,
            data_fraction_test: float = 1.0,
            mode: str = "train",
            use_knowledge_graph: bool = False,
            random_train_samples: bool = False,
            validation_split_strategy: str = "chronological",
            validation_split_percentage: float = 0.05,
            validation_split_seed: Optional[int] = None,
            word_threshold: int = 3,
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
            process_user_id: bool = False,
            max_entities: int = 1000,
            max_relations: int = 500,
            **kwargs
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
            max_entities=max_entities,
            max_relations=max_relations,
            download_if_missing=True,
            id_prefix="N",  # MIND uses "N" prefix for news IDs
            user_id_prefix="U",  # MIND uses "U" prefix for user IDs
        )
