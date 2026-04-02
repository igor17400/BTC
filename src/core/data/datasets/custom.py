"""
Simplified Custom Dataset class that inherits from NewsDatasetBase.
This class allows for flexible dataset configurations.
"""
from omegaconf import DictConfig

from src.core.data.datasets.dataset import NewsDatasetBase


class CustomDataset(NewsDatasetBase):
    """Generic custom dataset implementation for datasets following MIND format."""

    def __init__(
            self,
            name: str,
            version: str,
            language: str,
            data_path: str | None = None,
            urls: dict | None = None,
            download_if_missing: bool = True,
            max_title_length: int = 30,
            max_abstract_length: int = 50,
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            seed: int = 42,
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
            max_entities: int = 1000,
            max_relations: int = 500,
            id_prefix: str = "",  # Can be customized per dataset
            user_id_prefix: str = "",  # Can be customized per dataset
            **kwargs
    ):
        super().__init__(
            name=name,
            version=version,
            data_path=data_path,
            urls=urls,
            language=language,
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
            download_if_missing=download_if_missing,
            id_prefix=id_prefix,
            user_id_prefix=user_id_prefix,
            **kwargs  # Pass any additional parameters like auto_convert_format, auto_split_behaviors
        )
