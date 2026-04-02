"""
Japanese Dataset class that inherits from NewsDatasetBase.
This class provides Japanese-specific configurations and text processing.
"""
import re
from typing import Dict, List, Optional
from omegaconf import DictConfig

from src.core.data.datasets.dataset import NewsDatasetBase


class JapaneseDataset(NewsDatasetBase):
    """Japanese news dataset implementation with Japanese-specific text processing."""

    def __init__(
            self,
            name: str,
            version: str,
            data_path: Optional[str] = None,
            urls: Optional[Dict] = None,
            max_title_length: int = 30,
            max_abstract_length: int = 50,
            max_history_length: int = 50,
            max_impressions_length: int = 5,
            seed: int = 42,
            embedding_type: str = "random",  # Default to random for Japanese
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
            auto_split_behaviors: bool = True,
            auto_convert_format: bool = True,  # Auto-convert custom format to MIND format
            word_threshold: int = 3,
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
            process_user_id: bool = False,
            max_entities: int = 1000,
            max_relations: int = 500,
            download_if_missing: bool = True,
            id_prefix: str = "",  # Japanese datasets might not use prefixes
            user_id_prefix: str = "",  # Japanese datasets might not use prefixes
            **kwargs
    ):

        super().__init__(
            name=name,
            version=version,
            data_path=data_path,
            urls=urls,
            language="japanese",
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
            auto_split_behaviors=auto_split_behaviors,
            auto_convert_format=auto_convert_format,
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
        )

    def _segment_text_into_words(self, sent: str) -> List[str]:
        """
        Segment Japanese text into words.
        """
        if not isinstance(sent, str) or not sent.strip():
            return []

        sent = sent.lower()

        return self._simple_japanese_tokenize(sent)

    def _simple_japanese_tokenize(self, sent: str) -> List[str]:
        """
        Simple Japanese tokenization using character-based approach.

        This method:
        1. Splits on whitespace and common punctuation
        2. Treats consecutive hiragana, katakana, or kanji as separate tokens
        3. Keeps ASCII words together
        """
        # First split on obvious delimiters
        tokens = []

        # Pattern to match different character types
        # \u3040-\u309F: Hiragana
        # \u30A0-\u30FF: Katakana
        # \u4E00-\u9FAF: Kanji
        # \u3000-\u303F: CJK symbols and punctuation
        pattern = r'[\u3040-\u309F]+|[\u30A0-\u30FF]+|[\u4E00-\u9FAF]+|[a-zA-Z0-9]+|[.,!?;|。、！？]'

        matches = re.findall(pattern, sent)

        for match in matches:
            # Further split long sequences if needed
            if len(match) > 10 and re.match(r'[\u4E00-\u9FAF]+', match):
                # Split very long kanji sequences into smaller chunks
                for i in range(0, len(match), 3):
                    tokens.append(match[i:i + 3])
            else:
                tokens.append(match)

        return [token for token in tokens if token.strip()]
