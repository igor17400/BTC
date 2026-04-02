import keras

from typing import Any, Iterator, Tuple, Dict, Optional
import numpy as np


class TrainingSequence(keras.utils.Sequence):
    """Keras Sequence for news recommendation training data.
    
    This follows Keras best practices for custom data generators:
    - Inherits from keras.utils.Sequence
    - Implements __len__ and __getitem__ for proper batching
    - Handles shuffling via on_epoch_end
    - Returns data in the format expected by the model
    """

    def __init__(
            self,
            features: Dict[str, np.ndarray],
            labels: np.ndarray,
            batch_size: int,
            model_name: str = "nrms"
    ):
        """Initialize the training sequence.
        
        Args:
            features: Dictionary of feature arrays
            labels: Label array
            batch_size: Batch size for training
            model_name: Name of the model (for input name mapping)
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.model_name = model_name.lower()
        self.num_samples = len(labels)

        # Create indices for shuffling
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Get a batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (features_dict, labels_array)
        """
        # Calculate batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch features
        batch_features = {}
        for feature_name, feature_data in self.features.items():
            batch_features[feature_name] = feature_data[batch_indices]

        # Get batch labels
        batch_labels = self.labels[batch_indices]

        return batch_features, batch_labels

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        np.random.shuffle(self.indices)


class NewsDataLoader:
    """Factory class for creating news recommendation data loaders.
    
    This class provides static methods to create different types of data loaders
    for the news recommendation system.
    """

    @staticmethod
    def create_train_dataset(
            history_news_tokens: keras.KerasTensor,
            history_news_abstract_tokens: keras.KerasTensor,
            history_news_category: keras.KerasTensor,
            history_news_subcategory: keras.KerasTensor,
            candidate_news_tokens: keras.KerasTensor,
            candidate_news_abstract_tokens: keras.KerasTensor,
            candidate_news_category: keras.KerasTensor,
            candidate_news_subcategory: keras.KerasTensor,
            user_ids: keras.KerasTensor,
            labels: keras.KerasTensor,
            batch_size: int,
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
            process_user_id: bool = False,
            model_name: str = "nrms",
    ) -> TrainingSequence:
        """Create a training dataset for Keras model.fit().
        
        This method creates a TrainingSequence that is compatible with
        Keras 3's model.fit() method.
        
        Args:
            history_news_tokens: User history news tokens
            history_news_abstract_tokens: User history abstract tokens
            history_news_category: User history categories
            history_news_subcategory: User history subcategories
            candidate_news_tokens: Candidate news tokens
            candidate_news_abstract_tokens: Candidate abstract tokens
            candidate_news_category: Candidate categories
            candidate_news_subcategory: Candidate subcategories
            user_ids: User IDs
            labels: Training labels
            batch_size: Batch size
            process_title: Whether to include title features
            process_abstract: Whether to include abstract features
            process_category: Whether to include category features
            process_subcategory: Whether to include subcategory features
            process_user_id: Whether to include user ID features
            model_name: Name of the model for input mapping
            
        Returns:
            TrainingSequence instance
        """
        # Build features dictionary based on processing flags
        features = {}

        if process_title:
            features["hist_tokens"] = keras.ops.convert_to_numpy(history_news_tokens)
            features["cand_tokens"] = keras.ops.convert_to_numpy(candidate_news_tokens)

        if process_abstract:
            features["hist_abstract_tokens"] = keras.ops.convert_to_numpy(history_news_abstract_tokens)
            features["cand_abstract_tokens"] = keras.ops.convert_to_numpy(candidate_news_abstract_tokens)

        if process_category:
            features["hist_category"] = keras.ops.convert_to_numpy(history_news_category)
            features["cand_category"] = keras.ops.convert_to_numpy(candidate_news_category)

        if process_subcategory:
            features["hist_subcategory"] = keras.ops.convert_to_numpy(history_news_subcategory)
            features["cand_subcategory"] = keras.ops.convert_to_numpy(candidate_news_subcategory)

        if process_user_id:
            features["user_ids"] = keras.ops.convert_to_numpy(user_ids)

        # Convert labels to numpy
        labels_array = keras.ops.convert_to_numpy(labels)

        return TrainingSequence(
            features=features,
            labels=labels_array,
            batch_size=batch_size,
            model_name=model_name
        )


class ImpressionIterator:
    """Iterator for processing impressions during validation/testing.
    
    This iterator processes impressions one at a time, which is necessary
    for evaluation where each impression may have a different number of candidates.
    """

    def __init__(
            self,
            impression_tokens: Any,  # Can be List[List[List[int]]] or np.ndarray
            impression_abstract_tokens: Any,  # Can be List[List[int]] or np.ndarray  
            impression_category: Any,  # Can be List[List[int]] or np.ndarray
            impression_subcategory: Any,  # Can be List[List[int]] or np.ndarray
            labels: Any,  # Can be List[List[float]] or np.ndarray
            impression_ids: Any,  # Can be List[int] or np.ndarray
            candidate_ids: Any,  # Can be List[int] or np.ndarray
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
    ):
        """Initialize the impression iterator."""
        self.impression_tokens = impression_tokens
        self.impression_abstract_tokens = impression_abstract_tokens
        self.impression_category = impression_category
        self.impression_subcategory = impression_subcategory
        self.labels = labels
        self.impression_ids = impression_ids
        self.candidate_ids = candidate_ids
        self.num_impressions = len(labels)

        # Processing flags
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

        # Get float dtype from global policy
        policy = keras.mixed_precision.global_policy()
        self.float_dtype = "float16" if "float16" in str(policy.compute_dtype) else "float32"

    def __iter__(self) -> Iterator[Tuple[keras.KerasTensor, keras.KerasTensor, int, Any]]:
        """Iterate through impressions."""
        for idx in range(self.num_impressions):
            # Build features based on processing flags
            features = []

            if self.process_title:
                features.append(keras.ops.convert_to_tensor(self.impression_tokens[idx], dtype="int32"))
            if self.process_abstract:
                features.append(keras.ops.convert_to_tensor(self.impression_abstract_tokens[idx], dtype="int32"))
            if self.process_category:
                category = keras.ops.convert_to_tensor(self.impression_category[idx], dtype="int32")
                if len(keras.ops.shape(category)) == 1:
                    category = keras.ops.expand_dims(category, axis=1)
                features.append(category)
            if self.process_subcategory:
                subcategory = keras.ops.convert_to_tensor(self.impression_subcategory[idx], dtype="int32")
                if len(keras.ops.shape(subcategory)) == 1:
                    subcategory = keras.ops.expand_dims(subcategory, axis=1)
                features.append(subcategory)

            # Concatenate features
            if len(features) > 1:
                features = keras.ops.concatenate(features, axis=1)
            else:
                features = features[0]

            # Convert labels
            labels = keras.ops.convert_to_tensor(self.labels[idx], dtype=self.float_dtype)

            # Get impression ID and candidate IDs
            impression_id = self.impression_ids[idx]
            candidate_ids = self.candidate_ids[idx] if idx < len(self.candidate_ids) else []

            yield features, labels, impression_id, candidate_ids

    def __len__(self) -> int:
        """Return total number of impressions."""
        return self.num_impressions


class NewsBatchDataloader:
    """Dataloader for processing news articles in batches.
    
    This is used during validation/testing to precompute news embeddings.
    """

    def __init__(
            self,
            news_ids: np.ndarray,
            news_tokens: keras.KerasTensor,
            news_abstract_tokens: keras.KerasTensor,
            news_category_indices: keras.KerasTensor,
            news_subcategory_indices: keras.KerasTensor,
            batch_size: int = 1024,
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
    ):
        """Initialize the news batch dataloader."""
        self.news_ids = news_ids  # Keep as numpy for string IDs
        self.news_tokens = news_tokens
        self.news_abstract_tokens = news_abstract_tokens
        self.news_category_indices = news_category_indices
        self.news_subcategory_indices = news_subcategory_indices
        self.batch_size = batch_size
        self.num_news = len(news_ids)

        # Processing flags
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through news batches."""
        for i in range(0, self.num_news, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_news)

            # Get batch data
            batch_ids = self.news_ids[i:end_idx]
            batch_features = []

            if self.process_title:
                batch_features.append(self.news_tokens[i:end_idx])
            if self.process_abstract:
                batch_features.append(self.news_abstract_tokens[i:end_idx])
            if self.process_category:
                category = self.news_category_indices[i:end_idx]
                if len(keras.ops.shape(category)) == 1:
                    category = keras.ops.expand_dims(category, axis=1)
                batch_features.append(category)
            if self.process_subcategory:
                subcategory = self.news_subcategory_indices[i:end_idx]
                if len(keras.ops.shape(subcategory)) == 1:
                    subcategory = keras.ops.expand_dims(subcategory, axis=1)
                batch_features.append(subcategory)

            # Concatenate features
            batch_features = keras.ops.concatenate(batch_features, axis=1)

            yield {"news_id": batch_ids, "news_features": batch_features}

    def __len__(self) -> int:
        """Return total number of news articles."""
        return self.num_news


class UserHistoryBatchDataloader:
    """Dataloader for processing user histories in batches.
    
    This is used during validation/testing to precompute user embeddings.
    """

    def __init__(
            self,
            history_tokens: Any,  # Can be List[List[int]] or np.ndarray
            history_abstract_tokens: Any,  # Can be List[List[int]] or np.ndarray
            history_category: Any,  # Can be List[List[int]] or np.ndarray
            history_subcategory: Any,  # Can be List[List[int]] or np.ndarray
            impression_ids: Any,  # Can be List[int] or np.ndarray
            user_ids: Any = None,  # Can be List[int] or np.ndarray
            batch_size: int = 32,
            process_title: bool = True,
            process_abstract: bool = True,
            process_category: bool = True,
            process_subcategory: bool = True,
    ):
        """Initialize the user history dataloader."""
        self.history_tokens = history_tokens
        self.history_abstract_tokens = history_abstract_tokens
        self.history_category = history_category
        self.history_subcategory = history_subcategory
        self.impression_ids = impression_ids
        self.user_ids = user_ids
        self.batch_size = batch_size
        self.num_users = len(impression_ids)

        # Processing flags
        self.process_title = process_title
        self.process_abstract = process_abstract
        self.process_category = process_category
        self.process_subcategory = process_subcategory

    def __iter__(self) -> Iterator[Tuple[Any, Optional[keras.KerasTensor], keras.KerasTensor]]:
        """Iterate through user history batches."""
        for i in range(0, self.num_users, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_users)

            # Get batch impression IDs
            batch_impression_ids = self.impression_ids[i:end_idx]

            # Get batch user IDs if available
            batch_user_ids = None
            if self.user_ids is not None:
                batch_user_ids = keras.ops.convert_to_tensor(self.user_ids[i:end_idx])

            # Build features
            batch_features = []

            if self.process_title:
                batch_features.append(keras.ops.convert_to_tensor(self.history_tokens[i:end_idx]))
            if self.process_abstract:
                batch_features.append(keras.ops.convert_to_tensor(self.history_abstract_tokens[i:end_idx]))
            if self.process_category:
                category = keras.ops.convert_to_tensor(self.history_category[i:end_idx])
                if len(keras.ops.shape(category)) == 2:  # If it's 2D, add dimension
                    category = keras.ops.expand_dims(category, axis=2)
                batch_features.append(category)
            if self.process_subcategory:
                subcategory = keras.ops.convert_to_tensor(self.history_subcategory[i:end_idx])
                if len(keras.ops.shape(subcategory)) == 2:  # If it's 2D, add dimension
                    subcategory = keras.ops.expand_dims(subcategory, axis=2)
                batch_features.append(subcategory)

            # Concatenate features along the last dimension
            if len(batch_features) > 1:
                batch_features = keras.ops.concatenate(batch_features, axis=-1)
            else:
                batch_features = batch_features[0]

            yield batch_impression_ids, batch_user_ids, batch_features

    def __len__(self) -> int:
        """Return total number of users."""
        return self.num_users
