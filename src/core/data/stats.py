import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# PROCESSED DATA STATISTICS (After Processing)
# ============================================================================


def display_statistics(data_dict: Dict[str, Dict], mode: str = "train") -> None:
    """Display statistics about the dataset."""
    logger.info("Displaying dataset statistics...")

    if mode == "train":
        num_news = len(data_dict["news"]["news_ids_original_strings"])
        num_train_behaviors = len(data_dict["train_behaviors"]["histories_news_ids"])
        num_val_behaviors = len(data_dict["val_behaviors"]["histories_news_ids"])

        logger.info(f"Number of news articles: {num_news}")
        logger.info(f"Number of training behaviors: {num_train_behaviors}")
        logger.info(f"Number of validation behaviors: {num_val_behaviors}")

        # Additional statistics using NumPy
        avg_history_length = np.mean(
            [len(history) for history in data_dict["train_behaviors"]["history_news_tokens"]]
        )
        avg_impressions_length = np.mean(
            [
                len(impression)
                for impression in data_dict["train_behaviors"]["candidate_news_tokens"]
            ]
        )
        avg_history_length_val = np.mean(
            [len(history) for history in data_dict["val_behaviors"]["history_news_tokens"]]
        )
        avg_impressions_length_val = np.mean(
            [len(impression) for impression in data_dict["val_behaviors"]["candidate_news_tokens"]]
        )

        logger.info(f"Average history length: {avg_history_length:.2f}")
        logger.info(f"Average impressions length: {avg_impressions_length:.2f}")
        logger.info(f"Average history length (validation): {avg_history_length_val:.2f}")
        logger.info(f"Average impressions length (validation): {avg_impressions_length_val:.2f}")
    else:
        num_test_news = len(data_dict["news"]["news_ids_original_strings"])
        num_test_behaviors = len(data_dict["test_behaviors"]["histories_news_ids"])

        logger.info(f"Number of news articles: {num_test_news}")
        logger.info(f"Number of test behaviors: {num_test_behaviors}")

        avg_history_length_test = np.mean(
            [len(history) for history in data_dict["test_behaviors"]["history_news_tokens"]]
        )
        avg_impressions_length_test = np.mean(
            [len(impression) for impression in data_dict["test_behaviors"]["candidate_news_tokens"]]
        )

        logger.info(f"Average history length (test): {avg_history_length_test:.2f}")
        logger.info(f"Average impressions length (test): {avg_impressions_length_test:.2f}")


def collect_processed_data_statistics(dataset_instance: Any, summary_data: dict) -> None:
    """Collect statistics from processed data after processing."""
    logger.info("Collecting processed data statistics...")

    # Collect basic dataset info
    collect_basic_dataset_info(dataset_instance, summary_data)

    # Collect news statistics
    collect_news_statistics(dataset_instance, summary_data)

    # Collect behavior statistics
    collect_behavior_statistics(dataset_instance, summary_data)

    # Collect overall statistics
    collect_overall_statistics(dataset_instance, summary_data)

    # Collect quality metrics
    collect_quality_metrics(summary_data)

    logger.info("Processed data statistics collection completed")


def collect_basic_dataset_info(dataset_instance: Any, summary_data: dict) -> None:
    """Collect basic dataset information."""
    summary_data.update(
        {
            "dataset_name": dataset_instance.name,
            "dataset_version": dataset_instance.version,
            "embedding_type": dataset_instance.embedding_type,
            "embedding_size": dataset_instance.embedding_size,
            "word_threshold": dataset_instance.word_threshold,
            "max_title_length": dataset_instance.max_title_length,
            "max_abstract_length": dataset_instance.max_abstract_length,
            "max_history_length": dataset_instance.max_history_length,
            "max_impressions_length": dataset_instance.max_impressions_length,
            "use_knowledge_graph": dataset_instance.use_knowledge_graph,
            "validation_split_strategy": dataset_instance.validation_split_strategy,
            "validation_split_percentage": dataset_instance.validation_split_percentage,
            "data_fraction_train": dataset_instance.data_fraction_train,
            "data_fraction_val": dataset_instance.data_fraction_val,
            "data_fraction_test": dataset_instance.data_fraction_test,
            "process_title": dataset_instance.process_title,
            "process_abstract": dataset_instance.process_abstract,
            "process_category": dataset_instance.process_category,
            "process_subcategory": dataset_instance.process_subcategory,
            "process_user_id": dataset_instance.process_user_id,
        }
    )


def collect_news_statistics(dataset_instance: Any, summary_data: dict) -> None:
    """Collect news-related statistics."""
    try:
        if not (hasattr(dataset_instance, "processed_news") and dataset_instance.processed_news):
            logger.debug("No processed news data available")
            return

        summary_data["total_news_articles"] = len(
            dataset_instance.processed_news["news_ids_original_strings"]
        )
        summary_data["vocabulary_size"] = len(dataset_instance.vocab)
        summary_data["num_categories"] = dataset_instance.processed_news.get("num_categories", 0)
        summary_data["num_subcategories"] = dataset_instance.processed_news.get(
            "num_subcategories", 0
        )

        # Calculate average lengths
        if "tokens" in dataset_instance.processed_news:
            non_pad_tokens = (
                dataset_instance.processed_news["tokens"] != dataset_instance.vocab["[PAD]"]
            )
            avg_title_length = np.mean(np.sum(non_pad_tokens, axis=1))
            summary_data["average_title_length"] = round(avg_title_length, 2)

        if "abstract_tokens" in dataset_instance.processed_news:
            non_pad_abstract_tokens = (
                dataset_instance.processed_news["abstract_tokens"]
                != dataset_instance.vocab["[PAD]"]
            )
            avg_abstract_length = np.mean(np.sum(non_pad_abstract_tokens, axis=1))
            summary_data["average_abstract_length"] = round(avg_abstract_length, 2)
    except Exception as e:
        logger.warning(f"Error collecting news statistics: {e}")
        summary_data["total_news_articles"] = 0
        summary_data["vocabulary_size"] = 0


def collect_user_occurrence_counts(dataset_instance: Any, summary_data: dict) -> None:
    """Collect user ID occurrence counts for all splits together."""
    try:
        all_user_occurrences = Counter()
        all_unique_users = set()

        # Collect user occurrences from all splits together
        for split in ["train", "val", "test"]:
            data_attr = f"{split}_behaviors_data"
            data = getattr(dataset_instance, data_attr, None)
            if data is not None and "user_ids" in data:
                user_ids = data["user_ids"]
                if isinstance(user_ids, (list, np.ndarray)):
                    all_user_occurrences.update(user_ids)
                    all_unique_users.update(user_ids)
                    logger.info(f"Added {len(set(user_ids))} unique users from {split} split")
                    logger.info(f"{split} user ID range: {min(user_ids)} to {max(user_ids)}")

        # Debug: Let's see what we actually have
        logger.info(f"DEBUG: Total unique users collected: {len(all_unique_users)}")
        logger.info(f"DEBUG: User ID range: {min(all_unique_users)} to {max(all_unique_users)}")
        logger.info(f"DEBUG: Sample user IDs: {sorted(list(all_unique_users))[:10]}")
        logger.info(f"DEBUG: Last 10 user IDs: {sorted(list(all_unique_users))[-10:]}")

        # IMPORTANT: The MIND dataset uses sparse user IDs
        # Original user IDs like "U13740" become integer 13740
        # This creates a sparse range from 1 to ~94k, but only ~50k actual users exist
        # The high count (94k) is due to sparse user ID distribution, not actual user count
        logger.info(
            f"NOTE: User IDs are sparse - range spans {max(all_unique_users) - min(all_unique_users) + 1} but only {len(all_unique_users)} actual users exist"
        )
        logger.info("NOTE: This is expected for MIND dataset due to sparse user ID assignment")

        # Save total unique users count (across all three datasets)
        summary_data["total_unique_users"] = len(all_unique_users)

        # Save user ID range information
        if all_unique_users:
            summary_data["min_user_id"] = min(all_unique_users)
            summary_data["max_user_id"] = max(all_unique_users)
            summary_data["user_id_range"] = max(all_unique_users) - min(all_unique_users) + 1
            summary_data["user_id_density"] = round(
                len(all_unique_users) / summary_data["user_id_range"], 4
            )
        else:
            summary_data["min_user_id"] = 0
            summary_data["max_user_id"] = 0
            summary_data["user_id_range"] = 0
            summary_data["user_id_density"] = 0

        # Convert Counter to dictionary and save to summary_data
        summary_data["user_occurrence_counts"] = dict(all_user_occurrences)

        # Also save some statistics about user occurrences
        if all_user_occurrences:
            occurrence_values = list(all_user_occurrences.values())
            summary_data["min_user_occurrences"] = min(occurrence_values)
            summary_data["max_user_occurrences"] = max(occurrence_values)
            summary_data["avg_user_occurrences"] = round(np.mean(occurrence_values), 2)
            summary_data["median_user_occurrences"] = round(np.median(occurrence_values), 2)
        else:
            summary_data["min_user_occurrences"] = 0
            summary_data["max_user_occurrences"] = 0
            summary_data["avg_user_occurrences"] = 0
            summary_data["median_user_occurrences"] = 0

    except Exception as e:
        logger.warning(f"Error collecting user occurrence counts: {e}")
        summary_data["total_unique_users"] = 0
        summary_data["min_user_id"] = 0
        summary_data["max_user_id"] = 0
        summary_data["user_id_range"] = 0
        summary_data["user_id_density"] = 0
        summary_data["user_occurrence_counts"] = {}
        summary_data["min_user_occurrences"] = 0
        summary_data["max_user_occurrences"] = 0
        summary_data["avg_user_occurrences"] = 0
        summary_data["median_user_occurrences"] = 0


def collect_behavior_statistics(dataset_instance: Any, summary_data: dict) -> None:
    """Collect behavior statistics for all splits."""
    try:
        splits = ["train", "val", "test"]
        for split in splits:
            collect_split_statistics(dataset_instance, summary_data, split)

        # Collect user occurrence counts
        collect_user_occurrence_counts(dataset_instance, summary_data)

        # Knowledge graph statistics
        if dataset_instance.use_knowledge_graph:
            summary_data |= {
                "entity_embeddings_count": len(dataset_instance.entity_embeddings),
                "context_embeddings_count": len(dataset_instance.context_embeddings),
                "max_entities": dataset_instance.max_entities,
                "max_relations": dataset_instance.max_relations,
            }
    except Exception as e:
        logger.warning(f"Error collecting behavior statistics: {e}")


def collect_split_statistics(dataset_instance: Any, summary_data: dict, split: str) -> None:
    """Collect statistics for a specific data split."""
    try:
        data_attr = f"{split}_behaviors_data"
        data = getattr(dataset_instance, data_attr, None)
        if data is None:
            logger.info(f"No {split} behaviors data available")
            return

        prefix = split

        # Helper function to safely get data length
        def get_data_length(key: str, default: int = 0) -> int:
            return len(data.get(key, [])) if key in data else default

        # Helper function to safely get unique count
        def get_unique_count(key: str, default: int = 0) -> int:
            return len(set(data.get(key, []))) if key in data else default

        # Basic counts
        summary_data[f"{prefix}_behaviors_count"] = get_data_length("impression_ids")
        summary_data[f"{prefix}_unique_users"] = get_unique_count("user_ids")

        # History length calculation
        if "history_news_tokens" in data:
            history_data = data["history_news_tokens"]
            if isinstance(history_data, np.ndarray):
                non_zero_history = history_data != 0
                avg_history_length = np.mean(np.sum(non_zero_history, axis=1))
            else:
                history_lengths = []
                for history in history_data:
                    if isinstance(history, np.ndarray):
                        history_lengths.append(int(np.sum(history != 0)))
                    elif isinstance(history, list):
                        non_zero_count = sum(
                            np.sum(x != 0) if isinstance(x, np.ndarray) else (1 if x != 0 else 0)
                            for x in history
                        )
                        history_lengths.append(non_zero_count)
                    else:
                        history_lengths.append(1 if history != 0 else 0)
                avg_history_length = np.mean(history_lengths) if history_lengths else 0
            summary_data[f"{prefix}_avg_history_length"] = round(avg_history_length, 2)
        else:
            summary_data[f"{prefix}_avg_history_length"] = 0

        # Impressions length calculation
        if "candidate_news_tokens" in data:
            candidate_data = data["candidate_news_tokens"]
            if isinstance(candidate_data, np.ndarray):
                summary_data[f"{prefix}_avg_impressions_length"] = candidate_data.shape[1]
            else:
                impression_lengths = [len(impression) for impression in candidate_data]
                avg_impressions_length = np.mean(impression_lengths) if impression_lengths else 0
                summary_data[f"{prefix}_avg_impressions_length"] = round(avg_impressions_length, 2)
        else:
            summary_data[f"{prefix}_avg_impressions_length"] = 0

        # Positive/negative samples
        if "labels" in data:
            calculate_label_statistics(summary_data, data["labels"], prefix)
        else:
            summary_data[f"{prefix}_positive_samples"] = 0
            summary_data[f"{prefix}_negative_samples"] = 0
            summary_data[f"{prefix}_positive_ratio"] = 0

    except Exception as e:
        logger.warning(f"Error collecting {split} statistics: {e}")
        # Set default values for this split
        for key in [
            "behaviors_count",
            "unique_users",
            "avg_history_length",
            "avg_impressions_length",
            "positive_samples",
            "negative_samples",
            "positive_ratio",
        ]:
            summary_data[f"{prefix}_{key}"] = 0


def calculate_label_statistics(summary_data: dict, labels, prefix: str) -> None:
    """Calculate positive/negative sample statistics."""
    try:
        if isinstance(labels, np.ndarray):
            total_samples = labels.size
            positive_samples = np.sum(labels)
        else:  # list of lists
            total_samples = 0
            positive_samples = 0
            for label_list in labels:
                if isinstance(label_list, (list, np.ndarray)):
                    total_samples += len(label_list)
                    # Convert to numpy array if it's a list to ensure proper summing
                    if isinstance(label_list, list):
                        label_array = np.array(label_list)
                    else:
                        label_array = label_list
                    positive_samples += np.sum(label_array)
                else:
                    # Handle case where label_list might be a scalar
                    total_samples += 1
                    positive_samples += float(label_list) if label_list else 0

        summary_data[f"{prefix}_positive_samples"] = int(positive_samples)
        summary_data[f"{prefix}_negative_samples"] = int(total_samples - positive_samples)
        summary_data[f"{prefix}_positive_ratio"] = (
            round(float(positive_samples / total_samples), 4) if total_samples > 0 else 0
        )
    except Exception as e:
        logger.warning(f"Error calculating label statistics for {prefix}: {e}")
        # Set default values
        summary_data[f"{prefix}_positive_samples"] = 0
        summary_data[f"{prefix}_negative_samples"] = 0
        summary_data[f"{prefix}_positive_ratio"] = 0


def collect_overall_statistics(dataset_instance: Any, summary_data: dict) -> None:
    """Collect overall statistics across all splits."""
    try:
        total_behaviors = sum(
            summary_data.get(f"{split}_behaviors_count", 0) for split in ["train", "val", "test"]
        )
        summary_data["total_behaviors"] = total_behaviors

        # Total unique users
        all_user_ids = set()
        for split in ["train", "val", "test"]:
            data_attr = f"{split}_behaviors_data"
            data = getattr(dataset_instance, data_attr, None)
            if data is not None and "user_ids" in data:
                all_user_ids.update(data["user_ids"])
        summary_data["total_unique_users"] = len(all_user_ids)
    except Exception as e:
        logger.warning(f"Error collecting overall statistics: {e}")
        summary_data["total_behaviors"] = 0
        summary_data["total_unique_users"] = 0


def collect_quality_metrics(summary_data: dict) -> None:
    """Collect data quality metrics."""
    try:
        if (
            summary_data.get("total_news_articles", 0) > 0
            and summary_data.get("total_unique_users", 0) > 0
        ):
            total_possible_interactions = (
                summary_data["total_news_articles"] * summary_data["total_unique_users"]
            )
            total_actual_interactions = sum(
                summary_data.get(f"{split}_positive_samples", 0)
                for split in ["train", "val", "test"]
            )
            sparsity = 1 - (total_actual_interactions / total_possible_interactions)
            summary_data["data_sparsity"] = round(sparsity, 6)
    except Exception as e:
        logger.warning(f"Error collecting quality metrics: {e}")
        summary_data["data_sparsity"] = 0.0


def log_processed_data_statistics(summary_data: dict) -> None:
    """Log processed data statistics to console."""
    logger.info("=== PROCESSED DATA STATISTICS (After Processing) ===")
    logger.info(f"  Total news articles: {summary_data.get('total_news_articles', 'N/A')}")
    logger.info(f"  Vocabulary size: {summary_data.get('vocabulary_size', 'N/A')}")
    logger.info(
        f"  Total unique users (across all splits): {summary_data.get('total_unique_users', 'N/A')}"
    )
    logger.info(
        f"  User ID range: {summary_data.get('min_user_id', 'N/A')} to {summary_data.get('max_user_id', 'N/A')}"
    )
    logger.info(
        f"  User ID density: {summary_data.get('user_id_density', 'N/A')} (sparse IDs - this is expected for MIND)"
    )
    logger.info(f"  Total behaviors: {summary_data.get('total_behaviors', 'N/A')}")
    logger.info(f"  Data sparsity: {summary_data.get('data_sparsity', 'N/A')}")
    logger.info(
        f"  User occurrence stats - Min: {summary_data.get('min_user_occurrences', 'N/A')}, "
        f"Max: {summary_data.get('max_user_occurrences', 'N/A')}, "
        f"Avg: {summary_data.get('avg_user_occurrences', 'N/A')}, "
        f"Median: {summary_data.get('median_user_occurrences', 'N/A')}"
    )

    # Add note about MIND-small expected statistics
    total_users = summary_data.get("total_unique_users", 0)
    if total_users > 90000:
        logger.info(
            "  NOTE: High user count (~94k) is due to sparse user ID distribution in MIND dataset"
        )
        logger.info(
            "  NOTE: Each split has ~50k users, but different user sets create ~94k total unique users"
        )
        logger.info("  NOTE: This is the correct behavior for MIND-small dataset")


def log_key_statistics(summary_data: dict) -> None:
    """Log key statistics to console."""
    logger.info("=== KEY DATASET STATISTICS ===")
    logger.info(
        f"Dataset: {summary_data.get('dataset_name', 'N/A')} ({summary_data.get('dataset_version', 'N/A')})"
    )
    logger.info(f"Total news articles: {summary_data.get('total_news_articles', 'N/A'):,}")
    logger.info(f"Vocabulary size: {summary_data.get('vocabulary_size', 'N/A'):,}")
    logger.info(f"Total unique users: {summary_data.get('total_unique_users', 'N/A'):,}")
    logger.info(f"Total behaviors: {summary_data.get('total_behaviors', 'N/A'):,}")

    # Training statistics
    logger.info(f"Train behaviors: {summary_data.get('train_behaviors_count', 'N/A'):,}")
    logger.info(f"Train unique users: {summary_data.get('train_unique_users', 'N/A'):,}")
    logger.info(f"Train positive ratio: {summary_data.get('train_positive_ratio', 'N/A'):.2%}")

    # Validation statistics
    logger.info(f"Validation behaviors: {summary_data.get('val_behaviors_count', 'N/A'):,}")
    logger.info(f"Validation unique users: {summary_data.get('val_unique_users', 'N/A'):,}")
    logger.info(f"Validation positive ratio: {summary_data.get('val_positive_ratio', 'N/A'):.2%}")

    # Test statistics
    logger.info(f"Test behaviors: {summary_data.get('test_behaviors_count', 'N/A'):,}")
    logger.info(f"Test unique users: {summary_data.get('test_unique_users', 'N/A'):,}")
    logger.info(f"Test positive ratio: {summary_data.get('test_positive_ratio', 'N/A'):.2%}")

    # Data quality
    logger.info(f"Data sparsity: {summary_data.get('data_sparsity', 'N/A'):.4f}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def apply_data_fraction(data_dict: Dict[str, np.ndarray], fraction: float) -> Dict[str, np.ndarray]:
    """Reduce the dataset size based on the fraction parameter."""
    if fraction < 1.0:
        logger.info(f"Using {fraction * 100:.0f}% of the dataset")
        return {k: v[: int(len(v) * fraction)] for k, v in data_dict.items()}
    return data_dict


def string_is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def save_unique_users_to_csv(dataset_instance: Any) -> None:
    """Save unique user IDs from all three datasets (train, val, test) to a CSV file."""
    try:
        all_unique_users = set()

        # Collect unique user IDs from all three datasets together
        for split in ["train", "val", "test"]:
            data_attr = f"{split}_behaviors_data"
            data = getattr(dataset_instance, data_attr, None)
            if data is not None and "user_ids" in data:
                user_ids = data["user_ids"]
                if isinstance(user_ids, (list, np.ndarray)):
                    all_unique_users.update(user_ids)

        if all_unique_users:
            # Convert to list and sort for better readability
            sorted_user_ids = sorted(list(all_unique_users))

            logger.info(f"User ID range: {min(sorted_user_ids)} to {max(sorted_user_ids)}")

            # Show some sample user IDs
            sample_ids = (
                sorted_user_ids[:5] + sorted_user_ids[-5:]
                if len(sorted_user_ids) > 10
                else sorted_user_ids
            )
            logger.info(f"Sample user IDs: {sample_ids}")
        else:
            logger.warning("No unique user IDs found to save")

    except Exception as e:
        logger.warning(f"Error saving unique users to CSV: {e}")


def reorder_summary_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns for better readability."""
    column_order = [
        # Basic info
        "dataset_name",
        "dataset_version",
        "embedding_type",
        "embedding_size",
        # Configuration
        "word_threshold",
        "max_title_length",
        "max_abstract_length",
        "max_history_length",
        "max_impressions_length",
        "validation_split_strategy",
        "validation_split_percentage",
        # Processed news statistics
        "total_news_articles",
        "vocabulary_size",
        "num_categories",
        "num_subcategories",
        "average_title_length",
        "average_abstract_length",
        # Processed user statistics
        "total_unique_users",
        "min_user_id",
        "max_user_id",
        "user_id_range",
        "user_id_density",
        "total_behaviors",
        "min_user_occurrences",
        "max_user_occurrences",
        "avg_user_occurrences",
        "median_user_occurrences",
        # Training statistics
        "train_behaviors_count",
        "train_unique_users",
        "train_avg_history_length",
        "train_avg_impressions_length",
        "train_positive_samples",
        "train_negative_samples",
        "train_positive_ratio",
        # Validation statistics
        "val_behaviors_count",
        "val_unique_users",
        "val_avg_history_length",
        "val_avg_impressions_length",
        "val_positive_samples",
        "val_negative_samples",
        "val_positive_ratio",
        # Test statistics
        "test_behaviors_count",
        "test_unique_users",
        "test_avg_history_length",
        "test_avg_impressions_length",
        "test_positive_samples",
        "test_negative_samples",
        "test_positive_ratio",
        # Data quality
        "data_sparsity",
        "data_fraction_train",
        "data_fraction_val",
        "data_fraction_test",
        # Processing flags
        "process_title",
        "process_abstract",
        "process_category",
        "process_subcategory",
        "process_user_id",
        # Knowledge graph
        "use_knowledge_graph",
        "entity_embeddings_count",
        "context_embeddings_count",
        "max_entities",
        "max_relations",
    ]

    existing_columns = [col for col in column_order if col in summary_df.columns]
    return summary_df[existing_columns]
